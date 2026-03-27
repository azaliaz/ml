# cvat_client.py
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Union, Any
import json
import requests

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from cvat_sdk.api_client.exceptions import ApiException

import certifi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- config ----------
os.environ["SSL_CERT_FILE"] = certifi.where()
# По умолчанию теперь ожидаем локальный CVAT (docker-compose по умолчанию поднимает на 8080).
# При желании можно переопределить через переменную окружения CVAT_URL.
CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080")
ACCESS_TOKEN = os.environ.get("CVAT_TOKEN")
CVAT_INSECURE = os.environ.get("CVAT_INSECURE", "").lower() in ("1", "true", "yes")
USE_BEARER = os.environ.get("CVAT_TOKEN_BEARER", "").lower() in ("1", "true", "yes")

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cvat_client")

if not ACCESS_TOKEN:
    raise RuntimeError("Set CVAT_TOKEN environment variable before running (export CVAT_TOKEN=...)")

# ---------- helpers ----------
def _wait_for_task_size(client, task_id: int, expected_count: int, timeout: int = 300, poll_interval: float = 2.0) -> bool:
    start = time.time()
    while True:
        task = client.tasks.retrieve(task_id)
        size = getattr(task, "size", None)
        try:
            size_int = int(size) if size is not None else 0
        except Exception:
            size_int = 0
        if size_int >= expected_count and expected_count > 0:
            return True
        if time.time() - start > timeout:
            return False
        time.sleep(poll_interval)

def create_task_and_upload(local_files: List[str], task_name: str = "streamlit_task", labels: Optional[List[dict]] = None, timeout: int = 600) -> int:
    if not local_files:
        raise ValueError("local_files must contain at least one path")

    logger.info("Creating CVAT task '%s' with %d resources", task_name, len(local_files))
    with make_client(CVAT_URL, access_token=ACCESS_TOKEN) as client:
        # 1) create task
        task = client.tasks.create(spec={"name": task_name, "labels": labels or [{"name": "object"}]})

        # 2) upload files (SDK streams files to server)
        try:
            # SDK accepts list[str] paths for streaming upload
            task.upload_data(resource_type=ResourceType.LOCAL, resources=local_files)
        except Exception as e:
            logger.exception("task.upload_data() failed: %s", e)
            raise RuntimeError(f"Failed to upload data via SDK: {e}") from e

        # optionally wait for size
        task_id = getattr(task, "id", None)
        if task_id is None:
            raise RuntimeError("Failed to get task id after create/upload")
        ok = _wait_for_task_size(client, task_id, expected_count=len(local_files), timeout=timeout)
        if not ok:
            logger.warning("Task %s not ready after %ss (size may be different). Proceeding anyway.", task_id, timeout)
        return int(task_id)
# вставьте это вместо старой export_annotations_by_id
from urllib.parse import quote_plus


def _auth_headers() -> dict:
    if USE_BEARER:
        return {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    else:
        return {"Authorization": f"Token {ACCESS_TOKEN}"}


def _http_get_paged(session: requests.Session, url: str, params: Optional[dict] = None, timeout: int = 30) -> List[dict]:
    """
    CVAT list endpoints are typically paginated (DRF): {count, next, previous, results}.
    This helper returns concatenated 'results' (or raw list if non-paginated).
    """
    out: List[dict] = []
    next_url: Optional[str] = url
    next_params = dict(params or {})
    while next_url:
        resp = session.get(next_url, params=next_params, timeout=timeout, verify=(not CVAT_INSECURE))
        if resp.status_code != 200:
            raise RuntimeError(f"GET {next_url} failed: status={resp.status_code} body={resp.text[:2000]}")
        try:
            body = resp.json()
        except Exception as e:
            raise RuntimeError(f"GET {next_url} returned non-JSON: {e}; body={resp.text[:2000]}")

        if isinstance(body, list):
            out.extend(body)
            break
        if isinstance(body, dict) and "results" in body:
            results = body.get("results") or []
            if isinstance(results, list):
                out.extend(results)
            next_url = body.get("next")
            next_params = {}
            continue
        raise RuntimeError(f"Unexpected list response schema from {next_url}: {type(body)} keys={list(body.keys()) if isinstance(body, dict) else None}")
    return out


def resolve_cvat_user_id(user: Union[int, str]) -> int:
    """
    Resolve CVAT user id by:
    - passing through int
    - searching by username/email (string) using /api/users with several fallbacks
    Returns integer user id or raises RuntimeError with helpful diagnostics.
    """
    if isinstance(user, int):
        return int(user)
    query = str(user).strip()
    if not query:
        raise ValueError("user must be a non-empty username/email or an integer id")
    # Allow passing numeric id as a string from UI/forms.
    if query.isdigit():
        return int(query)

    url_base = CVAT_URL.rstrip("/")
    session = _build_requests_session()
    session.headers.update(_auth_headers())

    # Important: be strict.
    # Do NOT silently pick the "first user" candidate — it can lead to assigning yourself
    # (common case when token can't list users / emails are hidden).
    is_email_query = "@" in query

    # Try several server-side filters first (search, username, email)
    attempts = [
        {"params": {"search": query, "page_size": 100}, "desc": "search"},
        {"params": {"username": query, "page_size": 100}, "desc": "username exact"},
        {"params": {"email": query, "page_size": 100}, "desc": "email exact"},
    ]

    all_candidates: List[dict] = []
    for a in attempts:
        try:
            users = _http_get_paged(session, f"{url_base}/api/users", params=a["params"])
        except Exception as e:
            # don't abort immediately — try next strategy
            logger.debug("User lookup attempt (%s) failed: %s", a["desc"], e)
            users = []
        if users:
            logger.debug("User lookup (%s) returned %d candidates", a["desc"], len(users))
            all_candidates.extend(users)
            # try to find exact match among returned candidates
            qlow = query.lower()
            for u in users:
                uname = str(u.get("username", "") or "").strip().lower()
                email = str(u.get("email", "") or "").strip().lower()
                if is_email_query:
                    if email and qlow == email:
                        logger.info(
                            "Resolved user by %s exact email match: %s (id=%s)",
                            a["desc"],
                            u.get("username"),
                            u.get("id"),
                        )
                        return int(u["id"])
                else:
                    if uname and qlow == uname:
                        logger.info(
                            "Resolved user by %s exact username match: %s (id=%s)",
                            a["desc"],
                            u.get("username"),
                            u.get("id"),
                        )
                        return int(u["id"])
            # do not pick a random candidate
            # continue other strategies / fallbacks

    # Fallback: try fetching first page of all users and filter locally for contains
    try:
        users_all = _http_get_paged(session, f"{url_base}/api/users", params={"page_size": 200})
    except Exception as e:
        users_all = []
        logger.debug("Fetching /api/users without filters failed: %s", e)

    if users_all:
        qlow = query.lower()
        matched: List[dict] = []
        for u in users_all:
            uname = str(u.get("username", "") or "").strip().lower()
            email = str(u.get("email", "") or "").strip().lower()
            if is_email_query:
                if email and qlow == email:
                    logger.info("Resolved user by local exact email match: %s (id=%s)", u.get("username"), u.get("id"))
                    return int(u["id"])
                # allow "contains" only as diagnostics, not auto-pick
                if qlow in email:
                    matched.append(u)
            else:
                if uname and qlow == uname:
                    logger.info("Resolved user by local exact username match: %s (id=%s)", u.get("username"), u.get("id"))
                    return int(u["id"])
                if qlow in uname:
                    matched.append(u)

    # If we have some candidates from earlier attempts but couldn't be confident — show diagnostic
    # build brief candidates summary (username/email pairs) from all_candidates and users_all
    candidates = all_candidates or users_all or []
    cand_summary = []
    for u in candidates[:10]:
        cand_summary.append(f"{u.get('username') or '<no-username>'} <{u.get('email') or ''}> (id={u.get('id')})")
    cand_count = len(candidates)

    # Helpful error message
    msg_lines = [
        f"Could not resolve CVAT user id for '{query}'. Candidates returned: {cand_count}.",
        "Searched by: search, username, email and local exact-match fallback.",
    ]
    if cand_summary:
        msg_lines.append("Sample candidates (up to 10):")
        msg_lines.extend(cand_summary)
    msg_lines.append(
        "Possible reasons: user does not exist in CVAT; you used email but CVAT API doesn't expose emails; "
        "or the token lacks permission to list users (use admin token / org owner)."
    )
    msg_lines.append(
        "Suggestions: use CVAT username instead of email, or pass the numeric user id directly (recommended), "
        "or use a token with permission to list users."
    )
    raise RuntimeError(" ".join(msg_lines))


def grant_validation_access_for_task(task_id: int, reviewer_user: Union[int, str], *, timeout: int = 30) -> dict:
    """
    Assign a user as validator/reviewer for the task you created.

    CVAT differs by version/config: some builds use job field `reviewer`, others rely on
    `assignee` + `stage`/`state`. This function tries multiple approaches and verifies
    the result by re-reading the job.
    """
    task_id_i = int(task_id)

    try:
        reviewer_id = resolve_cvat_user_id(reviewer_user)
    except Exception as e:
        raise RuntimeError(f"Failed to resolve user '{reviewer_user}': {e}") from e

    url_base = CVAT_URL.rstrip("/")
    session = _build_requests_session()
    session.headers.update(_auth_headers())

    # Get jobs for the task (try several endpoints/filters across CVAT versions)
    jobs: List[dict] = []
    job_list_attempt_errors: List[str] = []
    list_attempts = [
        ("jobs?task_id", f"{url_base}/api/jobs", {"task_id": task_id_i, "page_size": 100}),
        ("jobs?task", f"{url_base}/api/jobs", {"task": task_id_i, "page_size": 100}),
        ("tasks/{id}/jobs", f"{url_base}/api/tasks/{task_id_i}/jobs", {"page_size": 100}),
        ("tasks/{id}/jobs(no params)", f"{url_base}/api/tasks/{task_id_i}/jobs", None),
    ]
    for desc, url, params in list_attempts:
        try:
            jobs = _http_get_paged(session, url, params=params, timeout=timeout)
            if jobs:
                logger.info("Found %d jobs for task %s via %s", len(jobs), task_id_i, desc)
                break
        except Exception as e:
            job_list_attempt_errors.append(f"{desc}: {e}")
            jobs = []

    if not jobs:
        raise RuntimeError(
            f"No jobs found for task_id={task_id_i}. "
            f"Tried: {', '.join([a[0] for a in list_attempts])}. "
            f"Errors: {' | '.join(job_list_attempt_errors)[:1500]}. "
            "Is the task created and accessible to the token user?"
        )

    patched: List[int] = []
    failed: List[dict] = []

    for j in jobs:
        job_id = j.get("id")
        if job_id is None:
            continue

        job_id_i = int(job_id)
        patch_url = f"{url_base}/api/jobs/{job_id_i}"

        # Try several payload variants; CVAT API differs by version/config.
        payload_variants: List[dict] = [
            {"reviewer": reviewer_id},
            {"reviewer_id": reviewer_id},
            {"assignee": reviewer_id, "stage": "validation"},
            {"assignee": reviewer_id, "state": "validation"},
            {"assignee": reviewer_id, "stage": "review"},
            {"assignee": reviewer_id},
        ]

        last_attempt: Optional[dict] = None
        last_resp_info: Optional[dict] = None

        def _job_has_reviewer_or_validation(job_json: dict) -> bool:
            reviewer = job_json.get("reviewer")
            assignee = job_json.get("assignee")
            stage = job_json.get("stage") or job_json.get("state")
            rid = None
            if isinstance(reviewer, dict):
                rid = reviewer.get("id")
            elif isinstance(reviewer, int):
                rid = reviewer
            # accept either reviewer match, or assignee match + validation stage/state
            assignee_id = assignee.get("id") if isinstance(assignee, dict) else assignee if isinstance(assignee, int) else None
            if rid is not None and int(rid) == reviewer_id:
                return True
            if assignee_id is not None and int(assignee_id) == reviewer_id and str(stage or "").lower() in ("validation", "review"):
                return True
            return False

        # first, try to PATCH and verify for each payload
        for payload in payload_variants:
            last_attempt = payload
            try:
                resp = session.patch(
                    patch_url,
                    json=payload,
                    timeout=(10, timeout),
                    verify=(not CVAT_INSECURE),
                )
            except Exception as e:
                last_resp_info = {"exception": str(e)}
                continue

            body_snippet = ""
            try:
                body_snippet = (resp.text or "")[:2000]
            except Exception:
                body_snippet = "<unreadable body>"
            last_resp_info = {"status_code": resp.status_code, "body": body_snippet}

            if resp.status_code not in (200, 204):
                continue

            # verify by reading job back
            try:
                check = session.get(
                    patch_url,
                    timeout=(10, timeout),
                    verify=(not CVAT_INSECURE),
                )
                if check.status_code == 200:
                    jb = check.json()
                    if _job_has_reviewer_or_validation(jb):
                        patched.append(job_id_i)
                        last_resp_info = None
                        last_attempt = None
                        break
                    last_resp_info = {
                        "status_code": resp.status_code,
                        "body": body_snippet,
                        "verify": {
                            "reviewer": jb.get("reviewer"),
                            "assignee": jb.get("assignee"),
                            "stage": jb.get("stage"),
                            "state": jb.get("state"),
                        },
                    }
            except Exception as e:
                last_resp_info = {
                    "status_code": resp.status_code,
                    "body": body_snippet,
                    "verify_exception": str(e),
                }

        if job_id_i in patched:
            continue

        failed.append(
            {
                "job_id": job_id_i,
                "attempted_payload": last_attempt,
                "last_response": last_resp_info,
            }
        )

    return {
        "task_id": task_id_i,
        "reviewer_id": reviewer_id,
        "jobs_total": len(jobs),
        "jobs_patched": patched,
        "jobs_failed": failed,
    }
def export_annotations_by_id(task_id: int, out_zip_path: str, format_name: str = "COCO 1.0",
                             include_images: bool = False, timeout: int = 300) -> str:
    out_path = Path(out_zip_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Export request task=%s -> %s (format=%s, include_images=%s)", task_id, out_path, format_name, include_images)

    # If file exists already - return (avoids re-export collisions in UI)
    if out_path.exists() and out_path.is_file():
        logger.info("Export file already exists locally: %s — returning existing file", out_path)
        return str(out_path)

    # 1) SDK path (best effort)
    try:
        with make_client(CVAT_URL, access_token=ACCESS_TOKEN) as client:
            try:
                task = client.tasks.retrieve(task_id)
            except ApiException as e:
                logger.warning("SDK failed to retrieve task %s: %s", task_id, e)
                raise

            try:
                logger.info("Calling SDK task.export_dataset(...)")
                task.export_dataset(format_name, str(out_path), include_images=include_images)
            except Exception as e:
                logger.warning("SDK export_dataset failed: %s", e)
                raise

            # wait short time for file to appear
            waited = 0.0
            poll_interval = 0.5
            while waited < timeout:
                if out_path.exists() and out_path.stat().st_size > 0:
                    logger.info("Export finished, file present: %s", out_path)
                    return str(out_path)
                time.sleep(poll_interval)
                waited += poll_interval
            raise RuntimeError(f"Export via SDK did not produce output file {out_path} within {timeout}s")
    except Exception as sdk_exc:
        logger.info("SDK export unavailable/failed: %s — will try HTTP fallback", sdk_exc)

    # 2) HTTP fallback: try several variants
    url_base = CVAT_URL.rstrip("/")
    headers = {"Accept": "application/zip"}
    if USE_BEARER:
        headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
    else:
        headers["Authorization"] = f"Token {ACCESS_TOKEN}"

    session = _build_requests_session()
    session.headers.update(headers)

    # Prepare list of attempts (method, url, params, data)
    encoded = quote_plus(format_name)
    attempts = [
        ("GET", f"{url_base}/api/tasks/{task_id}/annotations", {"action": "export", "format": format_name}, None),
        ("GET", f"{url_base}/api/tasks/{task_id}/annotations", {"format": format_name}, None),
        # some CVAT builds accept url-encoded format in path/query
        ("GET", f"{url_base}/api/tasks/{task_id}/annotations?action=export&format={encoded}", None, None),
        # POST with form-data (some backends prefer POST)
        ("POST", f"{url_base}/api/tasks/{task_id}/annotations", None, {"format": format_name, "action": "export"}),
        ("POST", f"{url_base}/api/tasks/{task_id}/annotations?action=export", None, {"format": format_name}),
    ]

    last_err = None
    for method, url, params, data in attempts:
        try:
            logger.info("HTTP export attempt: %s %s params=%s data=%s", method, url, params, data)
            if method == "GET":
                resp = session.get(url, params=params, timeout=60, stream=True, verify=(not CVAT_INSECURE))
            else:
                resp = session.post(url, params=params, data=data, timeout=60, stream=True, verify=(not CVAT_INSECURE))
        except Exception as e:
            logger.debug("Request exception for %s %s: %s", method, url, e)
            last_err = e
            continue

        body_snippet = None
        try:
            body_snippet = resp.text[:2000]
        except Exception:
            body_snippet = "<no body>"

        logger.info("HTTP resp %s %s -> status=%s headers=%s body_snippet=%s", method, url, resp.status_code, dict(resp.headers), body_snippet)

        if resp.status_code in (200, 201):
            try:
                with open(out_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                if out_path.exists() and out_path.stat().st_size > 0:
                    logger.info("Saved export to %s (HTTP)", out_path)
                    return str(out_path)
                else:
                    last_err = RuntimeError("Downloaded file empty")
                    continue
            except Exception as e:
                logger.exception("Failed to write export to file %s: %s", out_path, e)
                last_err = e
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                continue

        # 202 -> asynchronous: try to obtain request id then poll + download
        if resp.status_code == 202:
            # try JSON body for id
            rq_id = None
            try:
                body = resp.json()
                # common keys: id, request_id, rq_id
                for k in ("id", "request_id", "rq_id"):
                    if k in body:
                        rq_id = int(body[k])
                        break
            except Exception:
                body = None

            # location header maybe present
            if rq_id is None:
                loc = resp.headers.get("Location") or resp.headers.get("location")
                if loc:
                    # sometimes Location ends with /api/requests/{id}
                    try:
                        candidate = loc.rstrip("/").split("/")[-1]
                        rq_id = int(candidate)
                    except Exception:
                        rq_id = None

            if rq_id is not None:
                logger.info("Received async export rq_id=%s, polling...", rq_id)
                # poll status then try download endpoint
                try:
                    start = time.time()
                    while time.time() - start < timeout:
                        st_url = f"{url_base}/api/requests/{rq_id}"
                        r = session.get(st_url, timeout=30, verify=(not CVAT_INSECURE))
                        if r.status_code == 200:
                            try:
                                jb = r.json()
                            except Exception:
                                jb = {}
                            status = jb.get("status") or jb.get("state") or jb.get("result")
                            logger.debug("Request %s status: %s", rq_id, status)
                            if status in ("completed", "success", "finished", "ok"):
                                # try to download: /api/requests/{id}/download
                                dl_url = f"{url_base}/api/requests/{rq_id}/download"
                                logger.info("Attempting to download async result from %s", dl_url)
                                rdl = session.get(dl_url, timeout=60, stream=True, verify=(not CVAT_INSECURE))
                                if rdl.status_code in (200, 201):
                                    with open(out_path, "wb") as fh:
                                        for chunk in rdl.iter_content(chunk_size=8192):
                                            if chunk:
                                                fh.write(chunk)
                                    if out_path.exists() and out_path.stat().st_size > 0:
                                        logger.info("Saved async export result to %s", out_path)
                                        return str(out_path)
                                else:
                                    logger.warning("Async download returned status %s", rdl.status_code)
                                    # maybe response contains direct link in body
                                    try:
                                        logger.debug("async download body: %s", rdl.text[:1000])
                                    except Exception:
                                        pass
                                # if failed - break and continue attempts
                                break
                            if status in ("failed", "error"):
                                raise RuntimeError(f"Export request {rq_id} failed: {jb}")
                        else:
                            logger.debug("Polling request returned status %s", r.status_code)
                        time.sleep(2.0)
                except Exception as e:
                    logger.warning("Async export polling/download failed: %s", e)
                    last_err = e
                    continue

        # otherwise, keep trying other variants
        last_err = RuntimeError(f"HTTP export attempt returned status {resp.status_code}: {body_snippet}")
        continue

    # if reached here - no successful attempts
    logger.error("All HTTP export attempts failed; last error: %s", last_err)
    raise RuntimeError(f"HTTP export attempts failed, last checked file: {out_path}") from last_err
def _build_requests_session(retries: int = 3, backoff_factor: float = 0.5, status_forcelist=(500, 502, 503, 504)) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _http_post_import_annotations(task_id: int, annotations_path: str, format_name: str, timeout: int = 600) -> Optional[Union[int, None]]:
    if not format_name or not str(format_name).strip():
        raise ValueError("format_name must be provided (e.g. 'COCO 1.0')")

    url_base = CVAT_URL.rstrip("/")
    # пробуем сначала импорт (recommended), затем просто /annotations без action, затем /dataset (legacy)
    upload_urls_with_params = [
        (f"{url_base}/api/tasks/{task_id}/annotations", {"action": "import", "format": format_name}),
        (f"{url_base}/api/tasks/{task_id}/annotations", {"format": format_name}),
        (f"{url_base}/api/tasks/{task_id}/dataset", None),
    ]

    headers = {}
    if USE_BEARER:
        headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
    else:
        headers["Authorization"] = f"Token {ACCESS_TOKEN}"

    files_field_candidates = ["annotation_file", "file", "data", "annotations"]
    verify_ssl = not CVAT_INSECURE
    session = _build_requests_session()

    last_resp_info = []
    for url, params in upload_urls_with_params:
        for field in files_field_candidates:
            try:
                with open(annotations_path, "rb") as fh:
                    multipart = {field: (Path(annotations_path).name, fh, "application/zip")}
                    # keep filename in form-data as well (some CVAT backends read it from form)
                    data = {"filename": Path(annotations_path).name}
                    logger.info("Attempting HTTP upload to %s (field=%s) params=%s", url, field, params)
                    # pass params (query string) and data (form fields) + files
                    resp = session.post(url, headers=headers, params=params, files=multipart, data=data, timeout=timeout, verify=verify_ssl)
            except requests.exceptions.SSLError:
                logger.exception("SSL error while uploading annotations to %s", url)
                raise
            except Exception as e:
                logger.debug("HTTP upload attempt failed for %s (field=%s): %s", url, field, e)
                last_resp_info.append((url, field, "exception", str(e)))
                continue

            # Log response details for debugging
            try:
                body_text = resp.text[:2000]  # limit
            except Exception:
                body_text = "<unable to read resp.text>"

            logger.info("Upload response from %s (field=%s): status=%s headers=%s body_snippet=%s", url, field, resp.status_code, dict(resp.headers), body_text)
            last_resp_info.append((url, field, resp.status_code, body_text))

            try:
                if resp.status_code in (200, 201, 202):
                    try:
                        body = resp.json()
                    except Exception:
                        body = None

                    rq_id = None
                    if isinstance(body, dict):
                        for key in ("id", "rq_id", "request_id", "requestId", "result"):
                            if key in body and isinstance(body[key], (int, str)):
                                try:
                                    rq_id = int(body[key])
                                    break
                                except Exception:
                                    pass

                    if rq_id is None:
                        loc = resp.headers.get("Location") or resp.headers.get("location")
                        if loc:
                            try:
                                candidate = loc.rstrip("/").split("/")[-1]
                                rq_id = int(candidate)
                            except Exception:
                                rq_id = None

                    if resp.status_code in (200, 201) and rq_id is None:
                        logger.info("Synchronous import success (status=%s) at %s", resp.status_code, url)
                        return None
                    if resp.status_code == 202 and rq_id is not None:
                        logger.info("Asynchronous import accepted (rq_id=%s) at %s", rq_id, url)
                        return rq_id
                    if resp.status_code == 202 and rq_id is None:
                        logger.info("Asynchronous import accepted (no rq_id) at %s", url)
                        return None
                else:
                    logger.debug("Upload returned status %s for %s (field=%s)", resp.status_code, url, field)
                    continue
            except Exception:
                logger.exception("Error parsing upload response from %s", url)
                continue

    logger.error("All HTTP upload attempts failed; samples: %s", last_resp_info[-5:])
    raise RuntimeError(f"HTTP import attempts failed: could not upload annotations via direct API (no usable response). Last attempts: {last_resp_info[-5:]}")
def _poll_request_status_http(rq_id: int, timeout: int = 600, poll_interval: float = 2.0) -> str:
    url = f"{CVAT_URL.rstrip('/')}/api/requests/{rq_id}"
    headers = {}
    if USE_BEARER:
        headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
    else:
        headers["Authorization"] = f"Token {ACCESS_TOKEN}"

    start = time.time()
    while True:
        resp = requests.get(url, headers=headers, timeout=30, verify=(not CVAT_INSECURE))
        if resp.status_code == 200:
            try:
                body = resp.json()
            except Exception:
                body = None
            status = None
            if isinstance(body, dict):
                status = body.get("status") or body.get("state") or body.get("result")
            if status in ("completed", "success", "finished", "ok"):
                logger.info("Import request %s completed (status=%s)", rq_id, status)
                return status
            if status in ("failed", "error"):
                logger.error("Import request %s failed (status=%s)", rq_id, status)
                raise RuntimeError(f"Import request {rq_id} failed (status={status})")
        else:
            logger.debug("Polling request %s returned status %s", rq_id, resp.status_code)
        if time.time() - start > timeout:
            raise TimeoutError(f"Import request {rq_id} not finished after {timeout}s")
        time.sleep(poll_interval)

def import_annotations_to_task(task_id: int, annotations_path: str, format_name: str = "COCO 1.0", timeout: int = 600) -> Optional[int]:
    """
    Try SDK import first, then fallback to HTTP endpoints.
    Returns request id if asynchronous import started; None if synchronous import or immediate success.
    """
    annotations_path = str(annotations_path)
    if not Path(annotations_path).exists():
        raise FileNotFoundError(f"annotations file not found: {annotations_path}")

    logger.info("Importing annotations %s into task %s (format=%s)", annotations_path, task_id, format_name)

    # First: SDK path (best-effort, handles async vs sync for many CVAT versions)
    try:
        with make_client(CVAT_URL, access_token=ACCESS_TOKEN) as client:
            try:
                # try modern signature first
                resp = client.tasks.create_annotations(id=task_id, filename=Path(annotations_path).name, format=format_name, annotation_file_request=open(annotations_path, "rb"))
            except TypeError:
                # fallback to older signature forms
                try:
                    resp = client.tasks.create_annotations(task_id, Path(annotations_path).name, format_name, open(annotations_path, "rb"))
                except AttributeError as e:
                    logger.exception("SDK create_annotations signatures exhausted: %s", e)
                    raise e
            rq_id = None
            if hasattr(resp, "id"):
                try:
                    rq_id = int(getattr(resp, "id"))
                except Exception:
                    rq_id = None
            elif isinstance(resp, dict):
                for key in ("id", "rq_id", "request_id"):
                    if key in resp:
                        try:
                            rq_id = int(resp[key])
                            break
                        except Exception:
                            pass

            if rq_id:
                # poll SDK request status if available
                try:
                    start = time.time()
                    while True:
                        rq = client.requests.retrieve(rq_id)
                        status = getattr(rq, "status", None)
                        if status in ("completed", "success"):
                            logger.info("SDK import request %s completed", rq_id)
                            return rq_id
                        if status in ("failed", "error"):
                            logger.error("SDK import request %s failed", rq_id)
                            raise RuntimeError(f"Import request {rq_id} failed: {getattr(rq, 'message', '')}")
                        if time.time() - start > timeout:
                            raise TimeoutError(f"Import request {rq_id} not finished after {timeout}s")
                        time.sleep(2.0)
                except Exception:
                    # if polling failed, still return rq_id to caller (they can inspect)
                    return rq_id
            else:
                # SDK returned sync success
                logger.info("SDK import returned synchronous success (no rq_id)")
                return None

    except AttributeError as e:
        logger.debug("SDK path raised AttributeError (falling back to HTTP): %s", e)
    except Exception as e:
        if isinstance(e, requests.exceptions.SSLError):
            logger.exception("SSL error during SDK import: %s", e)
            raise
        logger.warning("SDK import failed, falling back to HTTP: %s", e)

    # HTTP fallback
    try:
        rq = _http_post_import_annotations(task_id, annotations_path, format_name, timeout=timeout)
        if isinstance(rq, int):
            try:
                _poll_request_status_http(rq, timeout=timeout)
                return rq
            except Exception:
                return rq
        else:
            return None
    except requests.exceptions.SSLError as ssl_err:
        logger.exception("SSL error while importing annotations over HTTP: %s", ssl_err)
        raise RuntimeError(f"SSL error while importing annotations: {ssl_err}") from ssl_err
    except Exception as e:
        logger.exception("HTTP import failed: %s", e)
        raise RuntimeError(f"Import failed (both SDK and HTTP fallback attempts failed): {e}") from e