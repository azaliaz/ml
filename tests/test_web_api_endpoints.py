import importlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_web_api_with_fake_cvat():
    fake = types.ModuleType("cvat_client")
    fake.create_task_and_upload = mock.Mock(return_value=123)
    fake.export_annotations_by_id = mock.Mock(side_effect=RuntimeError("boom export"))
    fake.grant_validation_access_for_task = mock.Mock(return_value={"ok": True})
    fake.import_annotations_to_task = mock.Mock(return_value=None)

    sys.modules["cvat_client"] = fake
    if "web_api" in sys.modules:
        del sys.modules["web_api"]
    return importlib.import_module("web_api")


class TestWebApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.web_api = _load_web_api_with_fake_cvat()
        self.client = TestClient(self.web_api.app)
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.web_api.UPLOAD_DIR = root / "uploads"
        self.web_api.EXPORT_DIR = root / "exports"
        self.web_api.PREANN_DIR = self.web_api.UPLOAD_DIR / "preannotations"
        self.web_api.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.web_api.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.web_api.PREANN_DIR.mkdir(parents=True, exist_ok=True)
        self.web_api._PREANN_STATUS.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("ml_service_url", body)

    def test_preannotation_status_not_found(self):
        resp = self.client.get("/api/tasks/777/preannotation-status")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("No preannotation status", resp.json().get("detail", ""))

    def test_preannotation_status_success(self):
        self.web_api._set_status(42, {"status": "done", "phase": "done"})
        resp = self.client.get("/api/tasks/42/preannotation-status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "done")

    def test_grant_validation_requires_reviewer(self):
        resp = self.client.post("/api/tasks/1/grant-validation", json={"reviewer_user": "  "})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("reviewer_user is required", resp.json().get("detail", ""))

    def test_grant_validation_success(self):
        self.web_api.grant_validation_access_for_task = mock.Mock(return_value={"jobs_patched": [1], "jobs_failed": []})
        resp = self.client.post("/api/tasks/55/grant-validation", json={"reviewer_user": "alice"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["jobs_patched"], [1])
        self.web_api.grant_validation_access_for_task.assert_called_once_with(task_id=55, reviewer_user="alice", timeout=120)

    def test_grant_validation_maps_internal_error_to_500(self):
        self.web_api.grant_validation_access_for_task = mock.Mock(side_effect=RuntimeError("boom"))
        resp = self.client.post("/api/tasks/55/grant-validation", json={"reviewer_user": "alice"})
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json().get("detail"), "boom")

    def test_ml_health_ok(self):
        fake_resp = mock.Mock(status_code=200)
        fake_resp.json.return_value = {"alive": True}
        with mock.patch.object(self.web_api.requests, "get", return_value=fake_resp):
            resp = self.client.get("/api/ml/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True, "error": None, "data": {"alive": True}})

    def test_ml_health_non_200(self):
        fake_resp = mock.Mock(status_code=503)
        with mock.patch.object(self.web_api.requests, "get", return_value=fake_resp):
            resp = self.client.get("/api/ml/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["ok"], False)
        self.assertIn("status=503", resp.json()["error"])

    def test_ml_health_exception(self):
        with mock.patch.object(self.web_api.requests, "get", side_effect=RuntimeError("down")):
            resp = self.client.get("/api/ml/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["ok"], False)
        self.assertIn("down", resp.json()["error"])

    def test_upload_rejects_invalid_payload_json(self):
        resp = self.client.post(
            "/api/tasks/upload",
            data={"payload_json": "{bad json"},
            files=[("files", ("a.jpg", io.BytesIO(b"x"), "image/jpeg"))],
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid payload_json", resp.json().get("detail", ""))

    def test_upload_rejects_unsupported_extension(self):
        payload = '{"task_name":"demo","classes":["obj"]}'
        resp = self.client.post(
            "/api/tasks/upload",
            data={"payload_json": payload},
            files=[("files", ("a.txt", io.BytesIO(b"x"), "text/plain"))],
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No supported image files", resp.json().get("detail", ""))

    def test_upload_returns_500_when_create_task_fails(self):
        self.web_api.create_task_and_upload = mock.Mock(side_effect=RuntimeError("cvat down"))
        payload = '{"task_name":"demo","classes":["obj"]}'
        resp = self.client.post(
            "/api/tasks/upload",
            data={"payload_json": payload},
            files=[("files", ("a.jpg", io.BytesIO(b"x"), "image/jpeg"))],
        )
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Failed to create/upload CVAT task", resp.json().get("detail", ""))

    def test_upload_success_without_preannotation(self):
        self.web_api.create_task_and_upload = mock.Mock(return_value=901)
        payload = '{"task_name":"demo","classes":["car"],"run_preannot":false}'
        resp = self.client.post(
            "/api/tasks/upload",
            data={"payload_json": payload},
            files=[("files", ("a.jpg", io.BytesIO(b"x"), "image/jpeg"))],
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["task_id"], 901)
        self.assertEqual(body["preannotation"], None)
        st = self.web_api._PREANN_STATUS.get(901, {})
        self.assertEqual(st.get("status"), "skipped")

    def test_upload_starts_preannotation_thread(self):
        self.web_api.create_task_and_upload = mock.Mock(return_value=777)
        fake_thread = mock.Mock()
        with mock.patch.object(self.web_api.threading, "Thread", return_value=fake_thread) as thread_cls:
            payload = '{"task_name":"demo","classes":["car"],"run_preannot":true}'
            resp = self.client.post(
                "/api/tasks/upload",
                data={"payload_json": payload},
                files=[("files", ("a.jpg", io.BytesIO(b"x"), "image/jpeg"))],
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["task_id"], 777)
        self.assertEqual(body["preannotation"], {"status": "processing"})
        thread_cls.assert_called_once()
        fake_thread.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
