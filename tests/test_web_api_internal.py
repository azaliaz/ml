import base64
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

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


class TestWebApiInternal(unittest.TestCase):
    def setUp(self):
        self.web_api = _load_web_api_with_fake_cvat()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _mk_zip(self, name: str, entries: dict[str, bytes]) -> Path:
        p = self.root / name
        with zipfile.ZipFile(p, "w") as zf:
            for k, v in entries.items():
                zf.writestr(k, v)
        return p

    def test_build_preview_from_zip_returns_first_png(self):
        z = self._mk_zip(
            "a.zip",
            {
                "previews/z.png": b"zzz",
                "previews/a.png": b"aaa",
            },
        )
        fname, b64 = self.web_api._build_preview_from_zip(z)
        self.assertEqual(fname, "a.png")
        self.assertEqual(base64.b64decode(b64.encode("ascii")), b"aaa")

    def test_build_preview_from_zip_no_preview(self):
        z = self._mk_zip("a.zip", {"annotations/annotations_coco.json": b"{}"})
        fname, b64 = self.web_api._build_preview_from_zip(z)
        self.assertIsNone(fname)
        self.assertIsNone(b64)

    def test_validate_preannotation_archive_success(self):
        coco = {"images": [{"id": 1}], "annotations": [{"id": 1, "image_id": 1}], "categories": [{"id": 1}]}
        z = self._mk_zip("ok.zip", {"annotations/annotations_coco.json": json.dumps(coco).encode("utf-8")})
        out = self.web_api._validate_preannotation_archive(z)
        self.assertEqual(out["images_count"], 1)
        self.assertEqual(out["annotations_count"], 1)
        self.assertEqual(out["categories_count"], 1)

    def test_validate_preannotation_archive_missing_file(self):
        with self.assertRaisesRegex(RuntimeError, "не создан"):
            self.web_api._validate_preannotation_archive(self.root / "missing.zip")

    def test_validate_preannotation_archive_empty_zip(self):
        p = self.root / "empty.zip"
        p.write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "пустой"):
            self.web_api._validate_preannotation_archive(p)

    def test_validate_preannotation_archive_missing_annotations_json(self):
        z = self._mk_zip("bad.zip", {"x.txt": b"x"})
        with self.assertRaisesRegex(RuntimeError, "нет annotations/annotations_coco.json"):
            self.web_api._validate_preannotation_archive(z)

    def test_validate_preannotation_archive_invalid_images_type(self):
        coco = {"images": {}, "annotations": [{"id": 1}], "categories": []}
        z = self._mk_zip("bad.zip", {"annotations/annotations_coco.json": json.dumps(coco).encode("utf-8")})
        with self.assertRaisesRegex(RuntimeError, "формат images"):
            self.web_api._validate_preannotation_archive(z)

    def test_validate_preannotation_archive_zero_annotations(self):
        coco = {"images": [{"id": 1}], "annotations": [], "categories": []}
        z = self._mk_zip("bad.zip", {"annotations/annotations_coco.json": json.dumps(coco).encode("utf-8")})
        with self.assertRaisesRegex(RuntimeError, "ни одной аннотации"):
            self.web_api._validate_preannotation_archive(z)

    def test_poll_cvat_request_requires_token(self):
        with mock.patch.object(self.web_api, "CVAT_TOKEN", ""):
            with self.assertRaisesRegex(RuntimeError, "CVAT_TOKEN is not set"):
                self.web_api._poll_cvat_request(1, timeout_s=1, poll_s=0.01)

    def test_poll_cvat_request_completed(self):
        with mock.patch.object(self.web_api, "CVAT_TOKEN", "tok"), mock.patch.object(
            self.web_api.requests,
            "get",
            return_value=mock.Mock(status_code=200, json=lambda: {"status": "completed"}, text=""),
        ):
            out = self.web_api._poll_cvat_request(5, timeout_s=1, poll_s=0.01)
        self.assertEqual(out["status"], "completed")

    def test_poll_cvat_request_failed_status(self):
        with mock.patch.object(self.web_api, "CVAT_TOKEN", "tok"), mock.patch.object(
            self.web_api.requests,
            "get",
            return_value=mock.Mock(status_code=200, json=lambda: {"status": "failed"}, text=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "CVAT import request failed"):
                self.web_api._poll_cvat_request(5, timeout_s=1, poll_s=0.01)

    def test_poll_cvat_request_non_200(self):
        with mock.patch.object(self.web_api, "CVAT_TOKEN", "tok"), mock.patch.object(
            self.web_api.requests,
            "get",
            return_value=mock.Mock(status_code=500, text="oops"),
        ):
            with self.assertRaisesRegex(RuntimeError, "poll failed"):
                self.web_api._poll_cvat_request(5, timeout_s=1, poll_s=0.01)

    def test_poll_cvat_request_non_json(self):
        bad = mock.Mock(status_code=200, text="html")
        bad.json.side_effect = ValueError("bad json")
        with mock.patch.object(self.web_api, "CVAT_TOKEN", "tok"), mock.patch.object(self.web_api.requests, "get", return_value=bad):
            with self.assertRaisesRegex(RuntimeError, "non-JSON"):
                self.web_api._poll_cvat_request(5, timeout_s=1, poll_s=0.01)

    def test_poll_cvat_request_timeout(self):
        pending = mock.Mock(status_code=200, json=lambda: {"status": "queued"}, text="")
        with mock.patch.object(self.web_api, "CVAT_TOKEN", "tok"), mock.patch.object(
            self.web_api.requests, "get", return_value=pending
        ), mock.patch.object(self.web_api.time, "sleep", return_value=None):
            with self.assertRaises(TimeoutError):
                self.web_api._poll_cvat_request(5, timeout_s=0, poll_s=0)


if __name__ == "__main__":
    unittest.main()
