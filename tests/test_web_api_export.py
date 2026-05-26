import importlib
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
    fake.grant_validation_access_for_task = mock.Mock(return_value={})
    fake.import_annotations_to_task = mock.Mock(return_value=None)

    sys.modules["cvat_client"] = fake
    if "web_api" in sys.modules:
        del sys.modules["web_api"]
    module = importlib.import_module("web_api")
    return module


class TestExportFallback(unittest.TestCase):
    def setUp(self):
        self.web_api = _load_web_api_with_fake_cvat()
        self.client = TestClient(self.web_api.app)
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.web_api.PREANN_DIR = root / "preannotations"
        self.web_api.EXPORT_DIR = root / "exports"
        self.web_api.PREANN_DIR.mkdir(parents=True, exist_ok=True)
        self.web_api.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.web_api._PREANN_STATUS.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_export_uses_status_archive_fallback(self):
        archive = self.web_api.PREANN_DIR / "preann_task_2214190_1.zip"
        archive.write_bytes(b"zip-from-status")
        self.web_api._PREANN_STATUS[2214190] = {"archive_path": str(archive)}

        resp = self.client.post("/api/tasks/2214190/export", json={})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"zip-from-status")
        self.assertIn("preannotation.zip", resp.headers.get("content-disposition", ""))

    def test_export_uses_latest_local_archive_when_status_missing(self):
        older = self.web_api.PREANN_DIR / "preann_task_2214190_100.zip"
        newer = self.web_api.PREANN_DIR / "preann_task_2214190_200.zip"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")

        resp = self.client.post("/api/tasks/2214190/export", json={})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"new")

    def test_export_returns_500_when_no_cvat_and_no_local_archive(self):
        resp = self.client.post("/api/tasks/2214190/export", json={})

        self.assertEqual(resp.status_code, 500)
        body = resp.json()
        self.assertIn("Export failed:", body.get("detail", ""))

    def test_export_returns_cvat_file_when_export_succeeds(self):
        cvat_export = self.web_api.EXPORT_DIR / "from_cvat.zip"
        cvat_export.write_bytes(b"zip-from-cvat")
        self.web_api.export_annotations_by_id = mock.Mock(return_value=str(cvat_export))

        resp = self.client.post("/api/tasks/2214190/export", json={"task_name": "my-task"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"zip-from-cvat")
        self.assertIn("from_cvat.zip", resp.headers.get("content-disposition", ""))

    def test_export_ignores_empty_status_archive_and_uses_latest_valid_local(self):
        empty = self.web_api.PREANN_DIR / "preann_task_2214190_10.zip"
        valid = self.web_api.PREANN_DIR / "preann_task_2214190_20.zip"
        empty.write_bytes(b"")
        valid.write_bytes(b"valid-local")
        self.web_api._PREANN_STATUS[2214190] = {"archive_path": str(empty)}

        resp = self.client.post("/api/tasks/2214190/export", json={})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"valid-local")

    def test_export_ignores_invalid_status_path_and_uses_local_archive(self):
        valid = self.web_api.PREANN_DIR / "preann_task_2214190_77.zip"
        valid.write_bytes(b"local")
        self.web_api._PREANN_STATUS[2214190] = {"archive_path": "/tmp/not_found.zip"}

        resp = self.client.post("/api/tasks/2214190/export", json={})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"local")

    def test_export_sanitizes_filename_for_fallback_download(self):
        archive = self.web_api.PREANN_DIR / "preann_task_2214190_1.zip"
        archive.write_bytes(b"x")
        self.web_api._PREANN_STATUS[2214190] = {"archive_path": str(archive)}

        resp = self.client.post("/api/tasks/2214190/export", json={"task_name": "task <> bad / name"})

        self.assertEqual(resp.status_code, 200)
        cd = resp.headers.get("content-disposition", "")
        self.assertIn("task____bad___name_2214190_preannotation.zip", cd)


if __name__ == "__main__":
    unittest.main()
