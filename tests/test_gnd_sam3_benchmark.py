"""Benchmark pipeline smoke tests (synthetic data; optional GPU integration)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestBenchmarkManifest(unittest.TestCase):
    def test_load_manifest_jsonl(self):
        from scripts.run_gnd_sam3_benchmark import _load_manifest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "image": "img.png",
                        "gt_mask": "gt.png",
                        "prompt": "person",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = _load_manifest(path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["prompt"], "person")

    def test_dry_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from PIL import Image

            img = root / "images" / "x.jpg"
            gt = root / "gt_masks" / "x_person_0.png"
            img.parent.mkdir(parents=True)
            gt.parent.mkdir(parents=True)
            arr = np.zeros((32, 32), dtype=np.uint8)
            arr[10:22, 10:22] = 1
            Image.fromarray((arr * 255).astype(np.uint8)).save(img)
            Image.fromarray((arr * 255).astype(np.uint8)).save(gt)

            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "x_person_0",
                        "image": "images/x.jpg",
                        "gt_mask": "gt_masks/x_person_0.png",
                        "prompt": "person",
                        "scenario": "synthetic",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            import scripts.run_gnd_sam3_benchmark as bench

            fake_mask = arr.copy()
            with mock.patch.object(
                bench,
                "_run_strategy",
                return_value=([fake_mask], 5.0),
            ):
                argv = [
                    "run_gnd_sam3_benchmark.py",
                    "--manifest",
                    str(manifest),
                    "--strategy",
                    "sam3_native",
                    "--out-dir",
                    str(root / "run"),
                    "--sam-backend",
                    "sam3_native",
                ]
                with mock.patch.object(sys, "argv", argv):
                    with mock.patch.dict(
                        "os.environ",
                        {"SAM_BACKEND": "sam3_native"},
                        clear=False,
                    ):
                        # Avoid loading real torch models during import side effects.
                        with mock.patch("ml_service.models.load_sam_model_if_available"):
                            with mock.patch(
                                "ml_service.models.load_groundingdino_if_available"
                            ):
                                bench.main()

            summary_path = root / "run" / "summary_sam3_native.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["samples_ok"], 1)
            self.assertGreaterEqual(summary["mean_iou"], 0.99)


try:
    import pytest
    import torch

    @pytest.mark.gpu
    def test_gpu_single_sample_smoke():
        """Run one real inference sample if CUDA + models are available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from ml_service.inference import run_text_guided_segmentation_native
        import ml_service.models  # noqa: F401

        if not getattr(ml_service.models, "SAM_AVAILABLE", False):
            pytest.skip("SAM not loaded")

        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            p = Path(tmp) / "img.png"
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[40:90, 40:90] = 200
            Image.fromarray(arr).save(p)
            out = run_text_guided_segmentation_native(str(p), ["object"], score_threshold=0.1)
            assert isinstance(out, list)

except ImportError:
    pass
