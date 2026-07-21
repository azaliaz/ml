import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.coco_drift_utils import (  # noqa: E402
    bbox_iou_xywh,
    compare_preann_final,
    compute_batch_stats,
    load_coco_from_zip,
    normalize_image_key,
)


def _coco(images, annotations, categories=None):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories or [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    }


class TestCocoDriftUtils(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_zip(self, name: str, coco: dict) -> Path:
        path = self.root / name
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("annotations/annotations_coco.json", json.dumps(coco))
        return path

    def test_normalize_image_key_uses_basename(self):
        self.assertEqual(normalize_image_key("frames/frame_0001.jpg"), "frame_0001.jpg")

    def test_bbox_iou_identical(self):
        box = (10.0, 20.0, 30.0, 40.0)
        self.assertAlmostEqual(bbox_iou_xywh(box, box), 1.0)

    def test_compute_batch_stats_scores_and_histogram(self):
        coco = _coco(
            images=[{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [1, 1, 10, 10], "score": 0.6},
            ],
        )
        stats = compute_batch_stats(coco)
        self.assertEqual(stats.images_count, 2)
        self.assertEqual(stats.annotations_count, 2)
        self.assertAlmostEqual(stats.mean_score or 0.0, 0.7)
        self.assertEqual(stats.class_histogram["person"], 1)
        self.assertEqual(stats.class_histogram["car"], 1)
        self.assertEqual(stats.images_without_annotations, 1)

    def test_load_coco_from_zip(self):
        coco = _coco(
            images=[{"id": 1, "file_name": "x.jpg"}],
            annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}],
        )
        zpath = self._write_zip("pre.zip", coco)
        loaded, member = load_coco_from_zip(zpath)
        self.assertEqual(member, "annotations/annotations_coco.json")
        self.assertEqual(len(loaded["images"]), 1)

    def test_compare_preann_final_basic(self):
        pre = _coco(
            images=[{"id": 1, "file_name": "img.jpg"}],
            annotations=[
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10]},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10]},
            ],
        )
        final = _coco(
            images=[{"id": 1, "file_name": "subdir/img.jpg"}],
            annotations=[
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                {"id": 11, "image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10]},
                {"id": 12, "image_id": 1, "category_id": 1, "bbox": [200, 200, 10, 10]},
            ],
        )
        stats = compare_preann_final(pre, final, iou_threshold=0.5)
        self.assertEqual(stats.pre_total, 3)
        self.assertEqual(stats.matched, 1)
        self.assertEqual(stats.deleted, 1)
        self.assertEqual(stats.reclassified, 1)
        self.assertEqual(stats.added, 1)
        self.assertAlmostEqual(stats.edit_rate or 0.0, 2 / 3)
        self.assertAlmostEqual(stats.deletion_rate or 0.0, 1 / 3)
        self.assertAlmostEqual(stats.addition_rate or 0.0, 1 / 3)


if __name__ == "__main__":
    unittest.main()
