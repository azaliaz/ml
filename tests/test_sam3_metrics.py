"""Unit tests for mask metrics (no GPU / no models)."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_service.sam3_metrics import (  # noqa: E402
    MaskPair,
    align_masks,
    boundary_f1,
    evaluate_mask_pair,
    evaluate_pairs,
    mask_dice,
    mask_iou,
    pick_best_pred_mask,
    resolve_pairs_from_csv,
    summarize_rows,
    write_per_image_csv,
)


class TestSam3Metrics(unittest.TestCase):
    def test_perfect_overlap(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[8:24, 8:24] = 1
        m = evaluate_mask_pair(gt, gt.copy())
        self.assertAlmostEqual(m["iou"], 1.0)
        self.assertAlmostEqual(m["dice"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)

    def test_disjoint_masks(self):
        gt = np.zeros((20, 20), dtype=np.uint8)
        gt[0:10, :] = 1
        pred = np.zeros_like(gt)
        pred[10:20, :] = 1
        self.assertAlmostEqual(mask_iou(gt, pred), 0.0)
        self.assertAlmostEqual(mask_dice(gt, pred), 0.0)
        m = evaluate_mask_pair(gt, pred)
        self.assertAlmostEqual(m["recall"], 0.0)

    def test_align_smaller_pred(self):
        gt = np.ones((10, 12), dtype=np.uint8)
        pred = np.ones((6, 6), dtype=np.uint8)
        pa, ga = align_masks(pred, gt)
        self.assertEqual(pa.shape, ga.shape)
        self.assertGreaterEqual(mask_iou(pa, ga), 0.3)

    def test_pick_best_pred_mask(self):
        gt = np.zeros((20, 20), dtype=np.uint8)
        gt[5:15, 5:15] = 1
        bad = np.zeros_like(gt)
        bad[0:3, 0:3] = 1
        good = gt.copy()
        best, iou = pick_best_pred_mask([bad, good], gt)
        self.assertIsNotNone(best)
        self.assertAlmostEqual(iou, 1.0)

    def test_evaluate_pairs_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "gt.png"
            pred = root / "pred.png"
            from PIL import Image

            arr = np.zeros((16, 16), dtype=np.uint8)
            arr[4:12, 4:12] = 1
            Image.fromarray(arr * 255).save(gt)
            Image.fromarray(arr * 255).save(pred)

            csv_path = root / "pairs.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["sample_id", "gt_path", "pred_path", "scenario", "latency_ms"])
                writer.writerow(["s1", str(gt), str(pred), "unit", "12.5"])

            pairs = resolve_pairs_from_csv(csv_path)
            rows = evaluate_pairs(pairs)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["ok"])
            self.assertAlmostEqual(rows[0]["iou"], 1.0)

            out_csv = root / "out.csv"
            write_per_image_csv(rows, out_csv)
            self.assertTrue(out_csv.exists())

            summary = summarize_rows(rows)
            self.assertEqual(summary["samples_ok"], 1)
            self.assertAlmostEqual(summary["mean_iou"], 1.0)

    def test_missing_pred_counts_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "gt.png"
            from PIL import Image

            Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(gt)
            pairs = [
                MaskPair(
                    sample_id="x",
                    gt_path=gt,
                    pred_path=root / "missing.png",
                    scenario="unit",
                )
            ]
            rows = evaluate_pairs(pairs, ignore_missing_pred=False)
            self.assertFalse(rows[0]["ok"])
            self.assertEqual(rows[0]["error"], "missing_prediction")

    def test_boundary_f1_identical(self):
        m = np.zeros((24, 24), dtype=np.uint8)
        m[6:18, 6:18] = 1
        self.assertGreater(boundary_f1(m, m.copy(), tolerance_px=2), 0.9)


if __name__ == "__main__":
    unittest.main()
