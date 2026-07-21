# Benchmarks (server-side, no web UI)

## Recommended datasets

| Dataset | Why | License / link |
|---------|-----|----------------|
| **COCO val2017 subset** (built-in script) | Diverse scenes, many categories, official instance masks | [COCO](https://cocodataset.org/) |
| **LVIS v1.0 minival** | Very diverse objects, hard cases | [LVIS](https://www.lvisdataset.org/) |
| **ODinW-13** | 13 “in the wild” domains (wildlife, underwater, …) | [GLIP / ODinW](https://github.com/microsoft/GLIP) |
| **SHWD** | Safety helmets + workers (PPE, closer to your domain) | Search “Safety Helmet Wearing Dataset” |

For a quick start on the server, use the **COCO subset** script below (no manual labeling).

### Construction / PPE (optional)

COCO has `person` but not `helmet` / `tower crane`. For PPE-specific metrics, add a small **custom manifest** (your own images + PNG masks) or export from CVAT.

---

## 1. Prepare COCO subset

```bash
cd ~/gitml/ml
conda activate ml-gpu
# If you previously got "Wrote 0 samples", refresh annotations cache:
rm -rf benchmarks/_coco_cache/instances_val2017.json benchmarks/_coco_cache/annotations
python scripts/download_coco_val_subset.py \
  --out-dir benchmarks/coco_val_subset \
  --max-images 80 \
  --categories person,car,truck,bus,bicycle,backpack,bench \
  --force-refresh-annotations
```

The script downloads **one COCO image at a time** (not the full 18GB val2017.zip).
You should see lines like `COCO loaded: images=5000 annotations=...` and `Wrote N samples` with N > 0.

Output: `benchmarks/coco_val_subset/manifest.jsonl`, `images/`, `gt_masks/`.

---

## 2. Run benchmark (SAM3 native vs GND+SAM pipeline)

```bash
export PYTHONPATH=/home/agazizova/gitml/ml:$PYTHONPATH
export SAM_BACKEND=sam3_native
# GND paths as in your .env:
export GND_DINO_CHECKPOINT=weights/groundingdino_swint_ogc.pth
export GND_DINO_CONFIG=GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py

python scripts/run_gnd_sam3_benchmark.py \
  --manifest benchmarks/coco_val_subset/manifest.jsonl \
  --strategy both \
  --score-threshold 0.3 \
  --out-dir benchmarks/runs/coco_val_$(date +%Y%m%d_%H%M)
```

Strategies:

- `sam3_native` — official SAM3 text prompts only
- `gnd_sam_pipeline` — production path (native fallback → GND + SAM multimask)
- `gnd_sam` — GND boxes + SAM only (no native text first)
- `both` — runs `sam3_native` and `gnd_sam_pipeline`, writes two summaries

Reports:

- `summary_sam3_native.json`, `summary_gnd_sam_pipeline.json` — mean IoU, Dice, precision, recall, boundary F1, failure rate, latency p50/p95
- `per_image_*.csv` — per-sample metrics

---

## 3. Evaluate precomputed masks only

If predictions are already saved as PNG masks (same paths as GT tree):

```bash
python scripts/eval_sam3.py \
  --gt-dir benchmarks/coco_val_subset/gt_masks \
  --pred-dir benchmarks/runs/.../pred_masks/sam3_native \
  --per-image-out eval.csv
```

---

## 4. Unit tests (no GPU)

On Mac or server CPU:

```bash
cd ~/gitml/ml
pytest tests/test_sam3_metrics.py -q
```

GPU integration (loads real models, 1 sample):

```bash
pytest tests/test_gnd_sam3_benchmark.py -m gpu -q
```

---

## Custom manifest format

JSONL, one object per line:

```json
{
  "sample_id": "img01_person_0",
  "image": "images/img01.jpg",
  "gt_mask": "gt_masks/img01_person_0.png",
  "prompt": "person",
  "scenario": "construction_site"
}
```

Paths are relative to the manifest directory unless absolute.
