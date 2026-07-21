## Запуск приложения
1. Настроить окружение: 
```
python3 -m venv .venv
```

```
source .venv/bin/activate
```
2. Установить зависимости:

```
pip install -r requirements.txt
```

3. Экспортируйте переменные окружения (обязательно `CVAT_TOKEN` для загрузки в CVAT):
```
export CVAT_URL="https://app.cvat.ai"
export CVAT_TOKEN="…"
export SAM_CHECKPOINT="weights/sam_vit_b_01ec64.pth"
export GND_DINO_CHECKPOINT="weights/groundingdino_swint_ogc.pth"
export GND_DINO_CONFIG="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
export ML_SERVICE_URL="http://localhost:8000"
export QWEN_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
export QWEN_MAX_NEW_TOKENS=256
```

 export GND_DINO_CONFIG=groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py

4. Запустите ML-сервис (предразметка):
```commandline
uvicorn ml_service.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Интерфейс: **React** (рекомендуется) или Streamlit.

**React + API** — в одном терминале API (порт 5050), в другом фронтенд (Vite проксирует `/api`):
```commandline
uvicorn web_api:app --host 0.0.0.0 --port 5050 --reload
```
```commandline
cd frontend && npm install && npm run dev
```
Откройте в браузере адрес, который выведет Vite (обычно `http://127.0.0.1:5173`). Сборка статики: `cd frontend && npm run build`, затем раздавайте каталог `frontend/dist` любым HTTP-сервером и проксируйте запросы `/api` на `http://127.0.0.1:5050`.

Переменная `CORS_ORIGINS` (через запятую) задаёт разрешённые origin для API, по умолчанию включены `http://localhost:5173` и `http://127.0.0.1:5173`.

**Streamlit** (прежний вариант):
```commandline
streamlit run app.py
```

## Официальный SAM3 (Meta, не HF pipeline)

По умолчанию `SAM_BACKEND=sam3` использует Hugging Face `transformers.pipeline` (обёртка с fallback по боксам).

Для **официального** SAM3 (`Sam3Processor` из репозитория Meta):

1. Установите пакет (Python 3.12+, CUDA, доступ к чекпоинту на Hugging Face):

```commandline
cd /path/to/gitml/ml
git clone https://github.com/facebookresearch/sam3.git sam3
cd sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
hf auth login
```

Репозиторий должен лежать в **`ml/sam3/`** (рядом с `ml_service/`, `weights/`).  
`ml_service` автоматически добавляет `ml/sam3` в `sys.path`.  
При необходимости: `SAM3_LOCAL_DIR=/abs/path/to/sam3` в `.env`.

2. В `.env` на сервере:

```commandline
SAM_BACKEND=sam3_native
HF_TOKEN=...
HUGGINGFACE_HUB_TOKEN=...
SAM3_CONFIDENCE_THRESHOLD=0.5
```

3. Перезапустите `uvicorn ml_service.main:app` и проверьте `/health`:

- `sam_backend`: `sam3_native`
- `sam3_load_mode`: `native`
- `sam_predictor_available`: `true`

Текстовая сегментация идёт через `set_text_prompt` (без HF mask-generation pipeline).

## Бенчмарк GND+SAM3 на сервере (без web UI)

Полная инструкция: [benchmarks/README.md](benchmarks/README.md)

Кратко:

```bash
# 1) Скачать подмножество COCO val (разнообразные сцены + GT-маски)
python scripts/download_coco_val_subset.py --out-dir benchmarks/coco_val_subset --max-images 80

# 2) Прогнать SAM3-native vs GND+SAM и получить IoU/Dice/Boundary F1
export PYTHONPATH=$PWD:$PYTHONPATH SAM_BACKEND=sam3_native
python scripts/run_gnd_sam3_benchmark.py \
  --manifest benchmarks/coco_val_subset/manifest.jsonl \
  --strategy both \
  --out-dir benchmarks/runs/latest

# 3) Unit-тесты метрик (без GPU)
pytest tests/test_sam3_metrics.py -q
```

## Оценка качества SAM3 (метрики)

Скрипт `scripts/eval_sam3.py` считает:
- IoU (Jaccard)
- Dice (F1 для масок)
- Precision / Recall (по пикселям)
- Boundary F1
- failure rate
- latency p50 / p95 (если передать latency)

### Вариант 1: GT и pred в одинаковой структуре папок

```commandline
python scripts/eval_sam3.py \
  --gt-dir /path/to/gt_masks \
  --pred-dir /path/to/pred_masks \
  --per-image-out sam3_eval_per_image.csv
```

### Вариант 2: Явные пары через CSV

Используйте шаблон `scripts/sam3_pairs_template.csv`:

```commandline
python scripts/eval_sam3.py \
  --pairs-csv scripts/sam3_pairs_template.csv \
  --per-image-out sam3_eval_per_image.csv
```

CSV-колонки:
- `sample_id` — id примера (для отчёта)
- `gt_path` — абсолютный путь до GT-маски
- `pred_path` — абсолютный путь до предсказанной маски
- `scenario` — группа/сценарий (например `small_objects`, `occlusion`)
- `latency_ms` — время инференса для примера (опционально)
