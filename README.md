# Автодообучение CV-моделей

Платформа для автоматизации работы с компьютерным зрением: предразметка данных в CVAT, мониторинг дрейфа и подготовка к дообучению моделей.

**Модули**

| Модуль | Описание |
|--------|----------|
| **Предразметка** | Загрузка изображений/видео → Grounding DINO + SAM/SAM3 → экспорт в CVAT |
| **Источники видео** | Настройка mp4/RTSP-потоков для мониторинга дрейфа |
| **Мониторинг** | Дашборд метрик и алертов по дрейфу данных |

**Стек:** Python (FastAPI), React (Vite), CVAT SDK, Grounding DINO, SAM / SAM3.

---

## Быстрый запуск

### 1. Окружение

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Переменные окружения

Создайте файл `.env` в каталоге `ml/`:

```bash
CVAT_URL=https://app.cvat.ai
CVAT_TOKEN=your_token


SAM_CHECKPOINT=weights/sam_vit_b_01ec64.pth
GND_DINO_CHECKPOINT=weights/groundingdino_swint_ogc.pth
GND_DINO_CONFIG=GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py


ML_SERVICE_URL=http://localhost:8000
```

Для SAM3 (опционально): `SAM_BACKEND=sam3_native`, `HF_TOKEN=...` — см. [benchmarks/README.md](benchmarks/README.md).

### 3. Запуск (3 терминала)

**Терминал 1 — ML-сервис** (инференс моделей):

```bash
uvicorn ml_service.main:app --host 0.0.0.0 --port 8000 --reload
```

**Терминал 2 — Web API** (оркестрация, CVAT, загрузка файлов):

```bash
uvicorn web_api:app --host 0.0.0.0 --port 5050 --reload
```

**Терминал 3 — фронтенд:**

```bash
cd frontend && npm install && npm run dev
```


### 4. Продакшен-сборка фронтенда

```bash
cd frontend && npm run build
```


---

## Альтернатива: Streamlit

```bash
streamlit run app.py
```

---

## Дополнительно

- **Бенчмарки и SAM3:** [benchmarks/README.md](benchmarks/README.md)
- **Проверка ML-сервиса:** `GET http://localhost:8000/health`
- **CORS:** переменная `CORS_ORIGINS` (через запятую), по умолчанию `http://localhost:5173,http://127.0.0.1:5173`
