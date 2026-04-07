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
