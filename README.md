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

3. Экспортируйте переменные окружения
```
export CVAT_TOKEN="4c2qs4PC.KeFbJBXxSBTkqYjaMobY7vnlOCZ2DwIX"
export CVAT_URL="https://app.cvat.ai" 
export SAM_CHECKPOINT="weights/sam_vit_b_01ec64.pth"
export GND_DINO_CHECKPOINT="weights/groundingdino_swint_ogc.pth"
export GND_DINO_CONFIG="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
export ML_SERVICE_URL="http://localhost:8000"
```

 export GND_DINO_CONFIG=groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py
4. Запустите
```commandline
streamlit run app.py
```
```commandline
uvicorn ml_service.main:app --host 0.0.0.0 --port 8000 --reload
```
