## Запуск приложения
1. Настроить окружение: 
```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```
2. Установить зависимости:

   ```bash
   pip install -r requirements.txt
   ```
   
3. Экспортируйте переменные окружения
```
export CVAT_TOKEN="ваш_PAT_тут"
export CVAT_URL="https://app.cvat.ai"   
```
4. Запустите
```
streamlit run app.py
```