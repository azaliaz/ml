# app.py
import streamlit as st
import zipfile
import os
import json
import time
import requests
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)
from cvat_client import create_task_and_upload, export_annotations_by_id, import_annotations_to_task

UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")
PREANN_DIR = UPLOAD_DIR / "preannotations"
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
PREANN_DIR.mkdir(exist_ok=True)

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8000")

st.set_page_config(page_title="CVAT Авто-загрузка и предразметка", layout="wide")
st.title("CVAT — Авто-загрузка и предразметка")

st.sidebar.header("Параметры предразметки")
task_type = st.sidebar.selectbox(
    "Тип задачи",
    ["detection", "segmentation", "classification", "other"],
    format_func=lambda x: {
        "detection": "Детекция",
        "segmentation": "Сегментация",
        "classification": "Классификация",
        "other": "Другое"
    }[x]
)

if task_type != "segmentation":
    class_name = st.sidebar.text_input("Имя класса (для предразметки / создания меток)", value="object")
    class_description = st.sidebar.text_area("Описание класса (будет сохранено в метке)", value="", height=120)
else:
    class_name = ""
    class_description = ""

st.sidebar.markdown("---")

score_threshold = st.sidebar.slider("Порог score (для моделей detection, используется ML-сервисом)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
max_boxes = st.sidebar.number_input("Max боксов на изображение (для detection)", min_value=1, max_value=200, value=10)


st.sidebar.markdown("---")
run_preannot = st.sidebar.checkbox("Выполнить предразметку перед импортом в CVAT", value=False)

def fetch_ml_health(timeout: int = 5) -> dict:
    try:
        r = requests.get(f"{ML_SERVICE_URL}/health", timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return {"error": f"status_code={r.status_code}", "text": r.text}
    except Exception as e:
        return {"error": str(e)}

st.write("---")
st.header("Загрузить файлы в CVAT")

uploaded_files = st.file_uploader(
    "Выберите изображения или видео",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "mp4", "mov"],
    accept_multiple_files=True
)

task_name = st.text_input("Название задачи", value="streamlit_task")

def preannotate_stub(local_paths: List[str], class_name: str) -> dict:
    results = {}
    for p in local_paths:
        try:
            import cv2
            img = cv2.imread(p)
            h, w = img.shape[:2]
            bbox = [0, 0, w, h]
            ann = {
                "image": str(p),
                "predictions": [
                    {
                        "label": class_name,
                        "bbox": bbox,
                        "score": 1.0,
                        "type": "bbox_stub"
                    }
                ],
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            ann_path = PREANN_DIR / (Path(p).stem + ".preann.json")
            with open(ann_path, "w", encoding="utf-8") as fh:
                json.dump(ann, fh, ensure_ascii=False, indent=2)
            results[p] = str(ann_path)
        except Exception as e:
            results[p] = f"error: {e}"
    return results

def preannotate_via_service(local_paths: List[str], score_thr: float, max_boxes: int, task_type: str, class_name: Optional[str] = None) -> dict:
    files = []
    opened_files = []
    try:
        for p in local_paths:
            mime = "image/jpeg"
            name = Path(p).name
            fobj = open(p, "rb")
            opened_files.append(fobj)
            files.append(("images", (name, fobj, mime)))

        payload = {
            "score_threshold": float(score_thr),
            "max_boxes": int(max_boxes),
            "format": "coco",
            "task_type": task_type
        }
        if task_type != "segmentation" and class_name:
            payload["class_name"] = class_name

        resp = requests.post(f"{ML_SERVICE_URL}/preannotate", data={"payload": json.dumps(payload)}, files=files, timeout=600)
        if resp.status_code != 200:
            raise RuntimeError(f"ML service returned status {resp.status_code}: {resp.text}")

        ts = int(time.time())
        out_path = PREANN_DIR / f"preann_{ts}.zip"
        with open(out_path, "wb") as fh:
            fh.write(resp.content)

        return {"__archive__": str(out_path)}
    except Exception as e:
        raise RuntimeError(f"Ошибка запроса к ML-сервису: {e}")
    finally:
        for fo in opened_files:
            try:
                fo.close()
            except Exception:
                pass

if st.button("Загрузить в CVAT") and uploaded_files:
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    local_paths = []
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        if ext not in allowed_exts:
            st.warning(f"Файл {f.name} пропущен (не поддерживаемый формат для предразметки: {ext})")
            continue
        path = UPLOAD_DIR / f.name
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        local_paths.append(str(path))

    if not local_paths:
        st.error("Нет подходящих изображений для загрузки после фильтрации форматов.")
    else:
        if task_type == "segmentation":
            labels_to_create = None
        else:
            labels_to_create = [{"name": class_name or "object", "description": class_description or ""}]

        preann_results = {}

        with st.spinner("Создаём задачу и загружаем файлы в CVAT..."):
            try:
                task_id = create_task_and_upload(local_paths, task_name, labels=labels_to_create)
            except Exception as e:
                st.error(f"Ошибка при создании/загрузке задачи: {e}")
                task_id = None

        if task_id is not None:
            st.session_state["task_id"] = task_id
            st.session_state["uploaded_files"] = local_paths

            if run_preannot:
                with st.spinner("Выполняю предразметку на ML-сервисе..."):
                    try:
                        # отправляем task_type; если segmentation — class_name не передаётся в payload внутри функции
                        preann_results = preannotate_via_service(local_paths, score_threshold, max_boxes, task_type, class_name=(class_name if task_type != "segmentation" else None))
                    except Exception as e:
                        st.error(f"Ошибка предразметки: {e}")
                        preann_results = {}

                if preann_results:
                    st.success("Предразметка завершена.")
                    st.write("Результаты предразметки:")
                    for k, v in preann_results.items():
                        st.write(f"- {k} → {v}")

                    archive = preann_results.get("__archive__")
                    if archive and os.path.exists(archive):
                        # Сохраняем превью в session_state, чтобы не терялись
                        if "previews_data" not in st.session_state:
                            st.session_state["previews_data"] = {}

                        try:
                            # Показываем информацию об архиве и кнопку на скачивание
                            st.write(f"Архив предразметки: `{archive}` ({os.path.getsize(archive)} bytes)")
                            with open(archive, "rb") as f:
                                st.download_button(
                                    label="Скачать архив предразметки",
                                    data=f.read(),
                                    file_name=Path(archive).name,
                                    mime="application/zip"
                                )

                            # Извлекаем превью из архива и сохраняем в session_state
                            with zipfile.ZipFile(archive, "r") as zf:
                                preview_names = [n for n in zf.namelist() if n.startswith("previews/")]
                                for pn in preview_names:
                                    if pn not in st.session_state["previews_data"]:
                                        try:
                                            st.session_state["previews_data"][pn] = zf.read(pn)
                                        except Exception:
                                            continue

                        except Exception:
                            st.info("Не удалось открыть архив превью или показать превью.")

                        # Показываем превью из session_state
                        previews = st.session_state.get("previews_data", {})
                        if previews:
                            st.markdown("**Превью предразметки**")
                            num_cols = min(4, len(previews))  # максимум 4 колонки
                            cols = st.columns(num_cols)
                            for i, (pn, data) in enumerate(previews.items()):
                                try:
                                    col = cols[i % num_cols]
                                    col.image(data, width=300, caption=Path(pn).name)
                                except Exception:
                                    continue
                        with st.spinner("Импорт аннотаций в CVAT..."):
                            try:
                                rq = import_annotations_to_task(int(task_id), archive, format_name="COCO 1.0",
                                                                timeout=600)
                                if rq:
                                    st.success(f"Импорт запущен в фоне, request id: {rq}")
                                else:
                                    st.success("Аннотации импортированы в задачу.")
                            except Exception as e:
                                st.error(f"Ошибка импорта аннотаций в CVAT: {e}")
                else:
                    st.info("Предразметка не вернула результатов.")
            else:
                st.success(f"Загрузка в CVAT завершена. ID задачи: {task_id}")

            st.session_state["preann_results"] = preann_results

st.write("---")
st.header("Скачать / Экспорт аннотаций из CVAT")

export_format = st.selectbox("Формат экспорта", ["COCO 1.0", "YOLO 1.1", "Pascal VOC 1.1"])

if "task_id" in st.session_state:
    task_id = st.session_state["task_id"]
    uploaded_files = st.session_state.get("uploaded_files", [])
    st.markdown(f"**Текущий task_id:** `{task_id}`")

    include_images_for_export = st.checkbox("Включить изображения в экспорт", value=False)

    if st.button("Скачать аннотации (ZIP)"):
        base_name = f"{task_name}_annotations"
        out_name = f"{base_name}.zip"
        out_path = EXPORT_DIR / out_name
        if out_path.exists():
            ts = int(time.time())
            out_path = EXPORT_DIR / f"{base_name}_{ts}.zip"

        with st.spinner("Экспортируем аннотации..."):
            try:
                exported_path = export_annotations_by_id(int(task_id), str(out_path), format_name=export_format,
                                                         include_images=include_images_for_export)
            except Exception as e:
                st.error(f"Ошибка при экспорте: {e}")
            else:
                if exported_path and Path(exported_path).exists():
                    with open(exported_path, "rb") as f:
                        st.download_button(label="Скачать аннотации ZIP", data=f.read(),
                                           file_name=Path(exported_path).name, mime="application/zip")
                else:
                    st.error("Экспорт завершился, но файл не найден локально.")

else:
    st.warning("Сначала загрузите файлы в CVAT!")