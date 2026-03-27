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
from cvat_client import (
    create_task_and_upload,
    export_annotations_by_id,
    import_annotations_to_task,
    grant_validation_access_for_task,
)

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

st.sidebar.markdown("### Классы (по одному на строку)")
classes_text = st.sidebar.text_area("Введите имена классов (каждый в новой строке)", value="object\nperson", height=140)
# parse to list
class_lines = [c.strip() for c in classes_text.splitlines() if c.strip()]
if not class_lines:
    class_lines = ["object"]

# optional per-class descriptions (simple: same description for all or left blank)
class_description_global = st.sidebar.text_area("Описание класса (будет применено ко всем меткам, можно оставить пустым)", value="", height=80)

st.sidebar.markdown("---")

score_threshold = st.sidebar.slider("Порог score (для моделей detection, используется ML-сервисом)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
max_boxes = st.sidebar.number_input("Max боксов на изображение (для detection)", min_value=1, max_value=200, value=10)
use_clip = st.sidebar.checkbox("Использовать CLIP для проверки / фильтрации (если доступен)", value=False)
use_qwen = st.sidebar.checkbox(
    "Использовать Qwen для генерации промптов",
    value=False
)

qwen_instruction = st.sidebar.text_area(
    "Инструкция для Qwen",
    value="Определи, какие объекты нужно разметить на изображениях, и верни JSON с class_names и text_prompts.",
    height=120,
)
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

def preannotate_stub(local_paths: List[str], class_names: List[str]) -> dict:
    results = {}
    for p in local_paths:
        try:
            import cv2
            img = cv2.imread(p)
            h, w = img.shape[:2]
            bbox = [0, 0, w, h]
            # create a tiny preann JSON listing first class
            ann = {
                "image": str(p),
                "predictions": [
                    {
                        "label": class_names[0] if class_names else "object",
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

def preannotate_via_service(
    local_paths: List[str],
    score_thr: float,
    max_boxes: int,
    task_type: str,
    class_names: Optional[List[str]] = None,
    use_clip_flag: bool = False,
    use_qwen: bool = False,
    qwen_instruction: str = "",
) -> dict:
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
            "task_type": task_type,
            "class_names": class_names or [],
            "use_qwen": bool(use_qwen),
            "qwen_instruction": qwen_instruction or "",
        }
        if use_clip_flag:
            payload["use_clip"] = True

        resp = requests.post(f"{ML_SERVICE_URL}/preannotate", data={"payload": json.dumps(payload)}, files=files, timeout=18000)
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

# --- основной блок загрузки в CVAT (без изменений) ---
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
        # prepare labels for CVAT task creation
        labels_to_create = []
        if task_type != "segmentation":
            for cname in class_lines:
                labels_to_create.append({"name": cname, "description": class_description_global or ""})
        else:
            # segmentation: CVAT labels still created (even if SAM masks are not class-distinct)
            for cname in class_lines:
                labels_to_create.append({"name": cname, "description": class_description_global or ""})

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
                task_type_label = {
                    "detection": "Детекция",
                    "segmentation": "Сегментация",
                    "classification": "Классификация",
                    "other": "Другое",
                }.get(task_type, task_type)
                classes_str = ", ".join(class_lines) if class_lines else "-"
                desc_str = class_description_global.strip()
                extra_parts = []
                if desc_str:
                    extra_parts.append(f"описание: {desc_str}")
                extra_parts.append(f"score ≥ {score_threshold}")
                extra_parts.append(f"max_boxes = {int(max_boxes)}")
                if use_clip:
                    extra_parts.append("CLIP: да")
                info_line = f"{task_type_label} · классы: {classes_str} · " + " · ".join(extra_parts)

                st.markdown("**Предразметка**")
                st.caption(info_line)

                with st.spinner(f"Выполняю предразметку: {info_line}"):
                    try:
                        preann_results = preannotate_via_service(
                            local_paths,
                            score_threshold,
                            max_boxes,
                            task_type,
                            class_names=class_lines,
                            use_clip_flag=use_clip,
                            use_qwen=use_qwen,
                            qwen_instruction=qwen_instruction,
                        )
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

# --- Постоянный блок превью предразметки (не пропадает при других действиях) ---
previews_persist = st.session_state.get("previews_data", {})
if previews_persist:
    st.write("---")
    st.markdown("**Превью предразметки**")
    # Показываем только одно (первое) превью
    first_name, first_data = sorted(previews_persist.items())[0]
    st.image(first_data, width=300, caption=Path(first_name).name)

st.write("---")
st.header("Выдать доступ на валидацию (review) в CVAT")

current_task_id = st.session_state.get("task_id")
if current_task_id is None:
    st.info("Сначала создайте задачу (чтобы появился `task_id`).")
else:
    st.markdown(f"**task_id:** `{current_task_id}`")

    # Показываем результат в одном месте (без “шумных” блоков)
    result_box = st.empty()

    # Form prevents reruns on each keystroke, so previews won't flicker/disappear
    with st.form("assign_validator_form", clear_on_submit=False):
        reviewer_user = st.text_input("Валидатор (username / email / user id)", value="")
        submitted = st.form_submit_button("Назначить")

    if submitted:
        if not reviewer_user.strip():
            result_box.error("Введите username/email или user id.")
        else:
            with st.spinner("Назначаю валидатора..."):
                try:
                    res = grant_validation_access_for_task(
                        int(current_task_id), reviewer_user.strip(), timeout=120
                    )
                except Exception as e:
                    result_box.error(f"Не удалось назначить: {e}")
                else:
                    failed = res.get("jobs_failed") or []
                    patched = res.get("jobs_patched") or []
                    reviewer_id = res.get("reviewer_id")

                    if failed or not patched:
                        status_codes = []
                        for f in failed:
                            lr = f.get("last_response") or {}
                            sc = lr.get("status_code")
                            if isinstance(sc, int):
                                status_codes.append(sc)
                        if 403 in status_codes:
                            result_box.error("403: нет прав назначать пользователей на job. Используйте admin/owner токен.")
                        else:
                            result_box.error("Не удалось назначить валидатора.")
                        with st.expander("Детали"):
                            st.json(res)
                    else:
                        # Минимальный успех: имя/ввод + (id если введён числом)
                        name = reviewer_user.strip()
                        if name.isdigit() and reviewer_id is not None:
                            name = f"id={reviewer_id}"
                        result_box.success(f"Валидатор назначен: {name}")