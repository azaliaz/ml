# app.py
import streamlit as st
from pathlib import Path
import zipfile
import os

from cvat_client import create_task_and_upload, export_annotations_by_id

# Папки для временных файлов
UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

st.title("CVAT Авто-загрузка и выгрузка")

# --- Загрузка файлов ---
st.header("Загрузить файлы в CVAT")
uploaded_files = st.file_uploader(
    "Выберите изображения или видео",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "mp4", "mov"],
    accept_multiple_files=True
)
# ws6sJAG1.kjc6vFTAHf0E4b4tDklmjP1jKYNCnPQY


task_name = st.text_input("Название задачи", value="streamlit_task")

if st.button("Загрузить в CVAT") and uploaded_files:
    local_paths = []
    for f in uploaded_files:
        path = UPLOAD_DIR / f.name
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        local_paths.append(str(path))

    with st.spinner("Загрузка в CVAT..."):
        try:
            task_id = create_task_and_upload(local_paths, task_name)
        except Exception as e:
            st.error(f"Ошибка при загрузке: {e}")
        else:
            st.session_state["task_id"] = task_id
            st.session_state["uploaded_files"] = local_paths
            st.success(f"Загрузка завершена. ID задачи: {task_id}")

# --- Скачивание аннотаций ---
st.header("Скачать аннотации из CVAT")
export_format = st.selectbox("Формат экспорта", ["COCO 1.0", "YOLO 1.1", "Pascal VOC 1.1"])

if "task_id" in st.session_state:
    task_id = st.session_state["task_id"]
    uploaded_files = st.session_state.get("uploaded_files", [])
    st.markdown(f"**Текущий task_id:** `{task_id}`")

    if st.button("Скачать аннотации (только метки)") :
        export_path = EXPORT_DIR / f"{task_name}_annotations.zip"
        with st.spinner("Экспортируем аннотации..."):
            try:
                export_annotations_by_id(int(task_id), str(export_path), format_name=export_format, include_images=False)
            except Exception as e:
                st.error(f"Ошибка при экспорте аннотаций: {e}")
            else:
                with open(export_path, "rb") as f:
                    st.download_button(
                        label="Скачать аннотации ZIP",
                        data=f.read(),
                        file_name=export_path.name,
                        mime="application/zip"
                    )

    st.write("---")

    # кнопка для скачивания изображений (локально сохранённых)
    if uploaded_files:
        if st.button("Скачать изображения (локально загружённые)"):
            images_zip_path = EXPORT_DIR / f"{task_name}_images.zip"
            # создаём архив только по нажатию
            with zipfile.ZipFile(images_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in uploaded_files:
                    if os.path.exists(p):
                        zf.write(p, arcname=Path(p).name)
            with open(images_zip_path, "rb") as f:
                st.download_button(
                    label="Скачать изображения ZIP",
                    data=f.read(),
                    file_name=images_zip_path.name,
                    mime="application/zip"
                )
    else:
        st.info("Нет локально загруженных изображений для скачивания.")
else:
    st.warning("Сначала загрузите файлы в CVAT!")
