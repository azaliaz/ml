# cvat_client.py
import os
import time
from pathlib import Path
from typing import List, Optional

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from cvat_sdk.api_client.exceptions import ApiException

CVAT_URL = os.environ.get("CVAT_URL", "https://app.cvat.ai")
ACCESS_TOKEN = os.environ.get("CVAT_TOKEN")

if not ACCESS_TOKEN:
    raise RuntimeError("Set CVAT_TOKEN environment variable before running (export CVAT_TOKEN=...)")

def _wait_for_task_size(client, task_id: int, expected_count: int, timeout: int = 300, poll_interval: float = 2.0) -> bool:
    """Ожидаем, пока task.size станет >= expected_count. Возвращает True если готово, иначе False."""
    start = time.time()
    while True:
        task = client.tasks.retrieve(task_id)
        size = getattr(task, "size", None)
        try:
            size_int = int(size) if size is not None else 0
        except Exception:
            size_int = 0
        # debug print could be logged
        # print(f"poll task {task_id} size={size_int}")
        if size_int >= expected_count and expected_count > 0:
            return True
        if time.time() - start > timeout:
            return False
        time.sleep(poll_interval)

def create_task_and_upload(local_files: List[str], task_name: str = "streamlit_task", labels: Optional[List[dict]] = None, timeout: int = 600):
    """
    Создаёт задачу в CVAT и загружает локальные файлы.
    Возвращает task_id (int).
    """
    if not local_files:
        raise ValueError("local_files must contain at least one path")

    with make_client(CVAT_URL, access_token=ACCESS_TOKEN) as client:
        task_spec = {"name": task_name, "labels": labels or [{"name": "object"}]}
        try:
            task = client.tasks.create_from_data(
                spec=task_spec,
                resource_type=ResourceType.LOCAL,
                resources=list(local_files),
            )
        except ApiException as e:
            raise RuntimeError(f"Failed to create/upload task: {e}") from e

        task_id = getattr(task, "id", None)
        if task_id is None:
            raise RuntimeError("Failed to get task id after create_from_data")

        # Ждём пока сервер обработает загрузку (size >= number of files)
        ok = _wait_for_task_size(client, task_id, expected_count=len(local_files), timeout=timeout)
        if not ok:
            # не фатально — возвращаем task_id, но предупреждаем
            # Вы можете увеличить timeout при вызове
            print(f"Warning: task {task_id} not ready after {timeout}s (size may be different). Proceeding anyway.")
        return int(task_id)

def export_annotations_by_id(task_id: int, out_zip_path: str, format_name: str = "COCO 1.0", include_images: bool = False, timeout: int = 300):
    """
    Экспорт аннотаций по task_id. Возвращает путь к скачанному ZIP.
    include_images - если True и ваш аккаунт CVAT Cloud не поддерживает, сервер вернёт 403.
    """
    with make_client(CVAT_URL, access_token=ACCESS_TOKEN) as client:
        try:
            task = client.tasks.retrieve(task_id)
        except ApiException as e:
            raise RuntimeError(f"Failed to retrieve task {task_id}: {e}") from e

        # Попробуем экспортировать; SDK сохранит файл в out_zip_path
        try:
            task.export_dataset(format_name, out_zip_path, include_images=include_images)
            return out_zip_path
        except ApiException as e:
            # прокинем понятную ошибку (например 403 если include_images запрещён)
            raise RuntimeError(f"Export failed: {e}") from e
