# from pathlib import Path
#
# dirs = ["dataset_yolo/train/labels", "dataset_yolo/val/labels", "dataset_yolo/test/labels"]
# all_classes = set()
#
# for d in dirs:
#     labels_dir = Path(d)
#     for txt_file in labels_dir.glob("*.txt"):
#         with open(txt_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 1:
#                     all_classes.add(int(float(parts[0])))
#
# print("Всего уникальных классов в датасете:", len(all_classes))
# print("Классы:", sorted(all_classes))
# !/usr/bin/env python3
"""

Проверка всего датасета на повреждённые JPEG:
- восстанавливает файлы, если возможно
- удаляет неподлежащие восстановлению
- выводит отчёт
"""

from pathlib import Path
from PIL import Image
import os

dataset_dir = Path("dataset_yolo")

splits = ["train", "val", "test"]
corrupt_count = 0
restored_count = 0
deleted_count = 0

for split in splits:
    images_dir = dataset_dir / split / "images"
    print(f"Checking {split} images in {images_dir} ...")

    for img_path in images_dir.glob("*.jpg"):
        try:
            # открываем и загружаем все пиксели
            with Image.open(img_path) as im:
                im.load()
        except Exception as e:
            corrupt_count += 1
            print(f"Corrupt image detected: {img_path} -> {e}")
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    im.save(img_path)
                    restored_count += 1
                    print(f"Restored {img_path}")
            except Exception:
                os.remove(img_path)
                deleted_count += 1
                print(f"Deleted {img_path} (cannot restore)")

print(f"Total corrupt images found: {corrupt_count}")
print(f"Restored: {restored_count}")
print(f"Deleted: {deleted_count}")



