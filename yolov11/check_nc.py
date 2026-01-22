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
# """
#
# Проверка всего датасета на повреждённые JPEG:
# - восстанавливает файлы, если возможно
# - удаляет неподлежащие восстановлению
# - выводит отчёт
# """
#
# from pathlib import Path
# from PIL import Image
# import os
#
# dataset_dir = Path("dataset_yolo")
#
# splits = ["train", "val", "test"]
# corrupt_count = 0
# restored_count = 0
# deleted_count = 0
#
# for split in splits:
#     images_dir = dataset_dir / split / "images"
#     print(f"Checking {split} images in {images_dir} ...")
#
#     for img_path in images_dir.glob("*.jpg"):
#         try:
#             # открываем и загружаем все пиксели
#             with Image.open(img_path) as im:
#                 im.load()
#         except Exception as e:
#             corrupt_count += 1
#             print(f"Corrupt image detected: {img_path} -> {e}")
#             try:
#                 with Image.open(img_path) as im:
#                     im = im.convert("RGB")
#                     im.save(img_path)
#                     restored_count += 1
#                     print(f"Restored {img_path}")
#             except Exception:
#                 os.remove(img_path)
#                 deleted_count += 1
#                 print(f"Deleted {img_path} (cannot restore)")
#
# print(f"Total corrupt images found: {corrupt_count}")
# print(f"Restored: {restored_count}")
# print(f"Deleted: {deleted_count}")




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from pathlib import Path
# from typing import List
#
# # Импортируем ваши трансформации (предполагается, что src/ в PYTHONPATH или запущено из корня проекта)
# from src.augmentations import get_train_transforms, get_val_transforms, parse_yolo_label_file
#
# # -----------------------
# # Вспомогательные функции
# # -----------------------
# def yolo_to_xyxy_abs(yolo_box, img_w, img_h):
#     x_c, y_c, w, h = yolo_box
#     x_c *= img_w; y_c *= img_h; w *= img_w; h *= img_h
#     x1 = x_c - w/2; y1 = y_c - h/2
#     x2 = x_c + w/2; y2 = y_c + h/2
#     return [x1, y1, x2, y2]
#
# def xyxy_to_yolo_norm(box, img_w, img_h):
#     x1, y1, x2, y2 = box
#     bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
#     if img_w == 0 or img_h == 0:
#         return [0,0,0,0]
#     x_c = (x1 + x2) / 2.0 / img_w
#     y_c = (y1 + y2) / 2.0 / img_h
#     return [x_c, y_c, bw / img_w, bh / img_h]
#
# def draw_boxes(img, boxes_xyxy: List[List[float]], labels=None, color=(0,255,0), thickness=2, show=True, title=None):
#     img_vis = img.copy()
#     for i, box in enumerate(boxes_xyxy):
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
#         if labels is not None:
#             cv2.putText(img_vis, str(labels[i]), (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#     if show:
#         if title:
#             plt.figure(figsize=(8,8)); plt.title(title)
#         plt.axis('off'); plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)); plt.show()
#     return img_vis
#
# def tensor_to_image(tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
#     # tensor: torch.Tensor CxHxW normalized by A.Normalize + ToTensorV2 (float)
#     t = tensor.detach().cpu().numpy()
#     t = np.transpose(t, (1,2,0))  # HWC
#     # unnormalize
#     t = (t * std) + mean
#     t = np.clip(t * 255.0, 0, 255).astype(np.uint8)
#     return t
#
# # -----------------------
# # Визуальный тест
# # -----------------------
# def visual_test(image_path, label_path=None, img_size=640):
#     # load image
#     img = cv2.imread(str(image_path))
#     if img is None:
#         raise FileNotFoundError(image_path)
#     H, W = img.shape[:2]
#
#     # load labels (if есть) — возвращает normalized yolo boxes
#     if label_path and Path(label_path).exists():
#         yolo_boxes, labels = parse_yolo_label_file(str(label_path))
#     else:
#         # пример: один центрный bbox
#         labels = [0]
#         yolo_boxes = [[0.5, 0.5, 0.4, 0.3]]
#
#     # показ исходного
#     boxes_xyxy = [yolo_to_xyxy_abs(b, W, H) for b in yolo_boxes]
#     print("Исходные абсолютные bbox (xyxy):", boxes_xyxy)
#     draw_boxes(img, boxes_xyxy, labels=labels, title="Original")
#
#     # применить train трансформ
#     t_train = get_train_transforms(img_size)
#     out = t_train(image=img, bboxes=yolo_boxes, category_ids=labels)
#     img_t = out['image']         # tensor CxHxW
#     bboxes_t = out.get('bboxes', [])
#     labels_t = out.get('category_ids', [])
#
#     # визуализация: bboxes_t — в формате yolo (normalized) относительно transformed image size
#     # получим размер transformed image
#     _, Ht, Wt = img_t.shape
#     boxes_xyxy_t = [yolo_to_xyxy_abs(b, Wt, Ht) for b in bboxes_t]
#     print("После train transform (xyxy abs):", boxes_xyxy_t)
#     img_vis = tensor_to_image(img_t)
#     draw_boxes(img_vis, boxes_xyxy_t, labels=labels_t, title="After train transform")
#
#     # применить val трансформ
#     t_val = get_val_transforms(img_size)
#     out_v = t_val(image=img, bboxes=yolo_boxes, category_ids=labels)
#     img_v = out_v['image']
#     bboxes_v = out_v.get('bboxes', [])
#     _, Hv, Wv = img_v.shape
#     boxes_xyxy_v = [yolo_to_xyxy_abs(b, Wv, Hv) for b in bboxes_v]
#     print("После val transform (xyxy abs):", boxes_xyxy_v)
#     img_vis_v = tensor_to_image(img_v)
#     draw_boxes(img_vis_v, boxes_xyxy_v, labels=out_v.get('category_ids', []), title="After val transform")
#     return True
#
# # -----------------------
# # Автотесты (простые)
# # -----------------------
# def unit_tests(tmp_image=None):
#     # 1) Типы и размеры
#     if tmp_image is None:
#         tmp = np.zeros((800,1200,3), dtype=np.uint8) + 128
#         cv2.rectangle(tmp, (300,200),(900,600),(0,255,0),-1)
#         tmp_image = tmp
#     H,W = tmp_image.shape[:2]
#     yolo_box = [[(300+900)/2/W, (200+600)/2/H, (900-300)/W, (600-200)/H]]  # normalized
#     labels = [0]
#
#     t = get_train_transforms(640)
#     out = t(image=tmp_image, bboxes=yolo_box, category_ids=labels)
#     assert 'image' in out and 'bboxes' in out and 'category_ids' in out, "transform output missing keys"
#     img_t = out['image']
#     assert isinstance(img_t, torch.Tensor), "image is not tensor"
#     assert img_t.ndim == 3 and img_t.shape[0] == 3, f"tensor shape unexpected: {img_t.shape}"
#
#     # 2) check conversion round-trip approx
#     _, Ht, Wt = img_t.shape
#     b = out['bboxes'][0]
#     xy = yolo_to_xyxy_abs(b, Wt, Ht)
#     yolo_round = xyxy_to_yolo_norm(xy, Wt, Ht)
#     # allow small float error
#     for a,bv in zip(b, yolo_round):
#         assert abs(a - bv) < 1e-3, f"round-trip mismatch {a} vs {bv}"
#
#     # 3) min_visibility: create tiny bbox and check removal
#     tiny_box = [[0.01, 0.01, 0.01, 0.01]]  # almost invisible
#     out2 = t(image=tmp_image, bboxes=tiny_box, category_ids=[0])
#     # because min_visibility=0.3 default, tiny should be removed
#     assert len(out2.get('bboxes', [])) == 0, "min_visibility did not remove tiny box"
#
#     print("All unit tests passed.")
#     return True
#
# # -----------------------
# # Usage example (Colab/local)
# # -----------------------
# if __name__ == "__main__":
#     # укажи путь к любому image в твоём датасете или оставь None для синтетики
#     sample_img = "dataset_yolo/train/images/train_0.jpg"  # <-- поправь
#     sample_lbl = "dataset_yolo/train/labels/train_0.txt"  # <-- поправь
#
#     # Визуальная проверка (если хочешь — пропусти и запусти unit_tests)
#     try:
#         visual_test(sample_img, sample_lbl, img_size=640)
#     except Exception as e:
#         print("Visual test failed:", e)
#         print("Запускаю unit tests на синтетическом изображении")
#         unit_tests()

# debug_aug.py
import cv2
import matplotlib.pyplot as plt
from src.augmentations import get_train_transforms, parse_yolo_label_file

img = cv2.imread('dataset_yolo/train/images/train_0.jpg')  # замени на реальный путь
bboxes, labels = parse_yolo_label_file('dataset_yolo/train/labels/train_0.txt')
print('parsed:', bboxes, labels)
t = get_train_transforms(640)
out = t(image=img, bboxes=bboxes, category_ids=labels)
img_t = out['image']
if hasattr(img_t, 'permute'):
    im = (img_t.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
else:
    im = img_t
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
