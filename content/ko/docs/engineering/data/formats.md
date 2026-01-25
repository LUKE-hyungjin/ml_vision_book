---
title: "데이터 포맷"
weight: 2
---

# 데이터 포맷

## 개요

Vision 태스크별로 다양한 데이터 포맷이 사용됩니다.

---

## Classification

### ImageFolder 구조

```
dataset/
├── train/
│   ├── cat/
│   │   ├── cat_001.jpg
│   │   └── cat_002.jpg
│   └── dog/
│       ├── dog_001.jpg
│       └── dog_002.jpg
└── val/
    ├── cat/
    └── dog/
```

```python
from torchvision.datasets import ImageFolder

dataset = ImageFolder(root='dataset/train', transform=transform)
```

### CSV 형식

```csv
image_path,label
train/cat_001.jpg,cat
train/dog_001.jpg,dog
```

---

## Object Detection

### COCO Format

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 150],  // [x, y, width, height]
      "area": 30000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "cat"}
  ]
}
```

### Pascal VOC Format (XML)

```xml
<annotation>
  <filename>image1.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>100</ymin>
      <xmax>300</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

### YOLO Format (TXT)

```
# class_id x_center y_center width height (정규화된 값)
0 0.5 0.4 0.3 0.2
1 0.2 0.6 0.1 0.15
```

디렉토리 구조:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## Segmentation

### COCO Segmentation

```json
{
  "annotations": [
    {
      "segmentation": [[x1,y1,x2,y2,...]], // polygon
      "area": 1000,
      "iscrowd": 0,
      "image_id": 1,
      "bbox": [100, 100, 200, 150],
      "category_id": 1
    }
  ]
}
```

### Mask 이미지

```
dataset/
├── images/
│   └── image1.jpg
└── masks/
    └── image1.png  # 클래스별 픽셀값 (0: background, 1: class1, ...)
```

### RLE (Run-Length Encoding)

압축된 마스크 표현:

```python
from pycocotools import mask as mask_util

# Mask → RLE
rle = mask_util.encode(np.asfortranarray(binary_mask))

# RLE → Mask
mask = mask_util.decode(rle)
```

---

## 포맷 변환

### COCO ↔ YOLO

```python
def coco_to_yolo(bbox, img_width, img_height):
    """COCO [x, y, w, h] → YOLO [x_center, y_center, w, h] (normalized)"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]

def yolo_to_coco(bbox, img_width, img_height):
    """YOLO → COCO"""
    x_center, y_center, w_norm, h_norm = bbox
    w = w_norm * img_width
    h = h_norm * img_height
    x = x_center * img_width - w / 2
    y = y_center * img_height - h / 2
    return [x, y, w, h]
```

### VOC ↔ COCO

```python
def voc_to_coco(bbox):
    """VOC [xmin, ymin, xmax, ymax] → COCO [x, y, w, h]"""
    xmin, ymin, xmax, ymax = bbox
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def coco_to_voc(bbox):
    """COCO → VOC"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]
```

---

## 변환 도구

### FiftyOne

```python
import fiftyone as fo
import fiftyone.utils.coco as fouc

# COCO → FiftyOne
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="images",
    labels_path="annotations.json"
)

# FiftyOne → YOLO
dataset.export(
    export_dir="yolo_dataset",
    dataset_type=fo.types.YOLOv5Dataset,
)
```

### Roboflow

온라인 도구로 다양한 포맷 간 변환 지원

---

## 데이터셋 검증

### 기본 검증

```python
import json
from pathlib import Path

def validate_coco(json_path, image_dir):
    with open(json_path) as f:
        coco = json.load(f)

    errors = []

    # 이미지 파일 존재 확인
    for img in coco['images']:
        if not (Path(image_dir) / img['file_name']).exists():
            errors.append(f"Missing image: {img['file_name']}")

    # bbox 유효성 확인
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            errors.append(f"Invalid bbox: {ann['id']}")

    return errors
```

### FiftyOne으로 시각화

```python
import fiftyone as fo

dataset = fo.Dataset.from_dir(...)
session = fo.launch_app(dataset)
```

---

## 관련 콘텐츠

- [데이터 파이프라인](/ko/docs/engineering/data/pipeline) - 데이터 로딩 최적화
- [레이블링](/ko/docs/engineering/data/labeling) - 어노테이션 도구
- [Detection](/ko/docs/task/detection) - Detection 태스크

