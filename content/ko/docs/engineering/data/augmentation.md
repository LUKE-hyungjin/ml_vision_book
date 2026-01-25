---
title: "데이터 증강"
weight: 1
---

# 데이터 증강 (Data Augmentation)

## 개요

데이터 증강은 기존 데이터에 변환을 적용해 학습 데이터를 늘리는 기법입니다.

---

## 기본 증강 기법

### 기하학적 변환

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
])
```

| 기법 | 설명 | 주의사항 |
|------|------|----------|
| Flip | 좌우/상하 반전 | 방향 의미 있는 태스크 주의 |
| Rotate | 회전 | Detection bbox 변환 필요 |
| Scale | 크기 조정 | 객체 크기 분포 고려 |
| Crop | 랜덤 크롭 | 객체 잘림 확인 |

### 색상 변환

```python
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=7, p=0.3),
])
```

---

## 고급 증강 기법

### Cutout / Random Erasing

이미지의 일부를 가려서 robustness 향상:

```python
transform = A.Compose([
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        fill_value=0,
        p=0.5
    ),
])
```

### MixUp

두 이미지와 레이블을 혼합:

```python
def mixup(x1, y1, x2, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
```

### CutMix

한 이미지의 일부를 다른 이미지로 대체:

```python
def cutmix(x1, y1, x2, y2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)

    # 박스 좌표 계산
    H, W = x1.shape[1:3]
    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)

    cx, cy = np.random.randint(W), np.random.randint(H)
    x1_box = np.clip(cx - cut_w // 2, 0, W)
    x2_box = np.clip(cx + cut_w // 2, 0, W)
    y1_box = np.clip(cy - cut_h // 2, 0, H)
    y2_box = np.clip(cy + cut_h // 2, 0, H)

    # 이미지 합성
    x = x1.clone()
    x[:, y1_box:y2_box, x1_box:x2_box] = x2[:, y1_box:y2_box, x1_box:x2_box]

    # 레이블 비율 조정
    lam = 1 - (y2_box - y1_box) * (x2_box - x1_box) / (H * W)
    y = lam * y1 + (1 - lam) * y2

    return x, y
```

### Mosaic (YOLO)

4개의 이미지를 하나로 합침:

```
┌─────┬─────┐
│ img1│ img2│
├─────┼─────┤
│ img3│ img4│
└─────┴─────┘
```

---

## AutoAugment

### 개념

최적의 증강 정책을 자동으로 탐색:

```python
from torchvision.transforms import autoaugment

transform = autoaugment.AutoAugment(
    policy=autoaugment.AutoAugmentPolicy.IMAGENET
)
```

### 변형들

| 방법 | 특징 |
|------|------|
| AutoAugment | 강화학습으로 탐색 |
| RandAugment | 단순화된 무작위 선택 |
| TrivialAugment | 더 단순화 |

### RandAugment

```python
from torchvision.transforms import RandAugment

transform = RandAugment(
    num_ops=2,      # 적용할 변환 개수
    magnitude=9,    # 변환 강도 (0-30)
)
```

---

## Detection용 증강

Detection에서는 bbox도 함께 변환해야 합니다:

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.RandomCrop(width=640, height=640, p=0.5),
], bbox_params=A.BboxParams(
    format='pascal_voc',  # [x_min, y_min, x_max, y_max]
    min_area=100,         # 너무 작은 bbox 제거
    min_visibility=0.3,   # 잘린 bbox 최소 비율
))

# 사용
transformed = transform(
    image=image,
    bboxes=bboxes,
    class_labels=labels
)
```

---

## Segmentation용 증강

마스크도 동일하게 변환:

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ElasticTransform(p=0.3),  # 의료영상에 효과적
])

# 사용
transformed = transform(image=image, mask=mask)
```

---

## 도메인별 증강

### 의료 영상

```python
transform = A.Compose([
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),
    A.GridDistortion(p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.5),  # 대비 향상
])
```

### 위성/항공 영상

```python
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])
```

### 야간/저조도

```python
transform = A.Compose([
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.GaussNoise(var_limit=(10, 100), p=0.5),
])
```

---

## 증강 전략

### Train vs Validation

```python
# Train: 강한 증강
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandAugment(num_ops=2, magnitude=9),
    A.Normalize(),
])

# Validation: 증강 없음 또는 최소화
val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(),
])
```

### TTA (Test-Time Augmentation)

추론 시 여러 증강 적용 후 평균:

```python
def tta_predict(model, image, transforms):
    predictions = []
    for transform in transforms:
        augmented = transform(image=image)['image']
        pred = model(augmented)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

tta_transforms = [
    A.Compose([]),  # 원본
    A.Compose([A.HorizontalFlip(p=1.0)]),
    A.Compose([A.VerticalFlip(p=1.0)]),
]
```

---

## 주의사항

1. **과도한 증강**: 오히려 성능 저하 가능
2. **태스크 특성 고려**: 좌우 방향이 중요한 경우 Flip 주의
3. **검증 세트**: 증강 없이 평가해야 공정한 비교
4. **계산 비용**: 실시간 증강 vs 사전 증강 트레이드오프

---

## 관련 콘텐츠

- [데이터 파이프라인](/ko/docs/engineering/data/pipeline) - DataLoader 최적화
- [Classification](/ko/docs/task/classification) - 분류 태스크
- [Detection](/ko/docs/task/detection) - 객체 탐지 태스크

