---
title: "Mask R-CNN"
weight: 2
math: true
---

# Mask R-CNN

## 개요

- **논문**: Mask R-CNN (2017)
- **저자**: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick (FAIR)
- **핵심 기여**: Faster R-CNN에 mask branch를 추가하여 instance segmentation

## 핵심 아이디어

> "Faster R-CNN + Mask Branch = Instance Segmentation"

Detection과 segmentation을 동시에 수행하는 multi-task 모델입니다.

---

## 구조

### 전체 아키텍처

```
Input Image
    ↓
Backbone (ResNet + FPN)
    ↓
Region Proposal Network (RPN)
    ↓
RoI Align (not RoI Pooling!)
    ↓
┌─────────────────────────────────────────┐
│           Three Branches                 │
├─────────────┬─────────────┬─────────────┤
│ Classification │ Box Regression │ Mask Prediction │
│   (class)   │   (x,y,w,h)  │  (28×28 binary) │
└─────────────┴─────────────┴─────────────┘
```

### Faster R-CNN과의 차이

| 구성 요소 | Faster R-CNN | Mask R-CNN |
|----------|--------------|------------|
| RoI 처리 | RoI Pooling | **RoI Align** |
| 출력 | class, box | class, box, **mask** |
| Backbone | ResNet | ResNet + **FPN** |

---

## 핵심 컴포넌트

### 1. RoI Align

RoI Pooling의 양자화 문제를 해결:

**RoI Pooling 문제:**
```
RoI: 10.6 → 10 (반올림)
실제 위치와 오차 발생 → segmentation 품질 저하
```

**RoI Align 해결:**
```
양자화 없이 bilinear interpolation
정확한 위치에서 특징 추출
```

### 2. Mask Branch

각 RoI에 대해 28×28 크기의 마스크 예측:

```python
# Mask Head 구조
FCN (4 Conv layers) → Deconv → 28×28×num_classes
```

**특징:**
- 클래스별로 독립적인 마스크 예측
- Binary mask (sigmoid)
- 최종 마스크는 box 크기로 resize

### 3. FPN (Feature Pyramid Network)

다양한 스케일의 객체 검출을 위한 feature pyramid:

```
P5 (작은 해상도, 큰 객체)
  ↑
P4
  ↑
P3
  ↑
P2 (큰 해상도, 작은 객체)
```

---

## 학습

### Multi-task Loss

$$L = L_{cls} + L_{box} + L_{mask}$$

- $L_{cls}$: Classification loss (cross-entropy)
- $L_{box}$: Bounding box regression (smooth L1)
- $L_{mask}$: Binary cross-entropy (픽셀별)

### Mask Loss

Ground truth 클래스에 해당하는 마스크에서만 loss 계산:

$$L_{mask} = -\frac{1}{m^2} \sum_{i,j} [y_{ij} \log \hat{y}_{ij} + (1-y_{ij}) \log(1-\hat{y}_{ij})]$$

---

## 구현 예시

```python
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# 사전 학습된 모델 로드
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 추론
with torch.no_grad():
    predictions = model([image])

# 결과 형태
pred = predictions[0]
print(pred['boxes'].shape)   # (N, 4) - bounding boxes
print(pred['labels'].shape)  # (N,) - class labels
print(pred['scores'].shape)  # (N,) - confidence scores
print(pred['masks'].shape)   # (N, 1, H, W) - instance masks
```

### 커스텀 학습

```python
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = maskrcnn_resnet50_fpn(pretrained=True)
num_classes = 10

# Box predictor 교체
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Mask predictor 교체
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes
)
```

---

## 성능

### COCO Dataset

| Backbone | box AP | mask AP |
|----------|--------|---------|
| ResNet-50-FPN | 38.2 | 34.7 |
| ResNet-101-FPN | 40.0 | 36.1 |
| ResNeXt-101-FPN | 41.3 | 37.1 |

---

## Mask R-CNN의 확장

### Keypoint Detection

Mask branch 대신 keypoint branch 추가:

```python
from torchvision.models.detection import keypointrcnn_resnet50_fpn

model = keypointrcnn_resnet50_fpn(pretrained=True)
# 17개 keypoint (COCO person keypoints)
```

### Panoptic Segmentation

Instance + semantic segmentation 통합

---

## 한계점

- Two-stage라 속도 제한 (~5 FPS)
- 겹치는 객체 처리 어려움
- 작은 객체 segmentation 품질 저하

---

## 관련 콘텐츠

- [Faster R-CNN](/ko/docs/architecture/detection/faster-rcnn) - 기반 모델
- [U-Net](/ko/docs/architecture/segmentation/unet) - Semantic segmentation
- [SAM](/ko/docs/architecture/segmentation/sam) - Promptable segmentation
- [ResNet](/ko/docs/architecture/cnn/resnet) - Backbone
- [Segmentation 태스크](/ko/docs/task/segmentation) - 평가 지표
