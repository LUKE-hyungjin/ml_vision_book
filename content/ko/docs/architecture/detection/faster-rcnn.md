---
title: "Faster R-CNN"
weight: 1
math: true
---

# Faster R-CNN

## 개요

- **논문**: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015)
- **저자**: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- **핵심 기여**: Region Proposal Network(RPN)으로 end-to-end 학습 가능한 detector

## R-CNN 계보

```
R-CNN (2014)
    ↓ ROI Pooling 도입
Fast R-CNN (2015)
    ↓ RPN 도입
Faster R-CNN (2015)
```

| 모델 | Region Proposal | 속도 |
|------|-----------------|------|
| R-CNN | Selective Search | ~47s/image |
| Fast R-CNN | Selective Search | ~2s/image |
| Faster R-CNN | RPN (학습됨) | ~0.2s/image |

---

## 구조

### 전체 아키텍처

```
Input Image
    ↓
Backbone (ResNet/VGG) → Feature Map
    ↓
┌───────────────────────────────────┐
│     Region Proposal Network       │
│  Feature Map → Anchors → RoIs     │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│        RoI Pooling/Align          │
│  다양한 크기 RoI → 고정 크기 특징   │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│    Detection Head (R-CNN Head)    │
│  Classification + Box Regression  │
└───────────────────────────────────┘
    ↓
Output: [class, x, y, w, h]
```

---

## 핵심 컴포넌트

### 1. Backbone

이미지에서 특징 맵 추출:
- VGG-16: 원 논문
- ResNet-50/101: 현재 주로 사용
- FPN: Feature Pyramid Network 추가 시 성능 향상

### 2. Region Proposal Network (RPN)

Feature map 위에서 객체 후보 영역 생성:

```
Feature Map (H×W×C)
    ↓
3×3 Conv
    ↓
┌─────────────────────────────────┐
│  1×1 Conv (cls)    1×1 Conv (reg) │
│  2k scores        4k coordinates  │
└─────────────────────────────────┘
```

**Anchor Box:**
- 각 위치에서 k개의 앵커 (기본: 9개)
- 3가지 크기 × 3가지 비율 = 9 anchors
- 크기: 128², 256², 512²
- 비율: 1:1, 1:2, 2:1

### 3. RoI Pooling / RoI Align

다양한 크기의 RoI를 고정 크기(예: 7×7)로 변환:

**RoI Pooling**: 양자화로 인한 오차 발생
**RoI Align**: 양자화 없이 bilinear interpolation 사용 (Mask R-CNN에서 제안)

### 4. Detection Head

각 RoI에 대해:
- Classification: softmax로 클래스 예측
- Box Regression: bounding box 좌표 미세 조정

---

## 학습

### Multi-task Loss

$$L = L_{cls} + \lambda L_{reg}$$

**Classification Loss (Cross-entropy):**
$$L_{cls} = -\log p_{c^*}$$

**Regression Loss (Smooth L1):**
$$L_{reg} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t_i - t_i^*)$$

### Box Parameterization

$$t_x = (x - x_a) / w_a, \quad t_y = (y - y_a) / h_a$$
$$t_w = \log(w / w_a), \quad t_h = \log(h / h_a)$$

### Training Strategy

1. RPN과 R-CNN Head를 번갈아 학습 (원 논문)
2. 또는 end-to-end joint training (현재 주로 사용)

---

## 구현 예시

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 사전 학습된 모델 로드
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 추론
with torch.no_grad():
    predictions = model([image])

# 결과 형태
# predictions[0]['boxes']: (N, 4) - bounding boxes
# predictions[0]['labels']: (N,) - class labels
# predictions[0]['scores']: (N,) - confidence scores
```

### 커스텀 클래스로 학습

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# backbone 고정, head만 교체
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 10  # 배경 포함

# Box predictor 교체
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

---

## 성능

### COCO Dataset

| Backbone | mAP | mAP@50 |
|----------|-----|--------|
| VGG-16 | 21.9 | 42.7 |
| ResNet-50 | 35.7 | 57.0 |
| ResNet-101 | 37.4 | 58.8 |
| ResNet-50-FPN | 37.0 | 58.5 |

---

## 한계점

- Two-stage 방식으로 속도 제한
- 작은 객체 검출 어려움 (FPN으로 완화)
- Anchor 설계에 도메인 지식 필요

---

## 관련 콘텐츠

- [YOLO](/ko/docs/architecture/detection/yolo) - One-stage detector
- [Mask R-CNN](/ko/docs/architecture/segmentation/mask-rcnn) - Faster R-CNN + segmentation
- [ResNet](/ko/docs/architecture/cnn/resnet) - 주로 사용되는 backbone
- [IoU & NMS](/ko/docs/math/iou-nms) - Detection 핵심 개념
- [Detection 태스크](/ko/docs/task/detection) - 평가 지표
