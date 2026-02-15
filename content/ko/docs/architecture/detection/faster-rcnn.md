---
title: "Faster R-CNN"
weight: 1
math: true
---

# Faster R-CNN

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d) · [Anchor Box](/ko/docs/components/detection/anchor) · [NMS](/ko/docs/components/detection/nms)
{{% /hint %}}

## 한 줄 요약
> **Faster R-CNN은 `후보 영역 제안(RPN)`과 `정교한 분류/박스 보정(2-stage head)`을 결합해, 정확도를 높인 대표적인 2-stage detector입니다.**

## 왜 이 모델인가?
초기 detector는 "물체 후보 영역"을 만들 때 Selective Search 같은 외부 알고리즘에 크게 의존했습니다.
이 단계가 느리고 end-to-end 학습이 어려워 전체 파이프라인 병목이 되었습니다.

Faster R-CNN은 이 후보 영역 생성기를 **신경망(RPN)** 으로 내부화해,
"후보 찾기"와 "분류/박스 보정"을 한 프레임워크 안에서 학습 가능하게 만든 모델입니다.

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

## 초보자 디버깅 체크리스트
- [ ] **RPN 제안 품질 확인**: proposal 개수는 많은데 GT와 IoU가 낮지 않은가?
- [ ] **NMS 임계값 점검**: 너무 낮으면 과억제, 너무 높으면 중복 박스 증가
- [ ] **RoI 단계 입력 좌표 스케일 확인**: 원본 해상도 좌표와 feature map 좌표를 혼용하지 않았는가?
- [ ] **분류/박스 손실 균형 확인**: 한쪽 loss만 과도하게 커지며 학습을 지배하지 않는가?

## 자주 하는 실수 (FAQ)
**Q1. proposal 수를 늘리면 성능이 무조건 좋아지나요?**  
A. 아닙니다. recall은 오를 수 있지만 계산량이 커지고 FP도 함께 늘 수 있습니다. 보통 상위 proposal 수를 점진적으로 늘려 검증합니다.

**Q2. Faster R-CNN은 YOLO보다 항상 정확한가요?**  
A. 데이터/백본/학습 설정에 따라 다릅니다. 일반적으로 two-stage가 localization에 강한 경향은 있지만, 최신 one-stage도 매우 높은 정확도를 냅니다.

**Q3. mAP@50은 괜찮은데 mAP@75가 낮습니다. 왜 그런가요?**  
A. 클래스 분류는 맞지만 박스 정렬 정밀도가 부족한 경우가 많습니다. RPN anchor 설정, box regression 학습률, RoI Align 설정을 우선 점검하세요.

---

## 관련 콘텐츠

- [YOLO](/ko/docs/architecture/detection/yolo) - One-stage detector
- [Mask R-CNN](/ko/docs/architecture/segmentation/mask-rcnn) - Faster R-CNN + segmentation
- [ResNet](/ko/docs/architecture/cnn/resnet) - 주로 사용되는 backbone
- [IoU](/ko/docs/components/detection/iou) & [NMS](/ko/docs/components/detection/nms) - Detection 핵심 개념
- [Detection 태스크](/ko/docs/task/detection) - 평가 지표
