---
title: "YOLO"
weight: 2
math: true
---

# YOLO (You Only Look Once)

## 개요

- **논문**: You Only Look Once: Unified, Real-Time Object Detection (2016)
- **저자**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **핵심 기여**: Detection을 단일 회귀 문제로 재정의, 실시간 처리 가능

## 핵심 아이디어

> "Detection을 한 번의 forward pass로 해결"

Two-stage detector와 달리, 이미지를 그리드로 나누고 각 그리드에서 직접 박스와 클래스를 예측합니다.

---

## YOLO 버전 변화

| 버전 | 연도 | 핵심 개선 | mAP (COCO) | FPS |
|------|------|----------|------------|-----|
| YOLOv1 | 2016 | 첫 one-stage detector | 63.4 (VOC) | 45 |
| YOLOv2 | 2016 | Batch Norm, Anchor | 78.6 (VOC) | 67 |
| YOLOv3 | 2018 | Multi-scale, FPN | 33.0 | 30 |
| YOLOv4 | 2020 | CSPNet, Mish | 43.5 | 65 |
| YOLOv5 | 2020 | PyTorch, 편의성 | 50.7 | 140 |
| YOLOv8 | 2023 | Anchor-free, SOTA | 53.9 | 100+ |

---

## YOLOv1 구조

### 동작 원리

```
1. 이미지를 S×S 그리드로 분할 (S=7)
2. 각 그리드 셀이 B개의 bounding box 예측 (B=2)
3. 각 박스: (x, y, w, h, confidence)
4. 각 그리드 셀이 C개 클래스 확률 예측 (C=20)

Output: S × S × (B×5 + C) = 7 × 7 × 30
```

### 네트워크 구조

```
Input (448×448×3)
    ↓
24 Conv Layers (GoogLeNet inspired)
    ↓
2 FC Layers
    ↓
Output (7×7×30)
```

### Confidence Score

$$\text{Confidence} = P(\text{Object}) \times \text{IoU}_{pred}^{truth}$$

### Loss Function

$$L = \lambda_{coord} L_{coord} + L_{obj} + \lambda_{noobj} L_{noobj} + L_{class}$$

- $\lambda_{coord} = 5$: 좌표 loss 가중치
- $\lambda_{noobj} = 0.5$: 객체 없는 셀의 loss 가중치

---

## YOLOv3 구조

### 개선점

1. **Multi-scale Prediction**: 3가지 스케일에서 예측
2. **FPN-like Structure**: 다양한 크기의 객체 검출
3. **Anchor Boxes**: 9개의 anchor (3 scales × 3 ratios)
4. **Darknet-53**: ResNet-like backbone

### 아키텍처

```
Input (416×416)
    ↓
Darknet-53 Backbone
    ↓
┌─────────────────────────────────────┐
│         FPN-like Neck               │
│  52×52 (small) + 26×26 + 13×13 (large) │
└─────────────────────────────────────┘
    ↓
3 Detection Heads
    ↓
Output:
  - 13×13×255 (대형 객체)
  - 26×26×255 (중형 객체)
  - 52×52×255 (소형 객체)

255 = 3 anchors × (4 coords + 1 obj + 80 classes)
```

---

## YOLOv8 구조 (최신)

### 주요 특징

1. **Anchor-free**: anchor box 없이 직접 예측
2. **Decoupled Head**: classification과 regression 분리
3. **C2f Module**: CSP + 더 효율적인 특징 융합
4. **Task-specific Heads**: detection, segmentation, pose 등

### 모델 변형

| 모델 | 파라미터 | mAP | FPS (V100) |
|------|----------|-----|------------|
| YOLOv8n | 3.2M | 37.3 | 430 |
| YOLOv8s | 11.2M | 44.9 | 350 |
| YOLOv8m | 25.9M | 50.2 | 220 |
| YOLOv8l | 43.7M | 52.9 | 160 |
| YOLOv8x | 68.2M | 53.9 | 100 |

---

## 구현 예시

### YOLOv8 (Ultralytics)

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')  # nano 모델

# 추론
results = model('image.jpg')

# 결과 확인
for result in results:
    boxes = result.boxes  # Bounding boxes
    print(boxes.xyxy)     # x1, y1, x2, y2 형식
    print(boxes.conf)     # confidence scores
    print(boxes.cls)      # class indices

# 학습
model.train(data='coco128.yaml', epochs=100)

# Export
model.export(format='onnx')
```

### YOLOv5 (torch hub)

```python
import torch

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 추론
results = model('image.jpg')

# 결과 출력
results.print()
results.show()

# pandas DataFrame으로 결과
df = results.pandas().xyxy[0]
```

---

## YOLO vs Faster R-CNN

| 측면 | YOLO | Faster R-CNN |
|------|------|--------------|
| 방식 | One-stage | Two-stage |
| 속도 | 빠름 (30-100+ FPS) | 느림 (~7 FPS) |
| 정확도 | 상대적으로 낮음 | 높음 |
| 작은 객체 | 어려움 | 상대적으로 좋음 |
| 실시간 | 가능 | 어려움 |
| 구현 | 단순 | 복잡 |

---

## 실제 활용

- **자율주행**: 실시간 객체 검출
- **CCTV 분석**: 사람/차량 검출
- **산업 자동화**: 제품 불량 검출
- **모바일**: 경량화 모델 (YOLOv8n)

---

## 관련 콘텐츠

- [Faster R-CNN](/ko/docs/architecture/detection/faster-rcnn) - Two-stage detector
- [IoU & NMS](/ko/docs/math/iou-nms) - Detection 핵심 개념
- [Anchor Box](/ko/docs/math/anchor) - 박스 예측 방식
- [Detection 태스크](/ko/docs/task/detection) - 평가 지표
