---
title: "IoU"
weight: 1
math: true
---

# IoU (Intersection over Union)

## 개요

IoU는 두 영역의 겹침 정도를 측정하는 지표입니다. 객체 검출의 핵심 메트릭입니다.

## 수식

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{교집합 넓이}}{\text{합집합 넓이}}
$$

- 범위: 0 (겹침 없음) ~ 1 (완전 일치)

## 박스 IoU 계산

```python
def box_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2] 형식
    """
    # 교집합
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 각 박스 넓이
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 합집합
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0
```

## 배치 IoU (Vectorized)

```python
import torch

def batch_iou(boxes1, boxes2):
    """
    boxes1: (N, 4), boxes2: (M, 4)
    Returns: (N, M) IoU matrix
    """
    # 교집합 좌표
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # 넓이
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection

    return intersection / union
```

## GIoU (Generalized IoU)

겹치지 않는 박스에도 그라디언트 제공:

$$
\text{GIoU} = \text{IoU} - \frac{|C - A \cup B|}{|C|}
$$

C: 두 박스를 감싸는 최소 박스

```python
def giou(box1, box2):
    iou = box_iou(box1, box2)

    # 최소 외접 박스
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])

    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    union = # ... (위 코드 참조)

    return iou - (c_area - union) / c_area
```

## DIoU (Distance IoU)

중심점 거리 고려:

$$
\text{DIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2}
$$

- ρ: 두 중심점 거리
- c: 외접 박스 대각선 길이

```python
def diou(box1, box2):
    iou = box_iou(box1, box2)

    # 중심점
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    rho2 = (cx1 - cx2)**2 + (cy1 - cy2)**2

    # 외접 박스 대각선
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])

    c2 = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2

    return iou - rho2 / c2
```

## CIoU (Complete IoU)

종횡비까지 고려:

$$
\text{CIoU} = \text{IoU} - \frac{\rho^2}{c^2} - \alpha v
$$

$$
v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2
$$

$$
\alpha = \frac{v}{(1 - \text{IoU}) + v}
$$

## IoU Loss

회귀 손실로 사용:

```python
def iou_loss(pred_boxes, target_boxes):
    iou = batch_iou(pred_boxes, target_boxes)
    return 1 - iou.diag().mean()

def giou_loss(pred_boxes, target_boxes):
    giou = batch_giou(pred_boxes, target_boxes)
    return 1 - giou.diag().mean()
```

## IoU 임계값

| 용도 | 임계값 | 의미 |
|------|--------|------|
| COCO AP | 0.5:0.95 | 다양한 IoU에서 평균 |
| Pascal VOC | 0.5 | 50% 이상 겹침 |
| NMS | 0.5~0.7 | 중복으로 간주 |
| 학습 positive | 0.5~0.7 | 양성 샘플 기준 |
| 학습 negative | < 0.3~0.4 | 음성 샘플 기준 |

## 관련 콘텐츠

- [NMS](/ko/docs/components/detection/nms)
- [Anchor Box](/ko/docs/components/detection/anchor)
