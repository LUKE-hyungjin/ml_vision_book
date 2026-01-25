---
title: "NMS"
weight: 2
math: true
---

# NMS (Non-Maximum Suppression)

## 개요

NMS는 중복된 검출 박스를 제거하여 객체당 하나의 최종 박스만 남기는 후처리 기법입니다.

## 기본 알고리즘

```
1. 점수로 박스 정렬 (내림차순)
2. 최고 점수 박스 선택
3. 선택된 박스와 IoU > threshold인 박스들 제거
4. 박스가 없을 때까지 2-3 반복
```

## 구현

```python
import numpy as np

def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4) [x1, y1, x2, y2]
    scores: (N,) 신뢰도 점수
    """
    # 점수로 정렬
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        # 최고 점수 박스
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # 나머지 박스들과 IoU 계산
        remaining = order[1:]
        ious = compute_iou(boxes[i], boxes[remaining])

        # IoU < threshold인 것만 남김
        mask = ious < iou_threshold
        order = remaining[mask]

    return np.array(keep)
```

## Batched NMS (클래스별)

클래스가 다르면 억제하지 않음:

```python
def batched_nms(boxes, scores, classes, iou_threshold):
    """각 클래스별로 독립적으로 NMS 수행"""
    keep = []

    for c in np.unique(classes):
        mask = classes == c
        class_keep = nms(boxes[mask], scores[mask], iou_threshold)
        keep.extend(np.where(mask)[0][class_keep])

    return np.array(keep)
```

## Soft-NMS

박스를 완전히 제거하지 않고 점수를 낮춤:

$$
s_i = \begin{cases}
s_i & \text{if IoU} < N_t \\
s_i \cdot e^{-\frac{\text{IoU}^2}{\sigma}} & \text{otherwise}
\end{cases}
$$

```python
def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS with Gaussian penalty
    """
    N = len(boxes)
    indices = np.arange(N)

    for i in range(N):
        # 최고 점수 찾기
        max_idx = i + scores[i:].argmax()

        # 스왑
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        indices[[i, max_idx]] = indices[[max_idx, i]]

        # IoU 계산
        ious = compute_iou(boxes[i], boxes[i+1:])

        # Gaussian 감쇠
        decay = np.exp(-(ious ** 2) / sigma)
        scores[i+1:] *= decay

    # 임계값 이상인 것만 반환
    keep = indices[scores > score_threshold]
    return keep
```

## DIoU-NMS

중심점 거리 고려:

```python
def diou_nms(boxes, scores, iou_threshold=0.5, diou_threshold=0.7):
    """중심점이 멀면 억제하지 않음"""
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        remaining = order[1:]
        ious = compute_iou(boxes[i], boxes[remaining])
        dious = compute_diou(boxes[i], boxes[remaining])

        # IoU 높아도 DIoU 낮으면 유지
        mask = (ious < iou_threshold) | (dious < diou_threshold)
        order = remaining[mask]

    return np.array(keep)
```

## Matrix NMS (병렬화)

행렬 연산으로 빠른 처리:

```python
def matrix_nms(boxes, scores, kernel='gaussian', sigma=2.0):
    """SOLO에서 사용된 Matrix NMS"""
    n = len(boxes)

    # 모든 쌍의 IoU 계산
    ious = batch_iou(boxes, boxes)  # (N, N)

    # 자기 자신 제외
    ious.fill_diagonal_(0)

    # 정렬
    sorted_idx = scores.argsort(descending=True)
    sorted_ious = ious[sorted_idx][:, sorted_idx]

    # 위쪽 삼각 행렬만 사용 (높은 점수가 낮은 점수 억제)
    sorted_ious = torch.triu(sorted_ious, diagonal=1)

    # 각 박스가 받는 최대 억제량
    max_iou = sorted_ious.max(dim=0)[0]

    # 감쇠 함수
    if kernel == 'gaussian':
        decay = torch.exp(-(max_iou ** 2) / sigma)
    else:  # linear
        decay = 1 - max_iou

    # 점수 업데이트
    new_scores = scores[sorted_idx] * decay

    return sorted_idx, new_scores
```

## PyTorch/TorchVision

```python
import torchvision

# 표준 NMS
keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)

# Batched NMS
keep = torchvision.ops.batched_nms(boxes, scores, classes, iou_threshold=0.5)
```

## 하이퍼파라미터

| 파라미터 | 일반적인 값 | 용도 |
|----------|-------------|------|
| IoU threshold | 0.45~0.65 | 밀집 객체: 낮게, 드문 객체: 높게 |
| Score threshold | 0.05~0.3 | 전처리 필터링 |
| Max detections | 100~300 | 메모리/속도 제한 |

## 관련 콘텐츠

- [IoU](/ko/docs/math/detection/iou)
- [Anchor Box](/ko/docs/math/detection/anchor)
