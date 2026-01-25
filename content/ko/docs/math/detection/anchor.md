---
title: "Anchor Box"
weight: 3
math: true
---

# Anchor Box

## 개요

Anchor Box는 미리 정의된 다양한 크기/비율의 박스로, 객체 검출의 기준점 역할을 합니다.

## 왜 Anchor인가?

직접 좌표 예측의 문제:
- 수렴이 어려움
- 다양한 크기의 객체 처리 어려움

Anchor 접근:
- 사전 정의된 박스와의 **차이**를 예측
- 더 안정적인 학습

## Anchor 구성

```
        Scales: [32, 64, 128]
        Ratios: [0.5, 1.0, 2.0]
              ↓
    총 9개 anchor per location
```

```python
def generate_anchors(feature_size, stride, scales, ratios):
    """
    feature_size: (H, W) 피처맵 크기
    stride: 원본 이미지 대비 축소 비율
    """
    anchors = []

    for y in range(feature_size[0]):
        for x in range(feature_size[1]):
            # 피처맵 위치 → 원본 좌표
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride

            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)

                    # [x1, y1, x2, y2]
                    anchors.append([
                        cx - w/2, cy - h/2,
                        cx + w/2, cy + h/2
                    ])

    return np.array(anchors)
```

## Box Regression

Anchor로부터 실제 박스까지의 변환:

$$
\begin{aligned}
t_x &= (x - x_a) / w_a \\
t_y &= (y - y_a) / h_a \\
t_w &= \log(w / w_a) \\
t_h &= \log(h / h_a)
\end{aligned}
$$

- $(x, y, w, h)$: GT 박스
- $(x_a, y_a, w_a, h_a)$: Anchor
- $(t_x, t_y, t_w, t_h)$: 예측 대상 (delta)

### 역변환 (디코딩)

```python
def decode_boxes(anchors, deltas):
    """
    anchors: (N, 4) [x1, y1, x2, y2]
    deltas: (N, 4) [tx, ty, tw, th]
    """
    # Anchor 중심과 크기
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + wa / 2
    ya = anchors[:, 1] + ha / 2

    # Delta 적용
    tx, ty, tw, th = deltas.T

    x = tx * wa + xa
    y = ty * ha + ya
    w = np.exp(tw) * wa
    h = np.exp(th) * ha

    # [x1, y1, x2, y2]로 변환
    boxes = np.stack([
        x - w/2, y - h/2,
        x + w/2, y + h/2
    ], axis=1)

    return boxes
```

## FPN 멀티스케일 Anchor

```python
# Feature Pyramid Network 설정
fpn_strides = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7
anchor_scales = [[32], [64], [128], [256], [512]]  # 레벨당 하나
anchor_ratios = [0.5, 1.0, 2.0]

all_anchors = []
for stride, scales in zip(fpn_strides, anchor_scales):
    level_anchors = generate_anchors(
        feature_size=(H // stride, W // stride),
        stride=stride,
        scales=scales,
        ratios=anchor_ratios
    )
    all_anchors.append(level_anchors)
```

## Anchor 할당 (Matching)

GT 박스와 anchor 매칭:

```python
def assign_anchors(anchors, gt_boxes, pos_iou=0.5, neg_iou=0.4):
    """
    Returns:
        labels: 1 (positive), 0 (negative), -1 (ignore)
        matched_gt_idx: 매칭된 GT 인덱스
    """
    ious = batch_iou(anchors, gt_boxes)  # (N, M)

    # 각 anchor의 최대 IoU GT
    max_iou, matched_idx = ious.max(dim=1)

    labels = torch.full((len(anchors),), -1)  # 기본: ignore

    # Negative: IoU < 0.4
    labels[max_iou < neg_iou] = 0

    # Positive: IoU >= 0.5
    labels[max_iou >= pos_iou] = 1

    # 각 GT의 최고 IoU anchor도 positive
    gt_max_idx = ious.argmax(dim=0)
    labels[gt_max_idx] = 1
    matched_idx[gt_max_idx] = torch.arange(len(gt_boxes))

    return labels, matched_idx
```

## K-Means Anchor

데이터셋에 최적화된 anchor 학습:

```python
from sklearn.cluster import KMeans

def kmeans_anchors(boxes, k=9):
    """
    boxes: 모든 GT 박스의 (width, height)
    """
    # 1 - IoU를 거리로 사용
    wh = boxes[:, 2:4] - boxes[:, :2]  # (N, 2)

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(wh)

    anchors = kmeans.cluster_centers_
    return anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # 크기순 정렬
```

## Anchor-Free 대안

최근 anchor 없는 방식도 인기:

| 방식 | 모델 |
|------|------|
| Center-based | FCOS, CenterNet |
| Keypoint-based | CornerNet |
| Query-based | DETR |

## 관련 콘텐츠

- [IoU](/ko/docs/math/detection/iou)
- [NMS](/ko/docs/math/detection/nms)
- [Focal Loss](/ko/docs/math/training/loss/focal-loss)
