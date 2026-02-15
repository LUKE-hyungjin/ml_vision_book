---
title: "NMS"
weight: 2
math: true
---

# NMS (Non-Maximum Suppression)

{{% hint info %}}
**Prerequisites**: [IoU](/en/docs/components/detection/iou)
{{% /hint %}}

## One-line Summary
> **NMS is a post-processing algorithm that keeps only the highest-confidence box among highly overlapping duplicate detections.**

## Why is this needed?
Object detectors often predict multiple similar boxes around the same object.
If we keep all of them, one object appears to be detected many times.
NMS removes these duplicates and keeps cleaner final predictions.

NMS is used in almost all detection families:
- 1-stage detectors: [YOLO](/en/docs/architecture/detection/yolo), SSD
- 2-stage detectors: Faster R-CNN variants

An analogy: when several people point at the same target, we keep the most confident pointer and ignore redundant ones.

## Formula / Symbols
NMS is mainly procedural, but its suppression rule is expressed with an IoU threshold:

$$
\text{suppress box } j \quad \text{if} \quad \mathrm{IoU}(b_i, b_j) > t
$$

**Symbol meanings:**
- $b_i$ : currently selected box (highest score)
- $b_j$ : candidate box to compare
- $\mathrm{IoU}(b_i, b_j)$ : overlap ratio between two boxes
- $t$ : NMS threshold (typically 0.45~0.7)

## Intuition
1. Pick the highest-scoring box.
2. Remove boxes that overlap too much with it (IoU > $t$).
3. Repeat on the remaining boxes.

The core idea is to keep boxes with **high confidence and low redundancy**.

## Implementation
```python
import numpy as np


def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4) each box is [x1, y1, x2, y2]
    scores: (N,) confidence score
    """
    order = scores.argsort()[::-1]  # high -> low
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        ious = compute_iou(boxes[i], boxes[rest])

        # Keep only boxes below the IoU threshold
        order = rest[ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)
```

### Common variants
- **Batched NMS**: do not suppress across different classes
- **Soft-NMS**: decay scores instead of hard removal
- **DIoU-NMS**: include center-distance information


## Easy-to-miss practical points
### 1) Check whether NMS is class-wise
Most detectors use **class-wise NMS**.
That means boxes from different classes should not suppress each other even if they overlap heavily.

For example, a person box and a backpack box can overlap a lot.
If you run global NMS without class separation, you may remove correct detections.

### 2) Top-K pre-filtering
If too many candidate boxes are passed to NMS, cost grows quickly (roughly $O(N^2)$ comparisons).
In practice, keep only top-$K$ boxes by score before NMS to save latency and memory.

- Example: keep top 1000 boxes per class, then run NMS
- Effect: lower latency, especially helpful on edge/mobile deployments

### 3) Threshold tuning order
A stable tuning order is usually:

1. Set score threshold first to remove very weak boxes
2. Then adjust NMS IoU threshold to control duplicate suppression strength
3. Finally fine-tune while checking AP/recall trade-offs

## Practical tips
- For crowded scenes, lower threshold (e.g., 0.45) can reduce duplicates
- For many small objects, avoid too-low threshold (may remove true positives)
- Tune together with score threshold for stable behavior

## Related Content
- [IoU](/en/docs/components/detection/iou)
- [Anchor Box](/en/docs/components/detection/anchor)
- [YOLO](/en/docs/architecture/detection/yolo)
