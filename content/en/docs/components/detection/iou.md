---
title: "IoU"
weight: 1
math: true
---

# IoU (Intersection over Union)

{{% hint info %}}
**Prerequisites**: None
{{% /hint %}}

## One-line Summary
> **IoU measures how much a predicted box overlaps with a ground-truth box on a 0 to 1 scale.**

## Why is this needed?
In object detection, we need a consistent rule to decide whether a predicted box is "close enough" to the target.
IoU provides that rule, so it is used for:

- evaluation (AP, mAP),
- positive/negative assignment during training,
- duplicate removal in [NMS](/en/docs/components/detection/nms).

A simple analogy: draw two rectangles on paper and compute the **ratio of overlapped area**.

## Formula / Symbols
$$
\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

**Symbol meanings:**
- $A$ : predicted bounding-box region
- $B$ : ground-truth bounding-box region
- $|A \cap B|$ : intersection (overlapped) area
- $|A \cup B|$ : union (combined) area

Its range is $0 \le \mathrm{IoU} \le 1$.

- 0: no overlap
- 1: perfect match

## Intuition
- IoU becomes large when **intersection is large** and **union is small**.
- Even a small shift can reduce intersection quickly.
- That is why different IoU thresholds (e.g., 0.5 vs 0.75) can change TP/FP decisions.

## Numeric example
Assume each box has area 100, and their overlap area is 60.

- Intersection: $|A \cap B| = 60$
- Union: $|A \cup B| = 100 + 100 - 60 = 140$
- So $\mathrm{IoU} = 60/140 \approx 0.43$

With an IoU threshold of 0.5, this is not a TP.
With a threshold of 0.4, it may be counted as a TP.
The key point is: **the same prediction can be judged differently depending on the threshold**.

## How threshold changes the decision
Even with the same IoU value, the final decision depends on the criterion.

| IoU value | Threshold 0.5 | Threshold 0.75 |
|---:|---|---|
| 0.43 | FP (miss) | FP (miss) |
| 0.62 | TP | FP (fails under stricter rule) |
| 0.81 | TP | TP |

As the criterion becomes stricter (e.g., 0.75), **precise localization** becomes more important.

## Common coordinate-format pitfall
The IoU formula is the same, but results break if your **box format** is mixed.

- `xyxy`: `[x1, y1, x2, y2]` (top-left / bottom-right)
- `cxcywh`: `[cx, cy, w, h]` (center / width / height)

If a model outputs `cxcywh` and you compute IoU as if it were `xyxy`, the intersection is wrong.
In practice, always **convert to one common format first**.

Simple conversion:
$$
\begin{aligned}
x_1 &= c_x - \frac{w}{2}, \quad y_1 = c_y - \frac{h}{2},\\
x_2 &= c_x + \frac{w}{2}, \quad y_2 = c_y + \frac{h}{2}
\end{aligned}
$$

## Edge cases (important in practice)
Before optimization, prioritize **numerically safe behavior**.

- If a box is flipped (`x2 < x1` or `y2 < y1`), clamp area with `max(0, ...)`.
- If both boxes are invalid and union becomes 0, return IoU = 0 to avoid divide-by-zero.
- Tiny negative values can appear from floating-point error; add an `eps` (e.g., `1e-9`) when needed.

These checks prevent many NaN issues during training.

## Limitations of IoU and extended metrics
IoU is intuitive, but when two boxes **do not overlap**, it is always 0,
so it cannot express "how far apart" they are.

That is why box-regression losses often use extended metrics:

- **GIoU**: adds information from the smallest enclosing box
- **DIoU/CIoU**: also considers center distance (and aspect ratio)

A common GIoU form is:
$$
\mathrm{GIoU} = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
$$

**Additional symbols:**
- $C$ : the smallest axis-aligned box that encloses both $A$ and $B$
- $|C \setminus (A \cup B)|$ : uncovered area inside the enclosing box

So even with zero overlap, GIoU can be negative, giving a more useful learning signal.

## Implementation
```python
def box_iou(box1, box2):
    """
    box: [x1, y1, x2, y2] (top-left, bottom-right)
    """
    # 1) Intersection coordinates
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    # 2) Intersection area
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    # 3) Area of each box
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    # 4) Union area
    union = area1 + area2 - inter

    # 5) IoU
    return inter / union if union > 0 else 0.0
```

## Mini practice: batched IoU + quick sanity checks
The snippet below computes IoU for multiple box pairs at once,
and includes the two guards beginners often miss: "flipped coordinates" and "safe division".

```python
import torch


def pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9):
    """
    boxes1, boxes2: [N, 4] in xyxy format
    returns: [N] IoU for pairwise rows
    """
    # 1) Fix coordinate ordering (x1<=x2, y1<=y2)
    x11 = torch.minimum(boxes1[:, 0], boxes1[:, 2])
    y11 = torch.minimum(boxes1[:, 1], boxes1[:, 3])
    x12 = torch.maximum(boxes1[:, 0], boxes1[:, 2])
    y12 = torch.maximum(boxes1[:, 1], boxes1[:, 3])

    x21 = torch.minimum(boxes2[:, 0], boxes2[:, 2])
    y21 = torch.minimum(boxes2[:, 1], boxes2[:, 3])
    x22 = torch.maximum(boxes2[:, 0], boxes2[:, 2])
    y22 = torch.maximum(boxes2[:, 1], boxes2[:, 3])

    # 2) Intersection
    ix1 = torch.maximum(x11, x21)
    iy1 = torch.maximum(y11, y21)
    ix2 = torch.minimum(x12, x22)
    iy2 = torch.minimum(y12, y22)

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # 3) Union
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1 + area2 - inter

    # 4) Safe divide
    return inter / (union + eps)


pred = torch.tensor([
    [10., 10., 30., 30.],  # fairly good match
    [10., 10., 20., 20.],  # small box far away from GT
    [30., 30., 10., 10.],  # flipped coordinates (intentional)
])

gt = torch.tensor([
    [12., 12., 28., 28.],
    [40., 40., 60., 60.],
    [12., 12., 28., 28.],
])

iou = pairwise_iou_xyxy(pred, gt)
print(iou)  # tensor([...])

# Quick check: IoU should stay in [0, 1] (with tiny numeric tolerance)
assert torch.all(iou >= -1e-6) and torch.all(iou <= 1.0 + 1e-6)
```

Beginner checkpoints:
- Pair 1 should have relatively high IoU.
- Pair 2 should be near 0 (far apart).
- Pair 3 had flipped coordinates, but still computes safely without NaN.

## Common thresholds in practice
| Usage | IoU threshold | Meaning |
|---|---:|---|
| Pascal VOC AP | 0.5 | TP if overlap is at least 50% |
| COCO AP | 0.5:0.95 | averaged over stricter thresholds |
| NMS | 0.5~0.7 | remove highly overlapped duplicates |
| Anchor matching (example) | pos: ≥0.5, neg: <0.4 | assign training labels |

## Debugging checklist (practical)
If performance suddenly drops or AP looks abnormally low, check in this order:

1. **Unify coordinate format**: ensure both prediction and GT are `xyxy`
2. **Validate coordinate ordering**: enforce `x1 <= x2`, `y1 <= y2`
3. **Match coordinate scale**: avoid mixing pixel coordinates with normalized (0~1) values
4. **Recheck NMS threshold**: too low causes over-suppression, too high keeps duplicates
5. **Verify eval protocol**: do not mix VOC-style IoU@0.5 with COCO-style AP@[0.5:0.95]

## Common mistakes (FAQ)
- **Q. Can IoU be negative?**  
  A. Standard IoU cannot be negative. A negative value usually indicates a bug in area/intersection code.

- **Q. If prediction fully contains GT, is IoU always 1?**  
  A. No. IoU is 1 only when both boxes are exactly identical.

- **Q. My classification confidence is good but AP is low. Why?**  
  A. A common reason is localization error: class is right, but box alignment misses the IoU threshold.

## 30-second memory trick: IoU in 4 steps
When implementing IoU for the first time, this 4-step checklist catches most bugs.

1. **Find overlap box coordinates**: `max(left)`, `max(top)`, `min(right)`, `min(bottom)`
2. **Compute overlap area safely**: `max(0, w) * max(0, h)`
3. **Compute union**: `area1 + area2 - inter`
4. **Guard division**: return 0 when `union == 0`

In real projects, most IoU bugs come from step 2 (missing clamp) and step 4 (divide-by-zero).

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First thing to inspect |
|---|---|---|
| AP50 is okay but AP75 drops hard | poor localization precision | NMS threshold, box-regression LR |
| IoU stays near 0 early in training | mixed coordinate formats | where `xyxy`/`cxcywh` conversion happens |
| intermittent NaN | union=0 or negative width/height | `max(0, ...)`, `eps`, input validation |
| class is right but too many FP boxes | NMS too loose | increase NMS IoU threshold |

## Related Content
- [Anchor Box](/en/docs/components/detection/anchor)
- [NMS](/en/docs/components/detection/nms)
- [YOLO](/en/docs/architecture/detection/yolo)
