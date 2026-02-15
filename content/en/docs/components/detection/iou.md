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

## Common thresholds in practice
| Usage | IoU threshold | Meaning |
|---|---:|---|
| Pascal VOC AP | 0.5 | TP if overlap is at least 50% |
| COCO AP | 0.5:0.95 | averaged over stricter thresholds |
| NMS | 0.5~0.7 | remove highly overlapped duplicates |
| Anchor matching (example) | pos: ≥0.5, neg: <0.4 | assign training labels |

## Related Content
- [Anchor Box](/en/docs/components/detection/anchor)
- [NMS](/en/docs/components/detection/nms)
- [YOLO](/en/docs/architecture/detection/yolo)
