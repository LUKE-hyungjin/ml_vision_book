---
title: "Anchor Box"
weight: 3
math: true
---

# Anchor Box

{{% hint info %}}
**Prerequisites**: [IoU](/en/docs/components/detection/iou)
{{% /hint %}}

## One-line Summary
> **Anchor boxes predefine reference boxes, and the model only predicts how to shift/scale them to match objects.**

## Why is this needed?
If a detector predicts box coordinates from scratch, optimization can be unstable early in training.

Anchor-based detection starts with predefined boxes (multiple scales and aspect ratios),
and learns only the **delta** from anchor to ground truth.

- Benefit 1: more stable training
- Benefit 2: easier handling of both small and large objects
- Used in: Faster R-CNN, RetinaNet, SSD-style detectors

## Formula / Symbols
Given anchor $(x_a, y_a, w_a, h_a)$ and ground truth $(x, y, w, h)$,
regression targets are:

$$
\begin{aligned}
t_x &= \frac{x - x_a}{w_a}, \\
t_y &= \frac{y - y_a}{h_a}, \\
t_w &= \log\left(\frac{w}{w_a}\right), \\
t_h &= \log\left(\frac{h}{h_a}\right)
\end{aligned}
$$

**Symbol meanings:**
- $(x, y)$ : ground-truth box center
- $(w, h)$ : ground-truth box width/height
- $(x_a, y_a)$ : anchor center
- $(w_a, h_a)$ : anchor width/height
- $(t_x, t_y, t_w, t_h)$ : offsets predicted by the model

Decoding back to a predicted box:

$$
\begin{aligned}
\hat{x} &= t_x w_a + x_a, \\
\hat{y} &= t_y h_a + y_a, \\
\hat{w} &= e^{t_w} w_a, \\
\hat{h} &= e^{t_h} h_a
\end{aligned}
$$

## Intuition
Think of anchors as pre-placed stickers of different shapes on an image.

- If a sticker is already similar to the target object, only a small correction is needed.
- If it is very different, a larger correction is needed.

So the task becomes "refine a reference" instead of "invent a box from nothing."

### Anchor assignment intuition
During training, each anchor is matched to ground-truth boxes to create **positive/negative** labels.

- Usually positive if $\mathrm{IoU} \ge \tau_{pos}$
- Usually negative if $\mathrm{IoU} < \tau_{neg}$
- Middle range is often ignored to reduce noisy supervision

For example, with $(\tau_{pos}, \tau_{neg}) = (0.7, 0.3)$:

| Anchor IoU | Label | Meaning |
|---|---|---|
| 0.82 | Positive | use this anchor for box regression + classification |
| 0.55 | Ignore | ambiguous, excluded from loss |
| 0.12 | Negative | train as background |

## Practical notes before implementation
1. **Designing anchor scales/aspect ratios**
   - Different FPN levels usually use different scales (e.g., P3 for small objects, P7 for large objects).
   - Aspect ratios should reflect your dataset statistics (how many wide vs tall objects).

2. **Regression target normalization**
   - Many implementations normalize $(t_x, t_y, t_w, t_h)$ with mean/std to stabilize optimization.
   - Example: compute loss on $t'_x = (t_x - \mu_x)/\sigma_x$ style normalized targets.

3. **Handling edge cases**
   - Tiny anchors or invalid boxes can cause numerical issues in $\log(w/w_a)$.
   - That is why code uses lower bounds like `np.clip(..., 1e-6, None)`.

4. **Reducing positive/negative imbalance**
   - In practice, negative anchors vastly outnumber positives, so classification loss can be dominated by background.
   - Common fixes are keeping a positive:negative ratio (e.g., 1:3) per mini-batch, or using hard negative mining/focal loss.

## Implementation
```python
import numpy as np


def encode_boxes(anchors, gt_boxes):
    """
    anchors, gt_boxes: (N, 4) where each box is [x1, y1, x2, y2]
    return: (N, 4) [tx, ty, tw, th]
    """
    # Anchor center/size
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha

    # GT center/size
    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    x = gt_boxes[:, 0] + 0.5 * w
    y = gt_boxes[:, 1] + 0.5 * h

    # Delta targets
    tx = (x - xa) / np.clip(wa, 1e-6, None)
    ty = (y - ya) / np.clip(ha, 1e-6, None)
    tw = np.log(np.clip(w, 1e-6, None) / np.clip(wa, 1e-6, None))
    th = np.log(np.clip(h, 1e-6, None) / np.clip(ha, 1e-6, None))

    return np.stack([tx, ty, tw, th], axis=1)
```

## Related Content
- [IoU](/en/docs/components/detection/iou)
- [NMS](/en/docs/components/detection/nms)
- [YOLO](/en/docs/architecture/detection/yolo)
