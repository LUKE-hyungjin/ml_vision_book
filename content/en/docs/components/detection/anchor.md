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

## 3-minute mini exercise: how the formula moves real numbers
Use one anchor and one GT box to see why regression targets are intuitive in practice.

- Anchor (center/size): $(x_a, y_a, w_a, h_a) = (50, 50, 40, 20)$
- GT (center/size): $(x, y, w, h) = (58, 46, 50, 24)$

Plug into the formulas:
$$
\begin{aligned}
t_x &= \frac{58-50}{40}=0.20,\\
t_y &= \frac{46-50}{20}=-0.20,\\
t_w &= \log\left(\frac{50}{40}\right)\approx 0.223,\\
t_h &= \log\left(\frac{24}{20}\right)\approx 0.182
\end{aligned}
$$

Interpretation:
- $t_x=0.20$ → move **right** by 20% of anchor width
- $t_y=-0.20$ → move **up** by 20% of anchor height
- $t_w, t_h > 0$ → **expand width/height** slightly

So the model does not memorize absolute coordinates; it learns how to **correct a reference anchor**.

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

## Mini implementation: IoU-based anchor labeling
The previous `encode_boxes` function only builds regression targets.
In a real training pipeline, you first label each anchor as positive/negative/ignore.

```python
import numpy as np


def assign_anchor_labels(iou_max, pos_thr=0.7, neg_thr=0.3):
    """
    iou_max: (N,) maximum IoU per anchor
    return: labels (N,) where 1=positive, 0=negative, -1=ignore
    """
    labels = np.full_like(iou_max, fill_value=-1, dtype=np.int64)  # default ignore
    labels[iou_max >= pos_thr] = 1
    labels[iou_max < neg_thr] = 0
    return labels


# Example: max IoU of 8 anchors
iou_max = np.array([0.82, 0.74, 0.55, 0.40, 0.29, 0.10, 0.68, 0.02])
labels = assign_anchor_labels(iou_max, pos_thr=0.7, neg_thr=0.3)

num_pos = (labels == 1).sum()
num_neg = (labels == 0).sum()
num_ign = (labels == -1).sum()

print(f"pos={num_pos}, neg={num_neg}, ignore={num_ign}")
# pos=2, neg=3, ignore=3
```

Key interpretation:
- If positives are too few (sometimes near zero in a batch), box regression learns very slowly.
- If negatives dominate, classification can collapse into mostly background predictions.
- So you tune thresholds (`pos_thr`, `neg_thr`) together with sampling ratio (e.g., pos:neg=1:3).

## Practical debugging checklist
- [ ] **Unify box format**: are `xyxy` and `cxcywh` mixed anywhere in the pipeline?
- [ ] **Scale matching**: if your dataset has many small objects, are large anchors overrepresented?
- [ ] **Positive-anchor ratio**: are positives too sparse per batch? (e.g., <1% can stall learning)
- [ ] **Regression explosion watch**: do `tw`, `th` show very large absolute values early in training?
- [ ] **Re-check matching thresholds**: are $\tau_{pos}, \tau_{neg}$ appropriate for dataset difficulty?

## Common mistakes (FAQ)
**Q1. If I add more anchors, will performance always improve?**  
A. Not always. Too many anchors usually increase negatives much more than positives, so training can become background-dominant. Start from dataset-aligned scales/ratios first.

**Q2. Is a higher IoU threshold always better?**  
A. Not necessarily. Very high thresholds reduce positive samples and can weaken training signals. For datasets with many small objects, a moderate threshold is often more stable as a starting point.

**Q3. My box regression loss becomes NaN. What should I check first?**  
A. Check (1) whether width/height ever become negative, (2) whether `log(w/w_a)` receives non-positive values, and (3) whether clipping/epsilon guards are applied.

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First thing to inspect |
|---|---|---|
| recall is low and FP is high while loss still decreases | anchor scales/ratios do not match dataset distribution | per-class box aspect-ratio histogram, positives per FPN level |
| predicted boxes keep drifting to one side | normalization/decoding scale mismatch | target normalization (mean/std), decode-formula implementation |
| box loss is abnormally large from early training | matching thresholds are off or flipped boxes entered pipeline | $\tau_{pos}, \tau_{neg}$, checks for `x1<x2, y1<y2` |
| model keeps missing only certain object sizes | missing anchors for those size bands | object-size distribution vs anchor-scale coverage |

## Quick experiment: how thresholds change the number of positives
The snippet below shows how positive/negative/ignore counts change
when you sweep `pos_thr` and `neg_thr` on the same IoU distribution.

```python
import numpy as np


def count_labels(iou_max, pos_thr, neg_thr):
    labels = np.full_like(iou_max, -1, dtype=np.int64)
    labels[iou_max >= pos_thr] = 1
    labels[iou_max < neg_thr] = 0
    return {
        "pos": int((labels == 1).sum()),
        "neg": int((labels == 0).sum()),
        "ignore": int((labels == -1).sum()),
    }


iou_max = np.array([0.82, 0.74, 0.55, 0.40, 0.29, 0.10, 0.68, 0.02])

for pos_thr, neg_thr in [(0.7, 0.3), (0.6, 0.3), (0.5, 0.4)]:
    print((pos_thr, neg_thr), count_labels(iou_max, pos_thr, neg_thr))
```

Interpretation guide:
- Lower `pos_thr` → more positives (stronger regression signal, but potentially noisier labels).
- Higher `neg_thr` → fewer negatives and more ignores (classification may become easier).
- For beginners, first check whether positives are too sparse (near zero in many batches).

## 1-minute self-check (beginner)
- [ ] I plotted object-size distribution (small/medium/large) at least once.
- [ ] I printed positive-anchor count per batch in training logs.
- [ ] I adjusted `pos_thr`/`neg_thr` gradually (0.05~0.1), not all at once.
- [ ] If NaN appears, I first verify `w,h>0` and safe `log` inputs (`clip/eps`).

## Related Content
- [IoU](/en/docs/components/detection/iou)
- [NMS](/en/docs/components/detection/nms)
- [YOLO](/en/docs/architecture/detection/yolo)
