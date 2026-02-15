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

## Understand it once with a tiny example
Assume three candidate boxes point to the same object. (IoU threshold $t=0.5$)

| Box | Score | IoU with A | Result |
|---|---:|---:|---|
| A | 0.92 | - | Keep (highest score) |
| B | 0.88 | 0.76 | Remove (too much overlap with A) |
| C | 0.61 | 0.31 | Keep (low overlap) |

After NMS, only A and C remain.  
The key idea is: **"high confidence is good, but duplicates of the same object should collapse to one box."**

## Formula → code mapping (beginner bridge)
When implementing NMS for the first time, map each formula step to one code line.

1. **Pick highest-score box**
   - Formula view: choose current reference box $b_i$
   - Code view: `i = order[0]`

2. **Compute IoU against remaining boxes**
   - Formula view: evaluate $\mathrm{IoU}(b_i, b_j)$
   - Code view: `ious = compute_iou(boxes[i], boxes[rest])`

3. **Suppress by threshold**
   - Formula view: suppress if $\mathrm{IoU}(b_i, b_j) > t$
   - Code view: `order = rest[ious <= iou_threshold]`

4. **Stop condition**
   - Formula view: stop when no candidates remain
   - Code view: `if len(order) == 1: break`

Once this mapping is clear, extending to Soft-NMS or class-wise NMS is much easier.

## Minimal class-wise NMS sketch
A very common production bug is running global NMS and accidentally suppressing different classes.
Use per-class NMS like this:

```python
def class_wise_nms(boxes, scores, labels, iou_threshold=0.5):
    keep_all = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep_c = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.append(idx[keep_c])  # map back to original indices
    return np.concatenate(keep_all)
```

## Debugging checklist (practical)
- [ ] **Class-wise NMS check**: are different classes accidentally suppressing each other?
- [ ] **Coordinate scale check**: are pixel coordinates and normalized (0~1) coordinates mixed?
- [ ] **Threshold-order check**: did you set score threshold first, then tune NMS IoU?
- [ ] **Top-K check**: are too many candidates making NMS cost explode?
- [ ] **Crowded-scene failure check**: are true positives disappearing in dense scenes?

## Common mistakes (FAQ)
**Q1. Is a lower NMS threshold always better?**  
A. No. It removes duplicates more aggressively, but can also remove true positives and hurt recall.

**Q2. Which should I tune first: score threshold or NMS threshold?**  
A. Usually score threshold first, then NMS threshold. Reversing this often makes diagnosis harder.

**Q3. If two boxes have high IoU but different classes, should one be removed?**  
A. Usually no (class-wise NMS). For example, person and backpack may overlap heavily and both be correct.

## Quick guide: choosing Hard NMS vs Soft-NMS
A frequent beginner pain point in production is: "When should I keep Hard NMS, and when should I switch to Soft-NMS?"

| Scenario | Recommended choice | Why |
|---|---|---|
| Real-time / edge deployment, latency is critical | Hard NMS | simpler implementation and faster runtime |
| Crowded scenes with many overlapping objects | Soft-NMS | reduces over-suppression of true positives |
| Baseline reproduction / early debugging | Start with Hard NMS | easier to isolate root causes |

Practical tip: build a stable baseline with Hard NMS first, then try Soft-NMS only if recall loss is noticeable on dense datasets.

## Visual asset prompts (Nanobanana)
- **EN Diagram 1 (NMS pipeline)**  
  "Dark theme background (#1a1a2e), object detection NMS process infographic. Input: 5 overlapping boxes around one object with confidence scores 0.95, 0.88, 0.74, 0.60, 0.41. Step 1: select highest-score box. Step 2: mark boxes with IoU>0.5 as suppressed in red. Step 3: repeat on remaining boxes. Final output keeps 2 boxes. Include arrows and step numbers (1,2,3), labels 'select/suppress/keep', clean vector style, high readability"

- **EN Diagram 2 (Threshold trade-off)**  
  "Dark theme background (#1a1a2e), 3-panel comparison of NMS IoU thresholds (0.3 / 0.5 / 0.7) on the same set of predicted boxes. Annotate remaining box count and TP/FP trend in each panel. Labels: 'over-suppression', 'balanced', 'duplicate residue'. Include compact legend table, minimal clean vector infographic"

## Related Content
- [IoU](/en/docs/components/detection/iou)
- [Anchor Box](/en/docs/components/detection/anchor)
- [YOLO](/en/docs/architecture/detection/yolo)
