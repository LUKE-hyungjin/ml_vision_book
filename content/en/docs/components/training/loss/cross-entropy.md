---
title: "Cross-Entropy Loss"
weight: 1
math: true
---

# Cross-Entropy Loss

{{% hint info %}}
**Prerequisites**: [Entropy](/en/docs/math/probability/entropy) | [Probability Distribution](/en/docs/math/probability/distribution) | [Softmax](/en/docs/components/activation/softmax)
{{% /hint %}}

## One-line Summary
> **Cross-Entropy is the standard classification loss that rewards assigning high probability to the correct class.**

## Why is this needed?
Classification models usually output class probabilities.
To train with backpropagation, we need a scalar that says "how wrong" the prediction is.

Cross-Entropy does exactly that:
- low probability on the true class → large penalty,
- high probability on the true class → small penalty.

Analogy: in an exam, being confidently wrong should lose more points than being uncertain.

## Formula

### Binary classification
$$
L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

### Multi-class classification
$$
L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

With one-hot labels, only the true class $c^*$ has $y_{c^*}=1$, so:
$$
L = -\log(\hat{y}_{c^*})
$$

**Symbol meanings**
- $C$ : number of classes
- $y_c$ : target label for class $c$ (usually one-hot)
- $\hat{y}_c$ : predicted probability for class $c$
- $c^*$ : index of the true class
- $L$ : loss value for one sample

## Intuition
- if $\hat{y}_{c^*}=1.0$, loss is $0$
- if $\hat{y}_{c^*}=0.1$, loss is about $2.30$
- if $\hat{y}_{c^*}=0.01$, loss is about $4.61$

So **confident wrong predictions** are penalized strongly.

## Softmax + Cross-Entropy (numerical stability)
In practice, pass raw logits directly and let the framework compute stably.

```python
import torch
import torch.nn.functional as F

B, C = 4, 3
logits = torch.tensor([
    [2.1, 0.3, -1.2],
    [0.1, 1.5, -0.2],
    [1.0, 0.2, 0.1],
    [-0.5, 0.4, 2.2],
], dtype=torch.float32)

targets = torch.tensor([0, 1, 0, 2])  # class indices (not one-hot)

# Recommended: uses log-sum-exp trick internally
loss = F.cross_entropy(logits, targets)
print(float(loss))
```

## Label Smoothing
Label smoothing softens one-hot targets by spreading a small mass to other classes.

$$
y'_c = (1 - \alpha) y_c + \frac{\alpha}{C}
$$

- $\alpha$ : smoothing strength (e.g., 0.1)
- effect: less overfitting, better calibration

```python
import torch.nn as nn
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## Quick input/output shape guide
A very common beginner failure point is mismatch between **model output shape** and **target shape**.

| Scenario | Model output (logits) | Target | Notes |
|---|---|---|---|
| Multi-class classification (basic) | `(B, C)` | `(B,)` (`long`) | use class indices |
| Sequence classification (token-level) | `(B, T, C)` | `(B, T)` (`long`) | often reshape to `(B*T, C)` before loss |
| Semantic segmentation (pixel-level) | `(B, C, H, W)` | `(B, H, W)` (`long`) | channel axis `C` exists only in logits |

> Key memory: **CrossEntropyLoss expects class-index targets by default, not one-hot targets.**

## Mini practice: wrong usage vs correct usage
This snippet shows how the same data behaves under correct and incorrect usage.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.1, -1.0]], dtype=torch.float32)  # (B=1, C=3)
target = torch.tensor([0], dtype=torch.long)                    # class index

# ✅ Correct: pass raw logits
loss_ok = F.cross_entropy(logits, target)

# ❌ Common mistake: apply softmax first
probs = torch.softmax(logits, dim=1)
loss_wrong = F.cross_entropy(probs, target)

print("correct:", float(loss_ok))
print("wrong  :", float(loss_wrong))
```

Why is this problematic?
- `F.cross_entropy` already includes `log_softmax` internally.
- Applying softmax outside can distort gradient scale and make training slower/less stable.

## Debugging checklist
- [ ] Did you avoid applying softmax before `CrossEntropyLoss`?
- [ ] Is target dtype `torch.long`?
- [ ] Is target shape `(B,)` (not `(B, C)` one-hot)?
- [ ] Are class indices within `[0, C-1]`?
- [ ] If loss is NaN, did you check logit scale / learning rate?

## Common mistakes (FAQ)
**Q1. Should I apply softmax before Cross-Entropy?**  
A. Usually no. `F.cross_entropy` expects logits and handles stable computation internally.

**Q2. How is BCE different from Cross-Entropy?**  
A. BCE is common for binary/multi-label tasks, while Cross-Entropy is common for single-label multi-class classification.

**Q3. Accuracy improves but loss decreases slowly. Is that wrong?**  
A. Not necessarily. If the model is still confidently wrong on some samples, loss can remain high.

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First check |
|---|---|---|
| `Expected target size` error | target shape mismatch | verify target is `(B,)` and remove accidental extra dims |
| `Target X is out of bounds` | class index out of range | with `num_classes=C`, target must be in `0~C-1` |
| loss becomes NaN early | too high LR or invalid input values | learning rate, NaN/Inf in inputs |
| accuracy rises but calibration is poor | over-confident predictions | consider label smoothing, temperature scaling |

## 5-minute troubleshooting experiment (beginner-friendly)
When training does not improve, change **one variable at a time** in this order.

1. **Input sanity first**: verify logits/target shapes and dtype, then run only 1-2 training steps
2. **Halve learning rate**: check whether NaN/oscillation disappears
3. **Apply label smoothing (0.1)**: check whether over-confidence reduces
4. **Check class imbalance**: if one class keeps failing, test class weights or focal loss

Small controlled changes help isolate root causes. If you change many knobs at once, diagnosis becomes unclear.

## Visual asset prompts (Nanobanana)
- **EN Diagram 1 (true-class probability vs loss curve)**  
  "Dark theme background (#1a1a2e), Cross-Entropy loss curve infographic. x-axis: true-class probability p(y*), y-axis: loss=-log(p). Highlight points p=1.0, 0.1, 0.01 with labels loss 0, 2.30, 4.61. Include labels 'confident correct', 'uncertain', 'confident wrong'. Add formula and subtle guide lines, clean vector style"

- **EN Diagram 2 (correct input pipeline)**  
  "Dark theme background (#1a1a2e), classification training pipeline diagram. Flow: logits -> internal CrossEntropyLoss (log_softmax + NLL) -> scalar loss -> backward. Also show wrong branch 'apply softmax before CE' with red warning icon. Include arrows, beginner-friendly labels, and compact correct-vs-wrong comparison panel"

## Related Content
- [Focal Loss](/en/docs/components/training/loss/focal-loss)
- [Entropy](/en/docs/math/probability/entropy)
- [Probability Distribution](/en/docs/math/probability/distribution)
- [Softmax](/en/docs/components/activation/softmax)
