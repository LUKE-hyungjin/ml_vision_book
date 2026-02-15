---
title: "Adam"
weight: 2
math: true
---

# Adam & AdamW

{{% hint info %}}
**Prerequisites**: [SGD](/en/docs/components/training/optimizer/sgd), [Weight Decay](/en/docs/components/training/regularization/weight-decay)
{{% /hint %}}

## One-line Summary
> **Adam automatically adjusts step size per parameter, and in practice AdamW (decoupled weight decay) is the default choice in most modern vision training setups.**

## Why is this needed?
A common training problem is: one global learning rate does not fit all parameters.

- Some parameters get noisy, large gradients.
- Others get tiny gradients and barely move.

Adam handles this by adapting updates per parameter.
Analogy: when walking on uneven terrain, you naturally take **smaller steps on slippery parts and larger steps on stable ground**.

## Core formulas

Let gradient at step $t$ be $g_t$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Bias correction (important at early steps):

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Update rule:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

### Symbol meanings
- $\theta_t$ : current parameter
- $g_t$ : gradient at current step
- $m_t$ : EMA of gradient (first moment)
- $v_t$ : EMA of squared gradient (second moment)
- $\beta_1$ : decay factor for first moment (usually 0.9)
- $\beta_2$ : decay factor for second moment (usually 0.999)
- $\eta$ : base learning rate
- $\epsilon$ : small constant to avoid division by zero

## Intuition: how Adam does "automatic step sizing"
You can view Adam's effective learning rate as:

$$
\text{effective lr} = \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}
$$

- Large $\hat{v}_t$ (high gradient variance) → smaller step (stability)
- Small $\hat{v}_t$ (stable gradients) → relatively larger step (faster progress)

So Adam updates "carefully where noisy, aggressively where stable."

## Why AdamW?
If you combine Adam with naive L2 regularization, weight decay can interact with adaptive scaling in an unintended way.
AdamW **decouples** weight decay from the adaptive gradient update.

Practical takeaway:
- Transformer/ViT/LLM-style training: AdamW is usually the default
- Unless you have a specific reason, start from AdamW rather than plain Adam

## Implementation

```python
import torch

# AdamW (recommended default)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

## Adam vs AdamW

| Item | Adam + L2 | AdamW |
|---|---|---|
| Weight decay application | mixed into gradient | applied separately to parameters |
| Interaction with adaptive scaling | stronger | weaker (more faithful decay behavior) |
| Typical choice for Transformer-like models | usually not preferred | preferred |

## Starter hyperparameters (beginner-friendly)
- **lr**: `1e-4` for Transformer/ViT, often `1e-3` for smaller models
- **betas**: start from `(0.9, 0.999)`
- **weight_decay**: start around `0.01`, increase if overfitting is strong
- **eps**: keep default `1e-8` unless debugging numeric issues

## 8-bit Adam (memory saving)
In large models, optimizer states consume significant memory. 8-bit Adam is a common option.

```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4,
)
```

Pros: reduces optimizer-state memory usage.
Caution: numerical behavior may vary by hardware/library stack, so run a short validation check.

## Debugging checklist
- [ ] If loss oscillates early, did you try lowering `lr` by 2~10x?
- [ ] If validation underperforms, is `weight_decay` too weak/strong?
- [ ] If gradient norm spikes, did you apply gradient clipping (`max_norm`)?
- [ ] In mixed precision, if NaN appears, did you check scaler/autocast settings?
- [ ] Are you accidentally comparing Adam and AdamW baselines inconsistently?

## Common mistakes (FAQ)
**Q1. Is Adam always better than SGD?**  
A. Not always. Adam/AdamW often converges faster, but SGD can still win on final generalization for some tasks.

**Q2. If I use AdamW, should I always set large weight decay?**  
A. No. Too large decay causes underfitting. Start around 0.01 and tune using validation metrics.

**Q3. Should I tune β1 and β2 first?**  
A. Usually no for beginners. Keep `(0.9, 0.999)` and tune learning rate / weight decay first.

## Related Content
- [SGD](/en/docs/components/training/optimizer/sgd)
- [LR Scheduler](/en/docs/components/training/optimizer/lr-scheduler)
- [Weight Decay](/en/docs/components/training/regularization/weight-decay)
- [LoRA](/en/docs/components/training/peft/lora)
