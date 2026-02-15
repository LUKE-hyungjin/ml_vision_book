---
title: "SGD"
weight: 1
math: true
---

# SGD (Stochastic Gradient Descent)

{{% hint info %}}
**Prerequisites**: [Gradient](/en/docs/math/calculus/gradient)
{{% /hint %}}

## One-line Summary
> **SGD updates parameters step by step in the opposite direction of the gradient to reduce loss.**

## Why is this needed?
Training a neural network means repeatedly deciding how to adjust parameters.
SGD is the simplest and most interpretable way to do that.

Analogy: if you are blindfolded on a hill, you feel the local slope (gradient) and take a small step downhill.

- Pros: simple, low memory usage, often strong generalization for large CNN training
- Cons: sensitive to learning rate, can be noisy/slow to converge

## Basic SGD

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

**Symbol meanings**
- $\theta_t$: model parameters at step $t$
- $L(\theta_t)$: loss at current parameters
- $\nabla L(\theta_t)$: gradient of the loss
- $\eta$: learning rate

### Intuition
- The gradient points uphill, so moving in the opposite direction reduces loss.
- If $\eta$ is too large, updates overshoot minima.
- If $\eta$ is too small, training becomes very slow.

```python
import torch

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for x, y in dataloader:
    optimizer.zero_grad()        # clear previous gradients
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()              # compute gradients
    optimizer.step()             # update parameters
```

## Most common beginner confusion: SGD vs mini-batch SGD
The textbook equation often looks like true stochastic updates (one sample at a time),
but practical training usually uses the **average gradient over a mini-batch**.
That is exactly what a standard PyTorch `DataLoader` loop does.

$$
g_t = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell_i(\theta_t),
\qquad
\theta_{t+1} = \theta_t - \eta g_t
$$

- $B$: batch size
- $\ell_i$: loss for sample $i$
- $g_t$: batch-averaged gradient

Intuition:
- If batch size is too small, gradient noise increases and loss may oscillate.
- If batch size is too large, memory cost rises and generalization can become less robust.

Practical starting heuristics:
- If memory is tight, reduce batch size and use gradient accumulation to increase **effective batch size**.
- If batch size is doubled, test a modest learning-rate increase (e.g., 1.5~2x) and verify stability.

### Why adjust learning rate when batch size changes?
With a larger batch, the averaged gradient is usually less noisy, so a slightly larger step can be stable.
Use the rule below as a **starting point**, not a strict law.

$$
\eta_2 = \eta_1 \times \frac{B_2}{B_1}
$$

- $\eta_1, \eta_2$: learning rates before/after batch-size change
- $B_1, B_2$: batch sizes before/after the change

Notes:
- This is not always optimal; it is an initial search rule.
- Warmup, augmentation strength, and model architecture can make strict linear scaling too aggressive.
- In practice, test 2~3 nearby candidates around this value (lower / base / higher).

## SGD with Momentum
Momentum remembers part of the previous update direction to reduce oscillation.

$$
\begin{aligned}
v_t &= \mu v_{t-1} + \nabla L(\theta_t),\\
\theta_{t+1} &= \theta_t - \eta v_t
\end{aligned}
$$

- $\mu$: momentum factor (commonly 0.9)
- Effect: less zig-zag and faster progress in narrow valleys

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)
```

## Nesterov Momentum
Nesterov computes gradient at a look-ahead position for a better correction.

$$
v_t = \mu v_{t-1} + \nabla L(\theta_t - \eta \mu v_{t-1})
$$

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True,
)
```

## Using Weight Decay
Weight decay helps control overly large weights and can improve generalization.

$$
\theta_{t+1} = \theta_t - \eta(\nabla L + \lambda \theta_t)
$$

- $\lambda$: weight decay strength

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)
```

In real projects, bias and normalization parameters are often excluded from weight decay.

```python
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or n.endswith(".bias"):
        no_decay.append(p)   # BN/LN scale, bias, etc.
    else:
        decay.append(p)

optimizer = torch.optim.SGD(
    [
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=0.01,
    momentum=0.9,
    nesterov=True,
)
```

## SGD vs Adam (practical)

| Item | SGD | Adam/AdamW |
|---|---|---|
| Early convergence speed | usually slower | usually faster |
| Final generalization | often strong (CNN) | strong for rapid iteration |
| Hyperparameter sensitivity | higher | relatively lower |
| Memory usage | lower | higher |

## Practical debugging checklist
- [ ] **Check learning rate first**: if loss diverges, test 10x lower lr
- [ ] **Monitor gradient norm**: log `grad_norm` for explosion signs
- [ ] **Tune momentum when oscillating**: e.g., 0.9 → 0.85
- [ ] **Review weight-decay scope**: avoid blindly applying same decay to norm/bias params
- [ ] **Verify scheduler + warmup**: large initial lr without warmup can destabilize training

## 5-minute mini practice: feel a 10x learning-rate gap
The snippet below keeps model/data fixed and changes only learning rate for 20 steps.
This quickly builds the beginner intuition that "SGD is highly sensitive to lr."

```python
import torch
import torch.nn as nn
import torch.optim as optim


def run_once(lr: float):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    losses = []
    for _ in range(20):
        x = torch.randn(128, 32)
        y = torch.randint(0, 10, (128,))

        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        losses.append(float(loss))

    return losses

loss_lr_1e2 = run_once(1e-2)
loss_lr_1e1 = run_once(1e-1)

print("lr=1e-2 first/last:", round(loss_lr_1e2[0], 4), round(loss_lr_1e2[-1], 4))
print("lr=1e-1 first/last:", round(loss_lr_1e1[0], 4), round(loss_lr_1e1[-1], 4))
```

How to read it:
- `lr=1e-2` is usually more stable.
- `lr=1e-1` may drop faster at first, but often oscillates more depending on init/data.
- In practice, we tune this stability/speed tradeoff with a scheduler.

## Common mistakes (FAQ)
**Q1. Is SGD always worse than Adam because it is slower?**  
A. No. It may converge slower early, but final generalization can be better in many setups.

**Q2. Is higher momentum always better?**  
A. No. Too much momentum can increase overshoot and instability.

**Q3. Loss is noisy across steps. Is training broken?**  
A. Not necessarily. Mini-batch SGD is naturally noisy. Focus on trend (moving average), not single-step values.

## Beginner quick-start recipe (common for vision classification/detection)
This gives a minimal "where to start" setup when you do not yet have tuning intuition.

- Optimizer: `SGD(momentum=0.9, nesterov=True)`
- Initial lr: `0.01` (starting point for batch size 64)
- Weight decay: `1e-4`
- Scheduler: warmup for 3-5 epochs, then cosine decay

Simple batch-size heuristic:
- If batch size is doubled → test lr in the 1.5~2x range
- If batch size is halved → first try lr at around 1/2~2/3 for stability

## First-response actions by failure pattern (ops checklist)
- **Loss explodes in first 100~300 steps** → lower lr by 10x and add warmup
- **Train improves but val stalls** → strengthen weight decay/augmentation
- **NaN or inf gradients** → temporarily add grad clipping (e.g., `max_norm=1.0`) and trace root cause
- **Training is too slow** → re-check scheduler and jointly retune `momentum`/batch size

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First check |
|---|---|---|
| loss explodes early | learning rate too high, no warmup | reduce lr by 10x, verify warmup |
| training loss drops but validation stalls | weak regularization | weight decay strength, augmentation setup |
| strong step-to-step oscillation | momentum too high or tiny batch | momentum 0.9→0.85, batch size |
| training is very slow | lr too low or no scheduler | increase base lr, add cosine/step scheduler |

## Related Content
- [Adam](/en/docs/components/training/optimizer/adam)
- [LR Scheduler](/en/docs/components/training/optimizer/lr-scheduler)
- [Weight Decay](/en/docs/components/training/regularization/weight-decay)
