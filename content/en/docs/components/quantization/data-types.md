---
title: "Data Types"
weight: 3
math: true
---

# Data Types

{{% hint info %}}
**Prerequisites**: None
{{% /hint %}}

## One-line Summary
> **A data type defines how many bits represent numbers, which directly controls speed, memory, and accuracy trade-offs.**

## Why is this needed?
Vision models often have millions (or billions) of values.
Choosing FP32 vs FP16/BF16 vs INT8 changes:

- GPU memory usage
- inference speed and power draw
- possible accuracy drop

Analogy: storing the same photo as a full-quality RAW file (FP32) vs a compressed mobile image (INT8).

## Core formula (quantization basics)
A common integer quantization form is:

$$
q = \mathrm{round}\left(\frac{x}{s}\right) + z
$$

$$
\hat{x} = s(q - z)
$$

**Symbol meanings:**
- $x$ : original real value (e.g., FP32 activation)
- $q$ : quantized integer value (INT8, etc.)
- $s$ : scale factor (real-to-integer conversion)
- $z$ : zero-point (where real zero maps in integer space)
- $\hat{x}$ : reconstructed approximation

Main idea: reducing bits lowers cost, but may increase approximation error.

## Quick map of common types

| Type | Bits | Range / characteristic | Typical use |
|---|---:|---|---|
| FP32 | 32 | wide range, high precision | baseline training/debug |
| FP16 | 16 | narrower range, fast | inference, mixed precision |
| BF16 | 16 | FP32-like exponent range | stable training + speed |
| INT8 | 8 | integer representation | PTQ/QAT inference |
| INT4 | 4 | stronger compression | ultra-light inference |
| FP8 | 8 | accelerator-optimized | latest GPU training/inference |

## Floating-point formats

### FP32 (Single Precision)
```
| 1 bit | 8 bits  | 23 bits    |
| sign  | exponent| mantissa   |
```
- Range: about $\pm 3.4 \times 10^{38}$
- Precision: about 7 decimal digits
- Strength: robust baseline precision

### FP16 (Half Precision)
```
| 1 bit | 5 bits  | 10 bits    |
| sign  | exponent| mantissa   |
```
- Range: about $\pm 65504$
- Lower precision than FP32
- Strength: half memory, faster compute
- Caution: very small gradients can underflow to zero

### BF16 (Brain Float)
```
| 1 bit | 8 bits  | 7 bits     |
| sign  | exponent| mantissa   |
```
- Same exponent width as FP32, so range is much wider than FP16
- Often more stable than FP16 for training

## Integer formats

### INT8
```
| 1 bit | 7 bits  |
| sign  | value   |
```
- Range: -128 to 127 (signed)
- Common default for post-training quantization

### INT4
```
| 4 bits |
| value  |
```
- Range: -8 to 7 (signed)
- Better compression, higher risk of accuracy loss

## Special formats

### FP8 (E4M3, E5M2)
```
E4M3: | 1 | 4 | 3 |
E5M2: | 1 | 5 | 2 |
```
- E4M3: precision-oriented
- E5M2: range-oriented

### NF4 (4-bit NormalFloat)
Used in QLoRA as a non-uniform 4-bit representation.

```python
# Example quantization points optimized for normally distributed weights
nf4_quantiles = [
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0
]
```

## Minimal runnable example

```python
import torch

x_fp32 = torch.randn(1024, 1024)
x_fp16 = x_fp32.half()
x_bf16 = x_fp32.bfloat16()

print("bytes per element")
print("fp32:", x_fp32.element_size())  # 4
print("fp16:", x_fp16.element_size())  # 2
print("bf16:", x_bf16.element_size())  # 2
```

## Mixed precision training (practical)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in train_loader:
    optimizer.zero_grad()

    with autocast():  # some ops run in FP16/BF16 automatically
        pred = model(x)
        loss = criterion(pred, y)

    scaler.scale(loss).backward()  # helps reduce underflow issues
    scaler.step(optimizer)
    scaler.update()
```

## Two beginner pain points: symmetric/asymmetric + per-tensor/per-channel
When people start using INT8, the most common confusion is mixing these two axes.

1. **symmetric vs asymmetric**: whether zero-point ($z$) is fixed to 0 or allowed to shift
2. **per-tensor vs per-channel**: whether one scale ($s$) is shared by the whole tensor or assigned per channel

### (1) Symmetric vs asymmetric
- **Symmetric quantization**: typically uses $z=0$ (simpler, hardware-friendly)
- **Asymmetric quantization**: allows $z\neq 0$ (often helpful when activation distribution is skewed)

For integer range $[q_{\min}, q_{\max}]$, a common form is:

$$
s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \quad
z = \mathrm{round}\left(q_{\min} - \frac{x_{\min}}{s}\right)
$$

### (2) Per-tensor vs per-channel
- **Per-tensor**: one scale for the whole tensor → simple and fast
- **Per-channel**: one scale per channel → usually better accuracy (especially for Conv weights)

In practice, many pipelines use **per-tensor for activations** and **per-channel for weights**.

## 5-minute experiment #2: per-tensor vs per-channel error
The snippet below compares reconstruction error (MAE) for Conv-like weights under both choices.

```python
import torch


def fake_int8_quant_per_tensor(w: torch.Tensor):
    qmin, qmax = -128, 127
    w_min, w_max = w.min(), w.max()
    s = (w_max - w_min) / (qmax - qmin + 1e-12)
    z = torch.round(torch.tensor(qmin, device=w.device) - w_min / (s + 1e-12))
    q = torch.round(w / (s + 1e-12) + z).clamp(qmin, qmax)
    w_hat = (q - z) * s
    return w_hat


def fake_int8_quant_per_channel(w: torch.Tensor):
    # Assume Conv2d weights: [out_channels, in_channels, kH, kW]
    qmin, qmax = -128, 127
    w_flat = w.view(w.size(0), -1)  # flatten per output channel

    w_min = w_flat.min(dim=1, keepdim=True).values
    w_max = w_flat.max(dim=1, keepdim=True).values

    s = (w_max - w_min) / (qmax - qmin + 1e-12)
    z = torch.round(qmin - w_min / (s + 1e-12))

    q = torch.round(w_flat / (s + 1e-12) + z).clamp(qmin, qmax)
    w_hat = (q - z) * s
    return w_hat.view_as(w)


w = torch.randn(64, 64, 3, 3) * torch.linspace(0.2, 2.0, 64).view(64, 1, 1, 1)

w_hat_tensor = fake_int8_quant_per_tensor(w)
w_hat_channel = fake_int8_quant_per_channel(w)

mae_tensor = (w - w_hat_tensor).abs().mean().item()
mae_channel = (w - w_hat_channel).abs().mean().item()

print(f"MAE per-tensor : {mae_tensor:.6f}")
print(f"MAE per-channel: {mae_channel:.6f}")
```

As channel-wise scale variance grows, per-channel quantization is often more accurate.

## Debugging checklist
- [ ] If NaN/Inf appears, did you first check FP16 underflow/overflow risk?
- [ ] If BF16 is supported, are you unnecessarily fixed to FP16 only?
- [ ] For INT8 inference, does calibration data match real deployment distribution?
- [ ] If accuracy drop is large, did you test per-channel quantization?
- [ ] Are you applying the exact same quantization policy to activations and weights without validation?

## Common mistakes (FAQ)
**Q1. Is FP16 always better than FP32?**  
A. It is often better for speed/memory, but numerical stability can be worse for some training setups.

**Q2. Which should I try first for training: BF16 or FP16?**  
A. If hardware supports BF16, many teams try BF16 first because its wider range is usually more stable.

**Q3. Does INT8 always cause huge accuracy loss?**  
A. Not always. The drop depends on model architecture, calibration quality, and data distribution.

## Beginner decision guide (what should I pick first?)
Use this order to reduce trial-and-error.

1. If **training stability** is the top priority, start with FP32 or BF16
2. If **memory is tight**, enable FP16/BF16 mixed precision
3. If **latency/power in deployment** is the goal, try INT8 (PTQ first, then QAT if needed)

Quick decision table:

| Situation | First choice | Why |
|---|---|---|
| first training run / debugging | FP32 or BF16 | easier root-cause isolation and reproducibility |
| GPU memory bottleneck | BF16/FP16 mixed precision | strong memory savings with speedup |
| reducing server inference cost | INT8 PTQ | large practical gain with moderate complexity |
| PTQ accuracy drop is too large | INT8 QAT | model adapts to quantization noise during training |

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First thing to inspect |
|---|---|---|
| loss suddenly becomes NaN/Inf | FP16 overflow/underflow | GradScaler logs, max grad norm |
| validation accuracy fluctuates heavily | mixed dtypes in unintended places | autocast scope, per-op dtype |
| specific classes degrade after INT8 deployment | calibration-distribution mismatch | class balance in calibration set |
| latency drops but memory barely changes | activations still FP16/FP32 | weight-only vs full quantization |

## 5-minute mini experiment: feel dtype error directly
The snippet below quickly measures mean absolute reconstruction error (MAE)
after converting the same tensor to FP16/BF16 and back to FP32.

```python
import torch


def reconstruction_mae(x: torch.Tensor, target_dtype: torch.dtype) -> float:
    x_q = x.to(target_dtype)
    x_hat = x_q.to(torch.float32)
    return (x - x_hat).abs().mean().item()


x = torch.randn(4096, dtype=torch.float32) * 3.0

mae_fp16 = reconstruction_mae(x, torch.float16)
mae_bf16 = reconstruction_mae(x, torch.bfloat16)

print(f"MAE(fp16): {mae_fp16:.6f}")
print(f"MAE(bf16): {mae_bf16:.6f}")
```

How to read it:
- Smaller error means better value preservation.
- Error patterns can differ between small/large magnitude tensors, so repeat on real feature distributions.

## Formula → code mapping (4-step quantization loop)
When implementing quantization for the first time, this fixed 4-step loop prevents most mistakes.

1. **Collect ranges (calibration)**
   - Formula view: estimate $x_{\min}, x_{\max}$
   - Code view: gather min/max (or percentile) stats on calibration data

2. **Compute scale/zero-point**
   - Formula view: compute $s, z$
   - Code view: choose `qmin, qmax` for target dtype (INT8, etc.) and derive scale/zero-point

3. **Quantize (real → integer)**
   - Formula view: $q = \mathrm{round}(x/s) + z$
   - Code view: round + clamp into integer range

4. **Dequantize + validate error (integer → real)**
   - Formula view: $\hat{x} = s(q-z)$
   - Code view: dequantize, then check MAE/MSE and task metrics (top-1, mAP)

Beginner note:
- If quality drops a lot, re-check step 1 (range collection) first.
- If deployment input distribution differs from calibration distribution, performance can collapse even with correct math/code in steps 2~4.

## Visual asset prompts (Nanobanana)
- **EN Diagram 1 (dtype trade-off map)**  
  "Dark theme background (#1a1a2e), data-type comparison infographic. Axes: x=efficiency (speed/memory), y=accuracy/stability. Place FP32, FP16, BF16, INT8, INT4, FP8 as labeled points. Add short labels such as 'baseline precision', 'stable training', 'deployment optimization'. Draw arrows to show precision-vs-efficiency trade-off. Clean educational vector style, high readability"

- **EN Diagram 2 (4-step quantization pipeline)**  
  "Dark theme background (#1a1a2e), 4-step quantization pipeline diagram. 1) Calibration(x_min/x_max) 2) Scale/zero-point computation 3) Quantize(round+clamp) 4) Dequantize + error check(MAE/mAP). Include step boxes with arrows, formula snippets q=round(x/s)+z and x_hat=s(q-z), concise English labels, clean instructional vector style"

## Related Content
- [PTQ](/en/docs/components/quantization/ptq)
- [QAT](/en/docs/components/quantization/qat)
- [QLoRA](/en/docs/components/training/peft/qlora)
