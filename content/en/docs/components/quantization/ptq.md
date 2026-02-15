---
title: "PTQ"
weight: 1
math: true
---

# PTQ (Post-Training Quantization)

{{% hint info %}}
**Prerequisites**: [Data Types](/en/docs/components/quantization/data-types)
{{% /hint %}}

## One-line Summary
> **PTQ converts a pretrained model to INT8/INT4 without retraining, making it the fastest way to reduce inference latency and memory.**

## Why is this needed?
In deployment, latency, memory, and serving cost are often more urgent than squeezing out the last 1~2% accuracy.
PTQ is useful when:

- retraining is expensive or unavailable,
- hardware memory is limited,
- you want a quick baseline before moving to [QAT](/en/docs/components/quantization/qat).

## Overview

PTQ quantizes an already-trained model without additional training. It is simple and fast, but may cause accuracy loss.

## Quantization formulas

### Asymmetric quantization

$$
Q(x) = \text{round}\left(\frac{x - x_{min}}{s}\right), \quad s = \frac{x_{max} - x_{min}}{2^b - 1}
$$

Dequantization:
$$
\hat{x} = s \cdot Q(x) + x_{min}
$$

### Symmetric quantization

$$
Q(x) = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}
$$

**Symbol guide (must-know for beginners):**
- $x$: original real-valued tensor element (weight/activation)
- $Q(x)$: mapped integer value (INT8, etc.)
- $\hat{x}$: dequantized approximation in real space
- $x_{min}, x_{max}$: observed min/max values (estimated during calibration)
- $s$: scale (real-to-integer step size)
- $b$: bit width (e.g., $b=8$ for INT8)

**When to use symmetric vs asymmetric?**
- Symmetric: often used for zero-centered distributions (especially weights), simpler pipeline
- Asymmetric: often better when distributions are shifted away from zero (common in activations)

```python
def symmetric_quantize(x, bits=8):
    """Symmetric quantization"""
    qmax = 2 ** (bits - 1) - 1

    scale = x.abs().max() / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)

    return q.to(torch.int8), scale


def asymmetric_quantize(x, bits=8):
    """Asymmetric quantization"""
    qmin, qmax = 0, 2 ** bits - 1

    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(-x_min / scale)

    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)

    return q.to(torch.uint8), scale, zero_point
```

## Calibration

Finding good scale/zero-point values:

```python
def calibrate(model, calibration_data, method='minmax'):
    """
    Estimate quantization parameters from calibration data
    """
    stats = {}

    def hook(name):
        def fn(module, input, output):
            if name not in stats:
                stats[name] = {'min': [], 'max': []}
            stats[name]['min'].append(output.min().item())
            stats[name]['max'].append(output.max().item())
        return fn

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)

    for h in handles:
        h.remove()

    scales = {}
    for name, s in stats.items():
        if method == 'minmax':
            scales[name] = {
                'min': min(s['min']),
                'max': max(s['max'])
            }
        elif method == 'percentile':
            scales[name] = {
                'min': np.percentile(s['min'], 1),
                'max': np.percentile(s['max'], 99)
            }

    return scales
```

## Granularity

| Level | # of scales | Accuracy | Efficiency |
|------|-------------|----------|------------|
| Per-tensor | 1 per tensor | lower | higher |
| Per-channel | 1 per channel | higher | medium |
| Per-group | 1 per N elements | highest | lower |

```python
def per_channel_quantize(weight, bits=8):
    """Per-channel quantization (Conv, Linear)"""
    qmax = 2 ** (bits - 1) - 1

    if weight.dim() == 4:  # Conv: (out, in, h, w)
        scales = weight.abs().amax(dim=(1, 2, 3)) / qmax
        q = torch.round(weight / scales.view(-1, 1, 1, 1))
    else:  # Linear: (out, in)
        scales = weight.abs().amax(dim=1) / qmax
        q = torch.round(weight / scales.view(-1, 1))

    return q.clamp(-qmax, qmax).to(torch.int8), scales
```

## Weight-only quantization

```python
class WeightOnlyLinear(nn.Module):
    def __init__(self, weight, scale, bits=4):
        super().__init__()
        self.register_buffer('weight_q', weight)  # INT4
        self.register_buffer('scale', scale)
        self.bits = bits

    def forward(self, x):
        weight = self.weight_q.float() * self.scale
        return F.linear(x, weight)
```

## PyTorch dynamic/static quantization

```python
import torch.quantization as quant

# Dynamic quantization
model_dynamic = quant.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# Static quantization (requires calibration)
model.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model)

with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

model_static = quant.convert(model_prepared)
```

## 10-minute PTQ quick procedure (beginner-friendly)
1. **Save FP32 baseline first**: record accuracy/latency/memory before quantization
2. **Prepare calibration samples**: use 100~1000 examples close to real deployment distribution
3. **Run simple INT8 PTQ first**: start with per-tensor + minmax
4. **Inspect sensitive layers**: check early Conv blocks and final detection/classification heads
5. **Apply low-cost improvements**: compare per-channel or percentile calibration

Beginner rule: **change one option at a time, and always compare against the FP32 baseline**.

## Mini experiment: quantify round-trip error (MSE)
The snippet below performs a simple FP32 → INT8 → FP32 round-trip and reports quantization error.

```python
import torch


def fake_ptq_roundtrip(x: torch.Tensor, bits: int = 8):
    # Symmetric per-tensor quantization
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max().clamp(min=1e-8) / qmax

    x_int = torch.round(x / scale).clamp(-qmax, qmax)
    x_deq = x_int * scale

    mse = torch.mean((x - x_deq) ** 2).item()
    max_abs_err = torch.max(torch.abs(x - x_deq)).item()
    return mse, max_abs_err


x = torch.randn(4096) * 0.5  # example activation distribution
mse, max_abs_err = fake_ptq_roundtrip(x, bits=8)
print(f"MSE={mse:.6e}, max_abs_err={max_abs_err:.6e}")
```

Interpretation guide:
- If MSE is small (often around `1e-6`~`1e-4`) and max error is acceptable, PTQ is likely viable for that block.
- Comparing this per layer quickly reveals **which blocks are quantization-sensitive**.
- For sensitive layers, try per-channel quantization or keep them in FP16 (partial quantization).

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First action |
|---|---|---|
| Sharp accuracy drop right after INT8 conversion | calibration distribution mismatch | rebuild calibration set (lighting/resolution/class balance) |
| Only specific classes degrade strongly | class-biased calibration samples | recalibrate with class-balanced samples |
| Speed gain is smaller than expected | too many FP32 fallback ops | inspect backend op support + trace unsupported layers |
| Unstable logits/box outputs | over-aggressive activation quantization | keep sensitive blocks in FP16/FP32 (partial quantization) |

## Practical debugging checklist
- [ ] Does calibration data match real deployment distribution (brightness, resolution, class mix)?
- [ ] Is accuracy drop concentrated in specific blocks (early Conv, final head)?
- [ ] If per-tensor fails, did you compare with per-channel quantization?
- [ ] Are unsupported ops causing FP32 fallback in TensorRT/ONNX Runtime?
- [ ] Did you benchmark after warm-up, not only first-run latency?

## Common mistakes (FAQ)
**Q1. Does PTQ always cause only a small accuracy drop?**  
A. No. It can be large depending on architecture and activation distribution.

**Q2. Can I do PTQ with zero data?**  
A. Technically yes in some forms, but not recommended. Even a small representative calibration set helps a lot.

**Q3. PTQ or QAT first?**  
A. Usually PTQ first for a fast baseline, then move to QAT if accuracy loss is beyond tolerance.

## Next-step guide
1. Measure PTQ baseline (accuracy/latency/memory)
2. Apply low-cost fixes (per-channel, percentile calibration)
3. If still insufficient, switch to [QAT](/en/docs/components/quantization/qat)

## Related Content

- [QAT](/en/docs/components/quantization/qat)
- [Data Types](/en/docs/components/quantization/data-types)
- [QLoRA](/en/docs/components/training/peft/qlora)
