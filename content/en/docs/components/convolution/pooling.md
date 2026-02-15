---
title: "Pooling"
weight: 2
math: true
---

# Pooling

{{% hint info %}}
**Prerequisites**: [Conv2D](/en/docs/components/convolution/conv2d)
{{% /hint %}}

> **One-line summary**: Pooling shrinks feature maps while keeping important signals, which reduces compute and improves robustness to small shifts.

## Why is pooling needed?

### Problem 1: feature maps grow expensive
As CNN depth grows, channels increase and compute cost rises quickly. If we reduce spatial size from `H×W` to `(H/2)×(W/2)`, many downstream operations become about **4× cheaper**.

### Problem 2: tiny position shifts change activations
A cat moved by 1–2 pixels is still a cat. Pooling keeps a regional summary, so the model is less sensitive to tiny translations.

{{< figure src="/images/components/convolution/en/pooling-overview.jpeg" caption="Pooling reduces spatial size and improves local translation robustness" >}}

## Max Pooling

### Formula
$$
y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}
$$

**Symbol meanings:**
- $R_{i,j}$: input region (often 2×2) mapped to output position $(i,j)$
- $x_{m,n}$: value in the input region
- $y_{i,j}$: pooled output value

### Intuition
Max pooling asks: “What is the strongest activation in this local window?”
It preserves dominant evidence (edge/texture response) and discards weaker values.

### Minimal code
```python
import torch
import torch.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.tensor([[[[1., 3., 2., 1.],
                    [5., 6., 7., 8.],
                    [4., 2., 1., 3.],
                    [10., 14., 12., 16.]]]])

y = pool(x)
print(y)
# tensor([[[[ 6.,  8.],
#           [14., 16.]]]])
print(x.shape, "->", y.shape)
# torch.Size([1, 1, 4, 4]) -> torch.Size([1, 1, 2, 2])
```

## Average Pooling

### Formula
$$
y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} x_{m,n}
$$

Average pooling keeps smoother regional information, while max pooling keeps peak evidence.

{{< figure src="/images/components/convolution/en/pooling-max-avg.png" caption="Max pooling keeps strongest responses; average pooling keeps regional mean" >}}

## Global Average Pooling (GAP)

GAP compresses each channel to one number:
- input `(B, C, H, W)`
- output `(B, C)`

This is widely used before classifiers in modern CNNs to reduce parameters and overfitting risk.

```python
import torch
import torch.nn as nn

gap = nn.AdaptiveAvgPool2d(1)
x = torch.randn(32, 512, 7, 7)
y = gap(x).flatten(1)
print(y.shape)  # torch.Size([32, 512])
```

## Practical debugging checklist
1. Confirm tensor order is `(B, C, H, W)`.
2. Verify output size from kernel/stride/padding settings.
3. If small objects disappear, pooling may be too aggressive.
4. If localization quality drops, replace some pooling with stride-conv or reduce downsampling depth.

## Common mistakes (FAQ)
- **Q. Is pooling always required in modern CNNs?**
  A. Not always. Many models use stride convolution for learnable downsampling.

- **Q. Why did accuracy drop after adding more pooling?**
  A. Over-downsampling can remove fine detail, especially for detection/segmentation.

- **Q. Max or average pooling?**
  A. Max is common in middle feature extraction; average is common for final global summarization (GAP).

## Output-size quick formula (must-know for beginners)

Pooling output size is computed almost the same way as convolution.

$$
O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1
$$

- $I$: input size
- $K$: kernel size
- $P$: padding
- $S$: stride
- $O$: output size

Example: `I=7, K=2, S=2, P=0`

$$
O = \left\lfloor \frac{7-2}{2} \right\rfloor + 1 = \lfloor 2.5 \rfloor + 1 = 3
$$

Key point:
- if division is not exact, output uses **floor** by default.
- this is a common source of unexpected shape mismatches with odd-sized inputs.

## Failure pattern in practice: small objects vanish

If you downsample too early and too aggressively, small-object signals can disappear before deeper layers use them.

Quick intuition:
- an 8×8 object after three stride-2 steps becomes roughly 1×1
- at that point, detailed boundary cues are mostly gone

Practical fixes (in order):
1. reduce early downsampling strength
2. keep higher-resolution branches (e.g., FPN in detection)
3. test learnable stride-conv instead of fixed max-pooling

## 1-minute debugging routine (Pooling)

- [ ] Did you print feature-map shapes at each stage?
- [ ] Did you verify floor behavior on odd input sizes?
- [ ] For small-object tasks, is early downsampling too strong?
- [ ] Are you separating classification design (GAP-heavy) from detection/segmentation design (resolution-preserving)?

## Related Content
- [Conv2D](/en/docs/components/convolution/conv2d)
- [Receptive Field](/en/docs/components/convolution/receptive-field)
- [AlexNet](/en/docs/architecture/cnn/alexnet)
- [ResNet](/en/docs/architecture/cnn/resnet)
