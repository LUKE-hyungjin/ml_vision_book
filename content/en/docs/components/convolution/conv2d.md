---
title: "2D Convolution"
weight: 1
math: true
---

# 2D Convolution

{{% hint info %}}
**Prerequisites**: [Matrix](/en/docs/math/linear-algebra/matrix)
{{% /hint %}}

## One-line Summary
> **Convolution asks: “Does this local patch look like the pattern this filter is searching for?”**

## Why is this needed?
A raw 224×224 RGB image has 150,528 values. If we flatten and feed it to a dense layer, parameters become huge and spatial relationships are weakened.

Convolution keeps training practical by:
- scanning small local windows,
- reusing the same filter everywhere,
- building hierarchical features from edges → textures → parts → objects.

## Two core principles beginners must not miss

Many learners memorize formulas but miss *why* Conv2d works in practice. Keep these two principles explicit:

1. **Local Connectivity**
   - Each output location looks at only a small input patch (K×K), not the whole image.
   - This keeps parameters manageable and captures local patterns like edges/corners.

2. **Weight Sharing**
   - The same kernel weights are reused at every spatial location.
   - So the model can detect the same pattern even when it appears in different positions.

In one sentence: **Conv2d looks locally, but applies one shared pattern detector globally.**

## Formula
$$
\mathrm{output}(i,j)=\sum_{m=0}^{K-1}\sum_{n=0}^{K-1}\mathrm{input}(i+m,j+n)\cdot \mathrm{kernel}(m,n)
$$

**Symbol meanings:**
- $\mathrm{input}(i+m,j+n)$: pixel value at local position
- $\mathrm{kernel}(m,n)$: learnable filter weight
- $K$: kernel size (commonly 3)
- $\mathrm{output}(i,j)$: feature value at output location $(i,j)$

## Intuition
Think of a small stamp sliding over an image.
- If the local patch matches the stamp pattern, output is high.
- If not, output is low.

So convolution is a **pattern-matching score map**.

## Output size formula
$$
O = \frac{I - K + 2P}{S} + 1
$$

- $I$: input size
- $K$: kernel size
- $P$: padding
- $S$: stride
- $O$: output size

Example:
- $I=32, K=3, P=1, S=1 \Rightarrow O=32$ (size preserved)
- $I=32, K=3, P=1, S=2 \Rightarrow O=16$ (downsampled)

## Minimal implementation
```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 224, 224)
y = conv(x)

print(x.shape)  # torch.Size([1, 3, 224, 224])
print(y.shape)  # torch.Size([1, 64, 224, 224])
```

## Practical debugging checklist
1. **Shape order**: PyTorch expects `(B, C, H, W)`.
2. **Padding mismatch**: if feature size unexpectedly shrinks, check `padding`.
3. **Stride side effects**: `stride=2` halves spatial resolution.
4. **Channel mismatch**: `in_channels` must equal input tensor channels.
5. **NaN spikes**: inspect learning rate, normalization placement, and mixed precision settings.

## Common mistakes (FAQ)
- **Q. Is bigger kernel always better?**  
  A. Usually no. Stacked 3×3 kernels are often more efficient and expressive.

- **Q. Does convolution preserve absolute position perfectly?**  
  A. Not exactly. It is robust to small translations, but strong invariance depends on full architecture (pooling/stride/augmentation).

- **Q. Why does my model overfit with conv layers?**  
  A. Check data size, augmentation, regularization, and model capacity before only changing kernel size.

## Beginner safety block: 60-second shape tracing

Most Conv2d bugs are shape bugs. Print these 3 lines first.

```python
import torch
import torch.nn as nn

x = torch.randn(2, 3, 32, 32)                 # (B, C, H, W)
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
y = conv(x)

print("input :", x.shape)   # torch.Size([2, 3, 32, 32])
print("weight:", conv.weight.shape)  # torch.Size([16, 3, 3, 3])
print("output:", y.shape)   # torch.Size([2, 16, 16, 16])
```

How to read it:
- `weight.shape = [C_out, C_in, K, K]`
- output channels are determined by `C_out`
- with `stride=2`, spatial size `(H, W)` is usually halved

## Symptom → likely cause quick map

| Observed symptom | Most common cause | First thing to inspect |
|---|---|---|
| `Expected ... to have 3 channels, but got 1` | input-channel mismatch | `in_channels`, RGB vs grayscale preprocessing |
| output is smaller than expected | missing padding or large stride | `padding`, `stride`, output-size formula |
| NaN loss in early training | high LR or unstable mixed precision | learning rate, AMP settings, gradient clipping |
| parameter count explodes | overly large channel/kernel setup | `C_in`, `C_out`, kernel size choices |

## Formula ↔ code bridge: pre-check output size

A very common beginner failure is: **the formula is memorized, but shape becomes non-integer in code**.
If you pre-check output size before defining many layers, you prevent a lot of mid-training shape crashes.

```python
def conv2d_out(i, k=3, p=1, s=1):
    numer = i - k + 2 * p
    assert numer % s == 0, (
        f"Output size is not integer: (I-K+2P)={numer}, stride={s}"
    )
    return numer // s + 1

h = conv2d_out(i=31, k=3, p=1, s=2)
print(h)  # 16
```

Checklist:
- Does `(I - K + 2P)` divide evenly by `stride`?
- If you stack `stride=2` blocks, is spatial resolution shrinking too fast?
- For detection/segmentation, is the final feature map still large enough?

## Practical mini-check: catch HWC ↔ CHW confusion fast

Image libraries (OpenCV, PIL, numpy) usually use `(H, W, C)`,
but PyTorch `Conv2d` expects `(B, C, H, W)`.

Add this 20-second check right after your dataloader to catch channel-order bugs early.

```python
import torch

# Assume this came from numpy/PIL: (H, W, C)
x_hwc = torch.randn(224, 224, 3)

# Convert for Conv2d: (C, H, W) -> (B, C, H, W)
x_chw = x_hwc.permute(2, 0, 1).contiguous()
x_bchw = x_chw.unsqueeze(0)

print("HWC :", x_hwc.shape)   # torch.Size([224, 224, 3])
print("BCHW:", x_bchw.shape)  # torch.Size([1, 3, 224, 224])
```

Checkpoint:
- Did you move channels first with `permute(2, 0, 1)`?
- Did you add batch dimension via `unsqueeze(0)`?
- Does model `in_channels` (e.g., 1 or 3) match actual input channels?

## 5-minute experiment: feel stride/padding directly

If "I know the formula but still have no intuition," this quick experiment helps fastest.
Keep the same input and change only `stride/padding`, then compare output shapes.

```python
import torch
import torch.nn as nn

x = torch.randn(1, 3, 32, 32)

settings = [
    {"k": 3, "s": 1, "p": 1},  # same
    {"k": 3, "s": 2, "p": 1},  # downsample
    {"k": 5, "s": 1, "p": 0},  # valid
]

for cfg in settings:
    conv = nn.Conv2d(3, 8, kernel_size=cfg["k"], stride=cfg["s"], padding=cfg["p"])
    y = conv(x)
    print(cfg, "->", tuple(y.shape))
```

Learning checks:
- Can you explain why `stride=2` reduces spatial resolution?
- Can you explain why `padding=0` loses border information?
- Can you predict how aggressive downsampling hurts detection/segmentation quality?

## Beginner completion check (Level 1 bridge)
If you can explain or verify these 5 items, you have passed Conv2d basics.

- [ ] You can read `weight.shape = [C_out, C_in, K, K]` and explain each axis.
- [ ] You can pre-compute output size with $O = \frac{I-K+2P}{S}+1$ before coding.
- [ ] You can perform `HWC -> CHW -> BCHW` conversion and explain why it is required.
- [ ] You can predict how `stride/padding` changes affect classification vs detection/segmentation.
- [ ] You can explain why Local Connectivity + Weight Sharing reduce parameters and improve robustness.

## 10-minute reinforcement drill (to unblock beginners)
This routine is highly effective when learners say: "I know the formula but still hit shape errors."

1. Build only 2 Conv2d layers with random input `(1,3,64,64)` and manually compute output shapes first.
2. Compare manual shapes with actual code output; if mismatched, adjust only `padding/stride`.
3. End with `assert y.shape[2] > 0 and y.shape[3] > 0` to catch spatial collapse early.

The key is to **fix shape math before attaching real training data**.

## Extra production failure pattern: `groups` / depthwise setup

When beginners implement Depthwise Separable Conv, the most common blocker is the `groups` argument.

Core rule:
- standard Conv: `groups=1`
- depthwise Conv: `groups=in_channels`, and `out_channels` is typically a multiple of `in_channels`

```python
import torch
import torch.nn as nn

x = torch.randn(1, 8, 32, 32)

# Correct depthwise example
ok = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8)
print(ok(x).shape)  # torch.Size([1, 8, 32, 32])

# Wrong example: groups does not divide channels
try:
    bad = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=3)
except Exception as e:
    print("error:", e)
```

Quick checks:
- [ ] `in_channels % groups == 0`
- [ ] `out_channels % groups == 0`
- [ ] Are depthwise and pointwise (1×1) stages separated correctly?

## Visual asset prompts (Nanobanana)
- **EN Diagram 1 (Stride/Padding comparison)**  
  "Dark theme background (#1a1a2e), Conv2d stride/padding comparison infographic. Use the same 32x32 input and show three side-by-side panels: (k=3,s=1,p=1), (k=3,s=2,p=1), (k=5,s=1,p=0). In each panel include input grid, filter movement spacing, and output size labels (32x32 / 16x16 / 28x28). Add clean arrows, beginner-friendly labels, modern vector style"

- **EN Diagram 2 (HWC→BCHW conversion)**  
  "Dark theme background (#1a1a2e), tensor-dimension conversion diagram. Show step-by-step flow: HWC (224,224,3) -> permute(2,0,1) -> CHW (3,224,224) -> unsqueeze -> BCHW (1,3,224,224). Emphasize axis names (C/H/W), include warning icon for common channel-order mistakes, clean educational vector style"

## Related Content
- [Pooling](/en/docs/components/convolution/pooling)
- [Receptive Field](/en/docs/components/convolution/receptive-field)
- [Transposed Convolution](/en/docs/components/convolution/transposed-conv)
- [AlexNet](/en/docs/architecture/cnn/alexnet)
