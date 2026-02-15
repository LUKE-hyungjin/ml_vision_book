---
title: "U-Net"
weight: 1
math: true
---

# U-Net

{{% hint info %}}
**Prerequisites**: [Conv2D](/en/docs/components/convolution/conv2d) | [Transposed Convolution](/en/docs/components/convolution/transposed-conv)
{{% /hint %}}

## One-line Summary
> **U-Net combines high-resolution localization cues (encoder skips) with semantic decoding, enabling strong pixel-level segmentation even with limited data.**

## Why this model?
Beginners usually hit these segmentation pain points first:

1. Class prediction is okay, but **boundaries are blurry**
2. Downsampling + upsampling causes **loss of spatial detail**
3. With small datasets, overfitting appears quickly

U-Net addresses this by combining compressed semantic features from the encoder with precise spatial features via skip connections.
Analogy: first read a compressed map to understand the area (encoder), then overlay detailed street-level notes from the original map (skip) to recover exact boundaries (decoder).

## Overview

- **Paper**: U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
- **Authors**: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **Key contribution**: Precise segmentation with Encoder-Decoder + Skip Connections

## Core Idea

> "Use the contracting path for context and the expanding path for localization."

It works especially well in domains like medical imaging where labeled data is limited.

---

## Architecture

### Overall Architecture

```
Input (572×572)
    ↓
┌───────────────────────────────────────────────────────────┐
│                    Contracting Path                        │
│  [Conv3×3 → ReLU → Conv3×3 → ReLU → MaxPool] × 4          │
│   64 → 128 → 256 → 512 → 1024                              │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│                    Expanding Path                          │
│  [UpConv2×2 → Concat(skip) → Conv3×3 → ReLU × 2] × 4      │
│   1024 → 512 → 256 → 128 → 64                              │
└───────────────────────────────────────────────────────────┘
    ↓
Conv 1×1 → Output (388×388×num_classes)
```

### U-shape intuition

```
Input                                           Output
  ↓                                               ↑
[Conv]─────────────────────────────────────→[UpConv+Concat]
  ↓                                               ↑
[Conv]───────────────────────────────→[UpConv+Concat]
  ↓                                         ↑
[Conv]─────────────────────────→[UpConv+Concat]
  ↓                               ↑
[Conv]──────────────────→[UpConv+Concat]
  ↓                    ↑
     [Bottleneck]
```

---

## Key Components

### 1. Contracting Path (Encoder)

- 3×3 Conv + ReLU × 2
- 2×2 Max Pooling (stride 2)
- Channel count doubles at each stage

### 2. Expanding Path (Decoder)

- 2×2 up-convolution (halve channels)
- Concatenate skip features from encoder
- 3×3 Conv + ReLU × 2

### 3. Skip Connection

Pass high-resolution encoder features to the decoder:
- preserve localization information
- recover fine boundaries

---

## Implementation Example

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + Skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)
```

---

## Training

### Loss Function

Binary Segmentation:
$$L = \text{BCE}(y, \hat{y}) + \text{Dice Loss}$$

Multi-class:
$$L = \text{CrossEntropy}(y, \hat{y})$$

**Symbol meanings:**
- $y$: ground-truth mask
- $\hat{y}$: predicted mask
- $L$: total loss to minimize

Beginner takeaway:
- BCE/CrossEntropy teaches pixel-wise correctness
- Dice compensates for class imbalance and improves boundary quality

### Data Augmentation

Medical segmentation often has limited labels, so augmentation is important:
- Elastic deformation
- Rotation / flipping
- Grayscale transforms

## Practical Debugging Checklist
- [ ] **Input/output resolution check**: does prediction mask size exactly match GT mask size?
- [ ] **Skip concat shape check**: before `torch.cat`, are H/W aligned (crop/interpolate if needed)?
- [ ] **Loss-activation consistency**: if using BCEWithLogitsLoss, avoid applying sigmoid twice
- [ ] **Class imbalance handling**: if foreground is sparse, did you add Dice/Focal-style terms?
- [ ] **Augmentation strength sanity check**: are heavy transforms breaking mask-label alignment?

## Common mistakes (FAQ)
**Q1. Is U-Net only for tiny datasets?**  
A. No. It is especially strong on small datasets, but still widely used as a robust baseline in industry.

**Q2. Is upsampling alone enough in the decoder?**  
A. Usually no. Without skip connections, boundary recovery often degrades.

**Q3. Training loss decreases, but mask boundaries look jagged. Why?**  
A. Check upsampling artifacts, class imbalance, and weak high-resolution feature transfer first.

## Top 3 beginner bottlenecks (quick fixes)

| Bottleneck | Common cause | First action |
|---|---|---|
| `torch.cat` shape error in skip path | input size not aligned with down/up-sampling stages | align H/W before concat via `F.interpolate` or center-crop |
| almost-empty predicted masks | wrong sigmoid/softmax/threshold pipeline | for binary, use `BCEWithLogitsLoss` and apply sigmoid exactly once at inference |
| Dice improves but IoU stalls | boundary errors remain while interior pixels are correct | increase small-object samples and inspect boundary-focused augmentation |

## Loss ↔ activation matching table (common production confusion)

| Task type | Output channels | Training loss | Inference activation | Note |
|---|---:|---|---|---|
| Binary segmentation | 1 | `BCEWithLogitsLoss` | `sigmoid` | do not apply sigmoid twice during training |
| Multi-class segmentation | C | `CrossEntropyLoss` | `softmax` + `argmax` | GT is usually class-index mask |
| Binary with severe imbalance | 1 | `BCEWithLogitsLoss + Dice` | `sigmoid` | helps small foreground objects |

## Mini validation code: shape + probability-range sanity check

```python
import torch

model = UNet(in_channels=3, out_channels=1).eval()
x = torch.randn(2, 3, 256, 256)

with torch.no_grad():
    logits = model(x)                  # (B, 1, H, W)
    probs = torch.sigmoid(logits)      # [0, 1]

print("logits shape:", logits.shape)
print("prob range:", float(probs.min()), float(probs.max()))

assert logits.shape == (2, 1, 256, 256)
assert probs.min() >= 0.0 and probs.max() <= 1.0
```

Passing this check early usually saves a lot of debugging time (especially for shape/activation bugs).

---

## U-Net Variants

| Variant | Key idea |
|------|------|
| **U-Net++** | Nested skip connections |
| **Attention U-Net** | Adds attention gates |
| **ResUNet** | Uses ResNet-style blocks |
| **TransUNet** | Uses a Transformer encoder |

---

## Applications

- **Medical imaging**: cell, tumor, organ segmentation
- **Satellite imagery**: building/road extraction
- **Autonomous driving**: road-area segmentation
- **Generative models**: influences structures like [Stable Diffusion](/en/docs/architecture/generative/stable-diffusion)

---

## Related Content

- [Transposed Convolution](/en/docs/components/convolution/transposed-conv) - Upsampling principle
- [ResNet](/en/docs/architecture/cnn/resnet) - Skip connection idea
- [Mask R-CNN](/en/docs/architecture/segmentation/mask-rcnn) - Instance segmentation
- [Segmentation Task](/en/docs/task/segmentation) - Evaluation metrics
