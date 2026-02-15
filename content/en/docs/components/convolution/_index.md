---
title: "Convolution"
weight: 1
bookCollapseSection: true
math: true
---

# Convolution

{{% hint info %}}
**Prerequisites**: [Matrix](/en/docs/math/linear-algebra/matrix)
{{% /hint %}}

> **One-line summary**: Convolution is the core CNN operation that extracts **local patterns** (edges, textures, shapes) by sliding shared filters over an image.

## Why convolution?

If we connect every image pixel directly to a dense layer, parameter count explodes and spatial structure is lost.
Convolution solves this with three ideas:

1. **Parameter sharing**: reuse the same filter across all positions.
2. **Local connectivity**: focus on nearby pixels first.
3. **Translation robustness**: similar patterns can be detected at different locations.

## Learning path

| Order | Topic | Core question | Role in deep learning |
|---|---|---|---|
| 1 | [Conv2D](/en/docs/components/convolution/conv2d) | How does a filter detect patterns? | Basic CNN building block |
| 2 | [Pooling](/en/docs/components/convolution/pooling) | How do we shrink features safely? | Downsampling + robustness |
| 3 | [Receptive Field](/en/docs/components/convolution/receptive-field) | How much input does one output see? | Key design signal |
| 4 | [Transposed Conv](/en/docs/components/convolution/transposed-conv) | How do we upsample learned features? | Decoders, generative models |
| 5 | [Depthwise Separable Conv](/en/docs/components/convolution/depthwise-separable-conv) | How to reduce compute massively? | Mobile/efficient models |
| 6 | [Dilated Conv](/en/docs/components/convolution/dilated-conv) | How to expand view without shrinking maps? | Segmentation |
| 7 | [Deformable Conv](/en/docs/components/convolution/deformable-conv) | How to adapt to irregular shapes? | Detection |
| 8 | [Grouped Conv](/en/docs/components/convolution/grouped-conv) | How to split channels efficiently? | ResNeXt, ShuffleNet |

## Related content

- [Matrix](/en/docs/math/linear-algebra/matrix) — matrix view of convolution
- [Batch Normalization](/en/docs/components/normalization/batch-norm) — typically used after conv
- [AlexNet](/en/docs/architecture/cnn/alexnet) — early deep CNN milestone
- [ResNet](/en/docs/architecture/cnn/resnet) — modern CNN baseline
