---
title: "Bottom-Up"
weight: 3
bookCollapseSection: true
---

# Bottom-Up Learning

Build up from fundamentals step by step.

{{< mermaid >}}
flowchart BT
    subgraph L1["Level 1: Math Basics"]
        M1[Linear Algebra]
        M2[Calculus]
        M3[Probability]
    end

    subgraph L2["Level 2: DL Basics"]
        D1[Convolution]
        D2[Backprop]
        D3[Loss & Optim]
    end

    subgraph L3["Level 3: Basic Arch"]
        A1[CNN]
        A2[ResNet]
    end

    subgraph L4["Level 4: Basic Tasks"]
        T1[Classification]
        T2[Detection]
        T3[Segmentation]
    end

    subgraph L5["Level 5+: Advanced"]
        H1[Transformer]
        H2[Diffusion]
        H3[VLM]
    end

    L1 --> L2 --> L3 --> L4 --> L5
{{< /mermaid >}}

## Level 1: Math Fundamentals
- [Linear Algebra](/docs/math/linear-algebra)
- [Calculus & Chain Rule](/docs/math/calculus)
- [Probability & Statistics](/docs/math/probability)

## Level 2: Deep Learning Basics
- [Convolution](/docs/math/convolution)
- [Backpropagation](/docs/math/backpropagation)
- [Loss Functions](/docs/math/loss-functions)
- [Optimization](/docs/math/optimization)

## Level 3: Basic Architectures
- [CNN Basics](/docs/architecture/cnn-basics)
- [AlexNet](/docs/architecture/alexnet)
- [VGG](/docs/architecture/vgg)
- [ResNet](/docs/architecture/resnet)

## Level 4: Basic Tasks
- [Classification](/docs/task/classification)
- [Detection](/docs/task/detection)
- [Segmentation](/docs/task/segmentation)

## Level 5: Advanced Concepts
- [Attention](/docs/math/attention)
- [Transformer](/docs/architecture/transformer)
- [ViT](/docs/architecture/vit)

## Level 6: Generative Models
- [Diffusion Math](/en/docs/math/generative)
- [Stable Diffusion](/docs/architecture/stable-diffusion)
- [ControlNet](/docs/architecture/controlnet)
- [DiT](/docs/architecture/dit)
- [Generation](/docs/task/generation)

## Level 7: Multimodal & 3D
- [Contrastive Learning](/docs/math/contrastive)
- [CLIP](/docs/architecture/clip)
- [VLM](/docs/architecture/vlm)
- [Vision-Language](/docs/task/vision-language)
- [NeRF](/docs/architecture/nerf)
- [3D Vision](/docs/task/3d-vision)
