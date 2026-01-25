---
title: "Timeline"
weight: 1
bookCollapseSection: true
---

# Learning Vision by Timeline

Follow the chronological evolution of Computer Vision technology.

{{< mermaid >}}
flowchart LR
    subgraph Classical["~2012"]
        A[Classical CV]
    end
    subgraph CNN["2012-2017"]
        B[CNN Revolution]
        C[Detection]
    end
    subgraph Attention["2017-2021"]
        D[Transformer]
        E[ViT & CLIP]
    end
    subgraph Gen["2021-Present"]
        F[Diffusion]
        G[VLM & 3D]
    end

    A --> B --> C --> D --> E --> F --> G
{{< /mermaid >}}

## ~2012: Classical Computer Vision
- [Linear Algebra](/docs/math/linear-algebra)
- [Geometry](/docs/math/geometry)
- [SIFT & HOG](/docs/architecture/classical-features)

## 2012-2015: The Rise of CNN
- [Convolution](/docs/math/convolution)
- [Backpropagation](/docs/math/backpropagation)
- [AlexNet](/docs/architecture/alexnet)
- [VGG](/docs/architecture/vgg)
- [ResNet](/docs/architecture/resnet)
- [Classification](/docs/task/classification)

## 2015-2017: Detection & Segmentation
- [IoU & NMS](/docs/math/iou-nms)
- [YOLO](/docs/architecture/yolo)
- [Faster R-CNN](/docs/architecture/faster-rcnn)
- [U-Net](/docs/architecture/unet)
- [Detection](/docs/task/detection)
- [Segmentation](/docs/task/segmentation)

## 2017-2019: Emergence of Attention
- [Attention](/docs/math/attention)
- [Transformer](/docs/architecture/transformer)

## 2020-2021: Vision Transformer & CLIP
- [ViT](/docs/architecture/vit)
- [CLIP](/docs/architecture/clip)
- [Contrastive Learning](/docs/math/contrastive)
- [Self-supervised Learning](/docs/task/self-supervised)

## 2021-2022: The Diffusion Era
- [Diffusion Process](/docs/math/diffusion-process)
- [Stable Diffusion](/docs/architecture/stable-diffusion)
- [Generation](/docs/task/generation)

## 2023: Controllable Generation & SAM
- [ControlNet](/docs/architecture/controlnet)
- [SAM](/docs/architecture/sam)

## 2023-2024: VLM & DiT
- [VLM](/docs/architecture/vlm)
- [DiT](/docs/architecture/dit)
- [Vision-Language](/docs/task/vision-language)

## 2024-Present: 3D & Video Generation
- [NeRF](/docs/architecture/nerf)
- [3D Gaussian Splatting](/docs/architecture/3dgs)
- [3D Vision](/docs/task/3d-vision)
