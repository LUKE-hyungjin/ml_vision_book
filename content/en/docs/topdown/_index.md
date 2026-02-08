---
title: "Top-Down"
weight: 2
bookCollapseSection: true
---

# Top-Down Learning

Start from problems and find the knowledge you need.

{{< mermaid >}}
flowchart TD
    subgraph Problem["🎯 Choose Problem"]
        P1[Classification]
        P2[Detection]
        P3[Generation]
        P4[VLM]
    end

    subgraph Architecture["🏗️ Architecture"]
        A1[ResNet]
        A2[YOLO]
        A3[Stable Diffusion]
        A4[CLIP]
    end

    subgraph Math["📐 Math Foundations"]
        M1[Loss Function]
        M2[IoU & NMS]
        M3[Diffusion]
        M4[Contrastive]
    end

    P1 --> A1 --> M1
    P2 --> A2 --> M2
    P3 --> A3 --> M3
    P4 --> A4 --> M4
{{< /mermaid >}}

## Image Classification
1. [Cross-entropy Loss](/en/docs/components/training/loss/cross-entropy)
2. [CNN Basics](/docs/architecture/cnn-basics)
3. [ResNet](/docs/architecture/resnet)
4. [Classification](/docs/task/classification)

## Object Detection
1. [IoU](/en/docs/components/detection/iou) & [NMS](/en/docs/components/detection/nms)
2. [Anchor Box](/en/docs/components/detection/anchor)
3. [Faster R-CNN](/docs/architecture/faster-rcnn)
4. [YOLO](/docs/architecture/yolo)
5. [Detection](/docs/task/detection)

## Segmentation
1. [Transposed Convolution](/en/docs/components/convolution/transposed-conv)
2. [U-Net](/docs/architecture/unet)
3. [Mask R-CNN](/docs/architecture/mask-rcnn)
4. [Segmentation](/docs/task/segmentation)

## Image Generation
1. [Probability Distributions](/docs/math/probability)
2. [VAE](/docs/architecture/vae)
3. [GAN](/docs/architecture/gan)
4. [Diffusion Math](/en/docs/components/generative)
5. [Stable Diffusion](/docs/architecture/stable-diffusion)
6. [ControlNet](/docs/architecture/controlnet)
7. [DiT](/docs/architecture/dit)
8. [Generation](/docs/task/generation)

## Vision-Language (VLM)
1. [Contrastive Learning](/docs/math/contrastive)
2. [CLIP](/docs/architecture/clip)
3. [VLM](/docs/architecture/vlm)
4. [Vision-Language](/docs/task/vision-language)

## 3D Vision
1. [Camera Model](/docs/math/geometry)
2. [NeRF](/docs/architecture/nerf)
3. [3D Vision](/docs/task/3d-vision)

## Model Deployment
1. [Quantization](/en/docs/components/quantization)
2. [Deployment](/docs/task/deployment)
