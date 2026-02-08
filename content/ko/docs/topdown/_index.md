---
title: "Top-Down"
weight: 2
bookCollapseSection: true
---

# Top-Down 학습

문제를 정하고, 필요한 지식을 찾아갑니다.

{{< mermaid >}}
flowchart TD
    subgraph Problem["🎯 문제 선택"]
        P1[Classification]
        P2[Detection]
        P3[Generation]
        P4[VLM]
    end

    subgraph Architecture["🏗️ 아키텍처"]
        A1[ResNet]
        A2[YOLO]
        A3[Stable Diffusion]
        A4[CLIP]
    end

    subgraph Math["📐 수학 기초"]
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
1. [Cross-entropy Loss](/ko/docs/components/training/loss/cross-entropy)
2. [CNN 기초](/ko/docs/architecture/cnn-basics)
3. [ResNet](/ko/docs/architecture/resnet)
4. [Classification](/ko/docs/task/classification)

## Object Detection
1. [IoU](/ko/docs/components/detection/iou) & [NMS](/ko/docs/components/detection/nms)
2. [Anchor Box](/ko/docs/components/detection/anchor)
3. [Faster R-CNN](/ko/docs/architecture/faster-rcnn)
4. [YOLO](/ko/docs/architecture/yolo)
5. [Detection](/ko/docs/task/detection)

## Segmentation
1. [Transposed Convolution](/ko/docs/components/convolution/transposed-conv)
2. [U-Net](/ko/docs/architecture/unet)
3. [Mask R-CNN](/ko/docs/architecture/mask-rcnn)
4. [Segmentation](/ko/docs/task/segmentation)

## Image Generation
1. [확률분포](/ko/docs/math/probability)
2. [VAE](/ko/docs/architecture/vae)
3. [GAN](/ko/docs/architecture/gan)
4. [Diffusion 수학](/ko/docs/components/generative)
5. [Stable Diffusion](/ko/docs/architecture/stable-diffusion)
6. [ControlNet](/ko/docs/architecture/controlnet)
7. [DiT](/ko/docs/architecture/dit)
8. [Generation](/ko/docs/task/generation)

## Vision-Language (VLM)
1. [Contrastive Learning](/ko/docs/math/contrastive)
2. [CLIP](/ko/docs/architecture/clip)
3. [VLM](/ko/docs/architecture/vlm)
4. [Vision-Language](/ko/docs/task/vision-language)

## 3D Vision
1. [카메라 모델](/ko/docs/math/geometry)
2. [NeRF](/ko/docs/architecture/nerf)
3. [3D Vision](/ko/docs/task/3d-vision)

## 모델 배포
1. [Quantization](/ko/docs/components/quantization)
2. [Deployment](/ko/docs/task/deployment)
