---
title: "타임라인"
weight: 1
bookCollapseSection: true
---

# 타임라인으로 배우는 Vision

연대순으로 Computer Vision의 발전 과정을 따라가며 학습합니다.

{{< mermaid >}}
flowchart LR
    subgraph Classical["~2012"]
        A[Classical CV]
    end
    subgraph CNN["2012-2017"]
        B[CNN 혁명]
        C[Detection]
    end
    subgraph Attention["2017-2021"]
        D[Transformer]
        E[ViT & CLIP]
    end
    subgraph Gen["2021-현재"]
        F[Diffusion]
        G[VLM & 3D]
    end

    A --> B --> C --> D --> E --> F --> G
{{< /mermaid >}}

## ~2012: Classical Computer Vision
- [선형대수](/ko/docs/math/linear-algebra)
- [기하학](/ko/docs/math/geometry)
- [SIFT & HOG](/ko/docs/architecture/classical-features)

## 2012-2015: CNN 시대의 시작
- [Convolution](/ko/docs/math/convolution)
- [Backpropagation](/ko/docs/math/backpropagation)
- [AlexNet](/ko/docs/architecture/alexnet)
- [VGG](/ko/docs/architecture/vgg)
- [ResNet](/ko/docs/architecture/resnet)
- [Classification](/ko/docs/task/classification)

## 2015-2017: Detection & Segmentation
- [IoU & NMS](/ko/docs/math/iou-nms)
- [YOLO](/ko/docs/architecture/yolo)
- [Faster R-CNN](/ko/docs/architecture/faster-rcnn)
- [U-Net](/ko/docs/architecture/unet)
- [Detection](/ko/docs/task/detection)
- [Segmentation](/ko/docs/task/segmentation)

## 2017-2019: Attention의 등장
- [Attention](/ko/docs/math/attention)
- [Transformer](/ko/docs/architecture/transformer)

## 2020-2021: Vision Transformer & CLIP
- [ViT](/ko/docs/architecture/vit)
- [CLIP](/ko/docs/architecture/clip)
- [Contrastive Learning](/ko/docs/math/contrastive)
- [Self-supervised Learning](/ko/docs/task/self-supervised)

## 2021-2022: Diffusion 시대
- [Diffusion Process](/ko/docs/math/diffusion-process)
- [Stable Diffusion](/ko/docs/architecture/stable-diffusion)
- [Generation](/ko/docs/task/generation)

## 2023: Controllable Generation & SAM
- [ControlNet](/ko/docs/architecture/controlnet)
- [SAM](/ko/docs/architecture/sam)

## 2023-2024: VLM & DiT
- [VLM](/ko/docs/architecture/vlm)
- [DiT](/ko/docs/architecture/dit)
- [Vision-Language](/ko/docs/task/vision-language)

## 2024-현재: 3D & Video Generation
- [NeRF](/ko/docs/architecture/nerf)
- [3D Gaussian Splatting](/ko/docs/architecture/3dgs)
- [3D Vision](/ko/docs/task/3d-vision)
