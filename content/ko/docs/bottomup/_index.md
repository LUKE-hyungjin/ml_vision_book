---
title: "Bottom-Up"
weight: 3
bookCollapseSection: true
---

# Bottom-Up 학습

기초 개념부터 차근차근 쌓아올립니다.

{{< mermaid >}}
flowchart BT
    subgraph L1["Level 1: 수학 기초"]
        M1[선형대수]
        M2[미적분]
        M3[확률/통계]
    end

    subgraph L2["Level 2: 딥러닝 기초"]
        D1[Convolution]
        D2[Backprop]
        D3[Loss & Optim]
    end

    subgraph L3["Level 3: 기본 아키텍처"]
        A1[CNN]
        A2[ResNet]
    end

    subgraph L4["Level 4: 기본 태스크"]
        T1[Classification]
        T2[Detection]
        T3[Segmentation]
    end

    subgraph L5["Level 5+: 고급"]
        H1[Transformer]
        H2[Diffusion]
        H3[VLM]
    end

    L1 --> L2 --> L3 --> L4 --> L5
{{< /mermaid >}}

## Level 1: 수학 기초
- [선형대수](/ko/docs/math/linear-algebra)
- [미적분 & Chain Rule](/ko/docs/math/calculus)
- [확률/통계](/ko/docs/math/probability)

## Level 2: 딥러닝 기초
- [Convolution](/ko/docs/math/convolution)
- [Backpropagation](/ko/docs/math/backpropagation)
- [Loss Functions](/ko/docs/math/loss-functions)
- [Optimization](/ko/docs/math/optimization)

## Level 3: 기본 아키텍처
- [CNN 기초](/ko/docs/architecture/cnn-basics)
- [AlexNet](/ko/docs/architecture/alexnet)
- [VGG](/ko/docs/architecture/vgg)
- [ResNet](/ko/docs/architecture/resnet)

## Level 4: 기본 태스크
- [Classification](/ko/docs/task/classification)
- [Detection](/ko/docs/task/detection)
- [Segmentation](/ko/docs/task/segmentation)

## Level 5: 고급 개념
- [Attention](/ko/docs/math/attention)
- [Transformer](/ko/docs/architecture/transformer)
- [ViT](/ko/docs/architecture/vit)

## Level 6: 생성 모델
- [Diffusion 수학](/ko/docs/math/generative)
- [Stable Diffusion](/ko/docs/architecture/stable-diffusion)
- [ControlNet](/ko/docs/architecture/controlnet)
- [DiT](/ko/docs/architecture/dit)
- [Generation](/ko/docs/task/generation)

## Level 7: Multimodal & 3D
- [Contrastive Learning](/ko/docs/math/contrastive)
- [CLIP](/ko/docs/architecture/clip)
- [VLM](/ko/docs/architecture/vlm)
- [Vision-Language](/ko/docs/task/vision-language)
- [NeRF](/ko/docs/architecture/nerf)
- [3D Vision](/ko/docs/task/3d-vision)
