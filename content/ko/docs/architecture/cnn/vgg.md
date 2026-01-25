---
title: "VGG"
weight: 2
math: true
---

# VGG

## 개요

- **논문**: Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)
- **저자**: Karen Simonyan, Andrew Zisserman (Oxford)
- **핵심 기여**: 작은 필터(3×3)를 깊게 쌓는 것이 효과적임을 증명

## 핵심 아이디어

> "3×3 필터를 여러 번 쌓으면 큰 필터와 같은 receptive field를 가지면서 파라미터는 더 적다"

{{< figure src="/images/architecture/cnn/ko/receptive-field.svg" caption="Receptive Field: 3x3 Conv 2번 = 5x5 receptive field, 파라미터는 28% 절약" >}}

| 방식 | Receptive Field | 파라미터 |
|------|-----------------|----------|
| 5×5 한 번 | 5×5 | 25C² |
| 3×3 두 번 | 5×5 | 18C² |

---

## 구조

### VGG-16 아키텍처

{{< figure src="/images/architecture/cnn/vgg-paper-table1.png" caption="Table 1 from paper: VGG 변형별 구성 (Simonyan & Zisserman, 2014)" >}}

```
Input (224×224×3)
    ↓
Block 1: [Conv3-64] × 2 → MaxPool
    ↓
Block 2: [Conv3-128] × 2 → MaxPool
    ↓
Block 3: [Conv3-256] × 3 → MaxPool
    ↓
Block 4: [Conv3-512] × 3 → MaxPool
    ↓
Block 5: [Conv3-512] × 3 → MaxPool
    ↓
FC: 4096 → 4096 → 1000
```

### VGG 변형

| 모델 | 레이어 수 | 파라미터 |
|------|----------|----------|
| VGG-11 | 11 | 133M |
| VGG-13 | 13 | 133M |
| VGG-16 | 16 | 138M |
| VGG-19 | 19 | 144M |

---

## 설계 원칙

### 1. 작은 필터 (3×3)

모든 Conv 레이어에서 3×3 필터 사용:
- 더 많은 비선형성 (ReLU 더 많이 적용)
- 파라미터 효율성
- 더 깊은 네트워크 가능

### 2. 일관된 구조

- 모든 Conv: 3×3, stride 1, padding 1
- 모든 Pool: 2×2, stride 2
- 채널 수: 64 → 128 → 256 → 512 → 512

### 3. 깊이의 중요성

레이어를 깊게 쌓을수록 성능 향상:
- VGG-11: 10.4% top-5 error
- VGG-16: 7.4% top-5 error
- VGG-19: 7.3% top-5 error

---

## 구현 예시

```python
import torch.nn as nn

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)

# VGG-16 config
cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
             512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = make_layers(cfg_vgg16)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

---

## 특징 추출기로서의 VGG

VGG의 중간 레이어는 범용적인 특징 추출기로 활용됩니다:

```python
# Perceptual Loss에서 VGG 특징 사용
vgg = models.vgg16(pretrained=True).features[:16]  # relu3_3까지

def perceptual_loss(pred, target):
    pred_features = vgg(pred)
    target_features = vgg(target)
    return F.mse_loss(pred_features, target_features)
```

**활용 분야:**
- Style Transfer
- Perceptual Loss
- Feature Matching

---

## 한계점

- **파라미터 과다**: 138M 파라미터 (대부분 FC 레이어)
- **메모리 사용량**: 학습 시 많은 GPU 메모리 필요
- **학습 어려움**: 깊은 네트워크에서 gradient vanishing

이러한 한계는 [ResNet](/ko/docs/architecture/cnn/resnet)의 Skip Connection으로 해결되었습니다.

---

## 관련 콘텐츠

- [AlexNet](/ko/docs/architecture/cnn/alexnet) - 이전 모델
- [ResNet](/ko/docs/architecture/cnn/resnet) - Skip Connection으로 깊이 문제 해결
- [CNN 기초](/ko/docs/architecture/cnn) - Convolution의 원리
