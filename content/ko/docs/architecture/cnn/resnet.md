---
title: "ResNet"
weight: 3
math: true
---

# ResNet (Residual Network)

## 개요

- **논문**: Deep Residual Learning for Image Recognition (2015)
- **저자**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- **핵심 기여**: Skip Connection으로 100+ 레이어 학습 가능하게 함

## 해결한 문제: Degradation Problem

네트워크가 깊어지면 성능이 오히려 떨어지는 현상:

```
20-layer: 7.4% error
56-layer: 8.5% error  ← 더 깊은데 성능이 나빠짐
```

이는 과적합이 아니라 **최적화의 어려움** 때문입니다.

---

## 핵심 아이디어: Residual Learning

### Skip Connection (Shortcut)

기존 학습:
$$H(x) = F(x)$$

Residual 학습:
$$H(x) = F(x) + x$$

여기서 $F(x)$는 **잔차(residual)**를 학습합니다.

### 직관

- 입력을 그대로 출력에 더함
- 네트워크는 "변화량"만 학습
- Identity mapping이 쉬워짐 → 깊은 네트워크 학습 가능

---

## 구조

### Residual Block 비교

{{< figure src="/images/architecture/cnn/resnet-paper-fig2.png" caption="Figure 2 from paper: Residual learning building block (He et al., 2015)" >}}

### Basic Block (ResNet-18/34)

```
x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+x) → ReLU
    └─────────────────────────────────────┘
                 skip connection
```

### Bottleneck Block (ResNet-50/101/152)

```
x → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+x) → ReLU
    └────────────────────────────────────────────────────────────┘
                              skip connection
```

1×1 Conv로 채널을 줄였다가 늘려 계산량 감소.

### ResNet 변형

| 모델 | 레이어 | 파라미터 | Top-5 Error |
|------|--------|----------|-------------|
| ResNet-18 | 18 | 11.7M | 10.92% |
| ResNet-34 | 34 | 21.8M | 9.46% |
| ResNet-50 | 50 | 25.6M | 7.13% |
| ResNet-101 | 101 | 44.5M | 6.44% |
| ResNet-152 | 152 | 60.2M | 6.16% |

---

## 전체 아키텍처 (ResNet-50)

{{< figure src="/images/architecture/cnn/resnet-paper-fig3.png" caption="Figure 3 from paper: VGG-19 vs 34-layer plain vs 34-layer residual (He et al., 2015)" >}}

```
Input (224×224×3)
    ↓
Conv 7×7, 64, stride 2
    ↓
MaxPool 3×3, stride 2
    ↓
Stage 1: [Bottleneck(64, 256)] × 3
    ↓
Stage 2: [Bottleneck(128, 512)] × 4
    ↓
Stage 3: [Bottleneck(256, 1024)] × 6
    ↓
Stage 4: [Bottleneck(512, 2048)] × 3
    ↓
Global Average Pool
    ↓
FC → 1000
```

---

## 구현 예시

```python
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection (차원 맞추기)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)

        return out
```

---

## ResNet의 영향

### Backbone으로 널리 사용

- Object Detection: [Faster R-CNN](/ko/docs/architecture/detection/faster-rcnn)
- Segmentation: [Mask R-CNN](/ko/docs/architecture/segmentation/mask-rcnn)
- 대부분의 Vision 태스크에서 기본 backbone

### 후속 연구

- **ResNeXt**: 그룹 컨볼루션 추가
- **DenseNet**: 모든 레이어를 연결
- **SE-Net**: 채널 attention 추가
- **EfficientNet**: 최적 스케일링

---

## 왜 Skip Connection이 효과적인가?

{{< figure src="/images/architecture/cnn/ko/skip-connection.svg" caption="Skip Connection: Gradient가 shortcut을 통해 직접 흐르므로 vanishing 문제 해결" >}}

### 1. Gradient Flow 개선

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \frac{\partial H}{\partial x} = \frac{\partial L}{\partial H} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

항상 1이 더해지므로 gradient가 소실되지 않습니다.

### 2. Ensemble 효과

Skip connection은 다양한 깊이의 경로를 만들어 앙상블과 유사한 효과를 냅니다.

---

## 관련 콘텐츠

- [VGG](/ko/docs/architecture/cnn/vgg) - 깊이의 한계를 보여준 모델
- [CNN 기초](/ko/docs/architecture/cnn) - Convolution의 원리
- [Detection](/ko/docs/task/detection) - ResNet을 backbone으로 사용
- [ViT](/ko/docs/architecture/transformer/vit) - ResNet을 대체하는 새로운 패러다임
