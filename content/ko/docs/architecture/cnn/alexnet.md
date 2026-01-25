---
title: "AlexNet"
weight: 1
math: true
---

# AlexNet

## 개요

- **논문**: ImageNet Classification with Deep Convolutional Neural Networks (2012)
- **저자**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **핵심 기여**: 딥러닝 시대의 시작을 알린 모델. ImageNet 2012에서 압도적 성능으로 우승

## 왜 중요한가?

AlexNet은 ImageNet에서 top-5 error를 **26% → 16%**로 크게 낮추며 딥러닝의 가능성을 증명했습니다. 이후 모든 컴퓨터 비전 연구가 CNN 기반으로 전환되는 계기가 되었습니다.

---

## 구조

### 전체 아키텍처

{{< figure src="/images/architecture/cnn/alexnet-paper-fig2.png" caption="Figure 2 from paper: 2개의 GPU에 나눠서 학습하는 구조 (Krizhevsky et al., 2012)" >}}

```
Input (224×224×3)
    ↓
Conv1: 96 filters, 11×11, stride 4 → ReLU → LRN → MaxPool
    ↓
Conv2: 256 filters, 5×5 → ReLU → LRN → MaxPool
    ↓
Conv3: 384 filters, 3×3 → ReLU
    ↓
Conv4: 384 filters, 3×3 → ReLU
    ↓
Conv5: 256 filters, 3×3 → ReLU → MaxPool
    ↓
FC6: 4096 → ReLU → Dropout
    ↓
FC7: 4096 → ReLU → Dropout
    ↓
FC8: 1000 (softmax)
```

### 주요 특징

| 요소 | 설명 |
|------|------|
| **ReLU** | Sigmoid 대신 사용하여 학습 속도 6배 향상 |
| **GPU 학습** | 2개의 GTX 580 GPU로 병렬 학습 |
| **Dropout** | FC 레이어에 0.5 비율로 적용하여 과적합 방지 |
| **LRN** | Local Response Normalization (현재는 사용 안 함) |
| **Data Augmentation** | 랜덤 크롭, 수평 뒤집기, 색상 변환 |

---

## 핵심 기술 상세

### 1. ReLU (Rectified Linear Unit)

$$f(x) = \max(0, x)$$

**장점:**
- Sigmoid/tanh보다 계산이 단순
- Gradient vanishing 문제 완화
- 학습 속도 대폭 향상

### 2. Dropout

학습 시 뉴런을 랜덤하게 비활성화하여 과적합 방지:

$$\hat{y} = \frac{1}{1-p} \cdot y \cdot \text{mask}$$

### 3. Data Augmentation

원본 256×256 이미지에서:
- 224×224 랜덤 크롭
- 수평 뒤집기
- PCA 기반 색상 변환

---

## 파라미터 수

| 레이어 | 파라미터 |
|--------|----------|
| Conv 레이어 | ~2.3M |
| FC 레이어 | ~58.6M |
| **총합** | **~60M** |

대부분의 파라미터가 FC 레이어에 집중되어 있습니다.

---

## 구현 예시

```python
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
```

---

## 한계점

- 큰 필터 크기 (11×11, 5×5) → 비효율적
- LRN은 효과 미미 → 이후 사용 안 함
- FC 레이어에 파라미터 집중 → 메모리 비효율

이러한 한계는 [VGG](/ko/docs/architecture/cnn/vgg)와 [ResNet](/ko/docs/architecture/cnn/resnet)에서 개선되었습니다.

---

## 관련 콘텐츠

- [CNN 기초](/ko/docs/architecture/cnn) - CNN의 기본 구조
- [VGG](/ko/docs/architecture/cnn/vgg) - 더 깊은 네트워크
- [ResNet](/ko/docs/architecture/cnn/resnet) - Skip Connection
- [Classification](/ko/docs/task/classification) - 이미지 분류 태스크
