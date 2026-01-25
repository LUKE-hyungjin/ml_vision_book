---
title: "CNN"
weight: 2
bookCollapseSection: true
math: true
---

# CNN (Convolutional Neural Network)

## 개요

CNN은 이미지 처리에 특화된 신경망 구조로, 2012년 AlexNet의 등장 이후 컴퓨터 비전의 핵심이 되었습니다.

## 핵심 구성 요소

### 1. Convolution Layer

필터(커널)를 이미지에 슬라이딩하며 특징을 추출합니다.

$$\text{Output}(i,j) = \sum_{m}\sum_{n} \text{Input}(i+m, j+n) \cdot \text{Kernel}(m,n)$$

**특징:**
- 파라미터 공유로 효율적
- 지역적 패턴 학습
- Translation equivariance

### 2. Pooling Layer

공간 해상도를 줄이고 위치 불변성을 제공합니다.

- **Max Pooling**: 영역 내 최댓값 선택
- **Average Pooling**: 영역 내 평균값 계산

### 3. Activation Function

비선형성을 추가합니다.

$$\text{ReLU}(x) = \max(0, x)$$

### 4. Fully Connected Layer

최종 분류를 위한 전결합 층입니다.

---

## 전형적인 CNN 구조

```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

---

## 주요 발전 과정

| 연도 | 모델 | 핵심 기여 |
|------|------|----------|
| 2012 | [AlexNet](/ko/docs/architecture/cnn/alexnet) | GPU 학습, ReLU, Dropout |
| 2014 | [VGG](/ko/docs/architecture/cnn/vgg) | 깊은 네트워크, 3x3 필터 |
| 2015 | [ResNet](/ko/docs/architecture/cnn/resnet) | Skip Connection, 152층 |

---

## 구현 예시

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 관련 콘텐츠

- [Convolution](/ko/docs/math/convolution) - Convolution 연산의 수학적 이해
- [Backpropagation](/ko/docs/math/backpropagation) - CNN 학습 원리
- [Classification](/ko/docs/task/classification) - CNN의 대표적 응용
