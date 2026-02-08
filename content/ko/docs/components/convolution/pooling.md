---
title: "Pooling"
weight: 2
math: true
---

# Pooling

## 개요

Pooling은 공간 차원을 줄이는 다운샘플링 연산으로, 위치 불변성과 연산 효율을 제공합니다.

## Max Pooling

영역 내 최댓값 선택:

$$
y_{i,j} = \max_{m,n \in R_{i,j}} x_{m,n}
$$

```python
import torch
import torch.nn as nn

# 2x2 Max Pooling, stride 2
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.tensor([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=torch.float)

y = pool(x)
# [[[[6, 8],
#    [14, 16]]]]
```

**특징**:
- 가장 강한 특징만 유지
- 작은 이동에 불변
- 학습 파라미터 없음

## Average Pooling

영역 평균값 계산:

$$
y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{m,n \in R_{i,j}} x_{m,n}
$$

```python
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
y = avg_pool(x)
# [[[[3.5, 5.5],
#    [11.5, 13.5]]]]
```

**특징**:
- 모든 정보의 평균 유지
- Max Pooling보다 부드러운 특징

## Global Pooling

전체 공간을 하나의 값으로:

```python
# Global Average Pooling
gap = nn.AdaptiveAvgPool2d(1)
x = torch.randn(32, 512, 7, 7)
y = gap(x)  # (32, 512, 1, 1)
y = y.flatten(1)  # (32, 512)

# 분류 헤드로 직접 연결
classifier = nn.Linear(512, 1000)
logits = classifier(y)
```

**장점**:
- FC layer 대비 파라미터 대폭 감소
- 임의의 입력 크기 처리 가능
- ResNet 이후 표준

## Stride vs Pooling

최근에는 Pooling 대신 stride=2 convolution 사용 추세:

```python
# Pooling 방식 (전통적)
nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1),
    nn.MaxPool2d(2, 2)
)

# Stride 방식 (현대적)
nn.Conv2d(64, 64, 3, stride=2, padding=1)
```

**Stride Convolution 장점**:
- 학습 가능한 다운샘플링
- 정보 손실 감소

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d)
- [Receptive Field](/ko/docs/components/convolution/receptive-field)
