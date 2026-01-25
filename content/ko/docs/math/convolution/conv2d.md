---
title: "2D Convolution"
weight: 1
math: true
---

# 2D Convolution

## 개요

2D Convolution은 커널(필터)을 이미지 위에서 슬라이딩하며 지역적 특징을 추출하는 연산입니다.

## 연산 정의

입력 이미지 I와 커널 K의 convolution:

$$
(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)
$$

## 출력 크기 계산

$$
O = \frac{I - K + 2P}{S} + 1
$$

- I: 입력 크기
- K: 커널 크기
- P: 패딩
- S: 스트라이드
- O: 출력 크기

### 예시

입력 32×32, 커널 3×3, 패딩 1, 스트라이드 1:
$$
O = \frac{32 - 3 + 2 \times 1}{1} + 1 = 32
$$

## 구현

```python
import torch
import torch.nn as nn

# 기본 Conv2D
conv = nn.Conv2d(
    in_channels=3,      # 입력 채널 (RGB)
    out_channels=64,    # 출력 채널 (필터 수)
    kernel_size=3,      # 커널 크기
    stride=1,           # 스트라이드
    padding=1           # 패딩
)

x = torch.randn(1, 3, 224, 224)  # (B, C, H, W)
y = conv(x)  # (1, 64, 224, 224)
print(f"Output shape: {y.shape}")
```

## 주요 파라미터

### Kernel Size

- **3×3**: 가장 일반적, 2개 쌓으면 5×5와 동일한 receptive field
- **1×1**: 채널 간 정보 혼합, 차원 축소/확장
- **5×5, 7×7**: 첫 번째 층에서 큰 특징 추출

### Stride

- **1**: 모든 위치에서 연산 (기본)
- **2**: 다운샘플링 효과, pooling 대체 가능

```python
# Stride 2: 출력 크기 절반
conv_downsample = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# 224 -> 112
```

### Padding

- **valid (padding=0)**: 경계 축소
- **same**: 입력/출력 크기 동일 (padding = kernel_size // 2)

### Dilation

커널 요소 사이에 간격 추가:

$$
O = \frac{I - D(K-1) - 1 + 2P}{S} + 1
$$

```python
# Dilated convolution: 넓은 receptive field
conv_dilated = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
# 3x3 커널이 5x5 영역 커버
```

## 파라미터 수 계산

```
파라미터 수 = (K × K × C_in + 1) × C_out
```

- K: 커널 크기
- C_in: 입력 채널
- C_out: 출력 채널
- +1: 편향(bias)

예: Conv2d(64, 128, 3):
```
(3 × 3 × 64 + 1) × 128 = 73,856 파라미터
```

## Depthwise Separable Convolution

파라미터 효율을 위한 분리:

1. **Depthwise**: 채널별로 독립적 convolution
2. **Pointwise**: 1×1 conv로 채널 혼합

```python
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            padding=kernel_size//2, groups=in_ch  # 채널별 독립
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# 파라미터 비교
# 일반: 3×3×64×128 = 73,728
# Separable: 3×3×64 + 64×128 = 576 + 8,192 = 8,768 (약 8.4배 감소)
```

## 관련 콘텐츠

- [Pooling](/ko/docs/math/convolution/pooling)
- [Receptive Field](/ko/docs/math/convolution/receptive-field)
- [Batch Normalization](/ko/docs/math/normalization/batch-norm)
