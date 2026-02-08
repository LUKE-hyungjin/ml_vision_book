---
title: "Receptive Field"
weight: 3
math: true
---

# Receptive Field (수용 영역)

## 개요

Receptive Field는 출력의 한 픽셀이 "볼 수 있는" 입력 영역의 크기입니다. 넓을수록 더 많은 컨텍스트를 고려합니다.

## 계산

L개 층의 receptive field:

$$
RF_L = RF_{L-1} + (K_L - 1) \times \prod_{i=1}^{L-1} S_i
$$

또는 재귀적으로:

$$
RF_L = 1 + \sum_{i=1}^{L} (K_i - 1) \times \prod_{j=1}^{i-1} S_j
$$

- K: 커널 크기
- S: 스트라이드

## 예시: VGG-style

```
Layer 1: 3×3 conv, stride 1 → RF = 3
Layer 2: 3×3 conv, stride 1 → RF = 3 + (3-1)×1 = 5
Layer 3: 3×3 conv, stride 1 → RF = 5 + (3-1)×1 = 7
Pooling: 2×2, stride 2       → RF = 7 + (2-1)×1 = 8
Layer 4: 3×3 conv, stride 1 → RF = 8 + (3-1)×2 = 12
```

## 코드로 계산

```python
def calc_receptive_field(layers):
    """
    layers: list of (kernel_size, stride)
    """
    rf = 1
    total_stride = 1

    for k, s in layers:
        rf = rf + (k - 1) * total_stride
        total_stride *= s

    return rf

# VGG 스타일 (3x3 conv 여러 개)
layers = [
    (3, 1), (3, 1),         # Conv block 1
    (2, 2),                  # Pool
    (3, 1), (3, 1),         # Conv block 2
    (2, 2),                  # Pool
    (3, 1), (3, 1), (3, 1), # Conv block 3
]
print(f"RF: {calc_receptive_field(layers)}")  # RF: 44
```

## Receptive Field 확장 전략

### 1. 더 많은 층 쌓기

3×3 conv 2개 = 5×5 영역, 파라미터는 적음

```python
# 5x5 conv: 25 × C² 파라미터
# 3x3 × 2: 18 × C² 파라미터 (더 적음)
nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),  # 5x5 효과
)
```

### 2. Dilated Convolution

간격을 두어 RF 확장:

```python
# dilation=2: 3x3 → 5x5 영역 커버
conv = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
```

Dilation과 RF:
- 3×3, dilation=1: RF = 3
- 3×3, dilation=2: RF = 5
- 3×3, dilation=4: RF = 9

### 3. Pooling / Stride

다운샘플링 후 같은 커널이 더 넓은 영역 커버

## Effective Receptive Field

실제로 출력에 영향을 미치는 영역은 이론적 RF보다 작습니다:

- 가장자리보다 중앙에 더 집중
- 가우시안 형태의 영향력 분포

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d)
- [Pooling](/ko/docs/components/convolution/pooling)
- [Attention](/ko/docs/components/attention) - 전역 RF
