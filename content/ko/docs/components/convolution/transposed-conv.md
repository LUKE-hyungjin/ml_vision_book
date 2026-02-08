---
title: "Transposed Convolution"
weight: 4
math: true
---

# Transposed Convolution

## 개요

Transposed Convolution (또는 Deconvolution)은 공간 해상도를 높이는 업샘플링 연산입니다. 디코더와 생성 모델에서 필수적입니다.

## 동작 원리

일반 convolution의 역방향 연산:
- Conv: 입력 → 작은 출력
- Transposed Conv: 입력 → 큰 출력

**주의**: 수학적 역연산이 아님. 원본 복원 X, 크기만 복원.

## 출력 크기 계산

$$
O = (I - 1) \times S - 2P + K + P_{out}
$$

- I: 입력 크기
- S: 스트라이드
- P: 패딩
- K: 커널 크기
- P_out: 출력 패딩

### 2배 업샘플링 예시

```python
import torch.nn as nn

# 입력 14x14 → 출력 28x28
deconv = nn.ConvTranspose2d(
    in_channels=256,
    out_channels=128,
    kernel_size=4,
    stride=2,
    padding=1
)
# O = (14-1)*2 - 2*1 + 4 = 28
```

## 체커보드 아티팩트

Transposed Convolution의 문제: 불균일한 오버랩

```
stride=2, kernel=4일 때:
중앙 픽셀: 4번 더해짐
가장자리: 1번 더해짐
→ 격자 패턴 발생
```

### 해결책 1: Resize + Conv

```python
class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))
```

### 해결책 2: Pixel Shuffle

```python
# Sub-pixel convolution
class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale**2, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))

# (B, 64, 32, 32) → conv → (B, 256, 32, 32) → shuffle → (B, 64, 64, 64)
```

## 구현 비교

```python
import torch
import torch.nn as nn

x = torch.randn(1, 256, 14, 14)

# 1. Transposed Convolution
deconv = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
y1 = deconv(x)  # (1, 128, 28, 28)

# 2. Upsample + Conv (권장)
up_conv = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(256, 128, 3, padding=1)
)
y2 = up_conv(x)  # (1, 128, 28, 28)

# 3. Pixel Shuffle
pixel_shuffle = nn.Sequential(
    nn.Conv2d(256, 128 * 4, 3, padding=1),
    nn.PixelShuffle(2)
)
y3 = pixel_shuffle(x)  # (1, 128, 28, 28)
```

## 사용 사례

- **U-Net**: 인코더 특징과 합치며 업샘플링
- **GAN**: Generator에서 이미지 생성
- **Super Resolution**: 저해상도 → 고해상도

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d)
- [Pooling](/ko/docs/components/convolution/pooling)
