---
title: "U-Net"
weight: 1
math: true
---

# U-Net

## 개요

- **논문**: U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
- **저자**: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **핵심 기여**: Encoder-Decoder + Skip Connection으로 정밀한 segmentation

## 핵심 아이디어

> "수축 경로(encoder)로 context를 잡고, 확장 경로(decoder)로 정밀한 localization"

의료 영상처럼 데이터가 적은 상황에서도 효과적으로 동작합니다.

---

## 구조

### 전체 아키텍처

```
Input (572×572)
    ↓
┌───────────────────────────────────────────────────────────┐
│                    Contracting Path                        │
│  [Conv3×3 → ReLU → Conv3×3 → ReLU → MaxPool] × 4          │
│   64 → 128 → 256 → 512 → 1024                              │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│                    Expanding Path                          │
│  [UpConv2×2 → Concat(skip) → Conv3×3 → ReLU × 2] × 4     │
│   1024 → 512 → 256 → 128 → 64                              │
└───────────────────────────────────────────────────────────┘
    ↓
Conv 1×1 → Output (388×388×num_classes)
```

### U자 형태

```
Input                                           Output
  ↓                                               ↑
[Conv]─────────────────────────────────────→[UpConv+Concat]
  ↓                                               ↑
[Conv]───────────────────────────────→[UpConv+Concat]
  ↓                                         ↑
[Conv]─────────────────────────→[UpConv+Concat]
  ↓                               ↑
[Conv]──────────────────→[UpConv+Concat]
  ↓                    ↑
     [Bottleneck]
```

---

## 핵심 컴포넌트

### 1. Contracting Path (Encoder)

- 3×3 Conv (unpadded) + ReLU × 2
- 2×2 Max Pooling (stride 2)
- 채널 수 2배씩 증가

### 2. Expanding Path (Decoder)

- 2×2 Up-convolution (채널 절반)
- Skip connection으로 encoder 특징 concat
- 3×3 Conv + ReLU × 2

### 3. Skip Connection

Encoder의 고해상도 특징을 decoder로 전달:
- 위치 정보 보존
- 정밀한 경계 복원

---

## 구현 예시

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + Skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)
```

---

## 학습

### Loss Function

Binary Segmentation:
$$L = \text{BCE}(y, \hat{y}) + \text{Dice Loss}$$

Multi-class:
$$L = \text{CrossEntropy}(y, \hat{y})$$

### Data Augmentation

의료 영상은 데이터가 적으므로 augmentation이 중요:
- Elastic deformation
- 회전, 뒤집기
- 그레이스케일 변환

---

## U-Net 변형

| 변형 | 특징 |
|------|------|
| **U-Net++** | 중첩된 skip connection |
| **Attention U-Net** | Attention gate 추가 |
| **ResUNet** | ResNet 블록 사용 |
| **TransUNet** | Transformer encoder |

---

## 활용 분야

- **의료 영상**: 세포, 종양, 장기 segmentation
- **위성 영상**: 건물, 도로 추출
- **자율주행**: 도로 영역 분할
- **생성 모델**: [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion)의 구조에 영향

---

## 관련 콘텐츠

- [Transposed Convolution](/ko/docs/math/transposed-conv) - Upsampling 원리
- [ResNet](/ko/docs/architecture/cnn/resnet) - Skip connection 아이디어
- [Mask R-CNN](/ko/docs/architecture/segmentation/mask-rcnn) - Instance segmentation
- [Segmentation 태스크](/ko/docs/task/segmentation) - 평가 지표
