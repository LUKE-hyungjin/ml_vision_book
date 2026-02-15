---
title: "U-Net"
weight: 1
math: true
---

# U-Net

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d) | [Transposed Convolution](/ko/docs/components/convolution/transposed-conv)
{{% /hint %}}

## 한 줄 요약
> **U-Net은 고해상도 위치 정보(encoder의 skip)와 의미 정보(decoder 복원)를 결합해, 적은 데이터에서도 픽셀 단위 분할을 잘 하도록 만든 구조입니다.**

## 왜 이 모델인가?
초보자가 segmentation을 처음 구현하면 자주 겪는 문제가 있습니다.

1. 분류는 맞는데 **경계가 흐릿함**
2. 다운샘플링 후 업샘플링 과정에서 **위치 정보가 사라짐**
3. 데이터가 적으면 쉽게 과적합

U-Net은 이 문제를 "encoder에서 압축한 의미 정보 + skip으로 전달한 위치 정보"를 합쳐 해결합니다.
비유하면, 큰 지도를 축약해 길을 찾은 뒤(encoder), 원본 지도의 골목 정보(skip)를 다시 덧붙여 정확한 경계를 복원(decoder)하는 방식입니다.

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

**기호 설명:**
- $y$: 정답 마스크(ground truth)
- $\hat{y}$: 모델이 예측한 마스크
- $L$: 최소화하려는 전체 손실

초보자 관점에서 핵심은 다음입니다.
- BCE/CrossEntropy: 픽셀별 "맞았는지"를 학습
- Dice: 경계/작은 객체처럼 클래스 불균형이 큰 영역을 보정

### Data Augmentation

의료 영상은 데이터가 적으므로 augmentation이 중요:
- Elastic deformation
- 회전, 뒤집기
- 그레이스케일 변환

## 실무 디버깅 체크리스트
- [ ] **입출력 해상도 확인**: 예측 마스크 크기와 정답 마스크 크기가 정확히 같은가?
- [ ] **skip concat shape 확인**: `torch.cat` 직전 텐서의 H/W가 다르면 center-crop 또는 interpolate로 정렬했는가?
- [ ] **손실 함수-출력 활성화 일치**: BCEWithLogitsLoss를 쓰면 출력에 sigmoid를 중복 적용하지 않았는가?
- [ ] **클래스 불균형 대응**: 전경 픽셀이 매우 적다면 Dice/Focal 계열을 함께 사용했는가?
- [ ] **증강 강도 점검**: 과한 deformation/회전으로 라벨 정합성이 깨지지 않았는가?

## 자주 하는 실수 (FAQ)
**Q1. U-Net은 무조건 작은 데이터셋에서만 쓰나요?**  
A. 아닙니다. 작은 데이터에서 강점이 두드러지지만, 산업 데이터셋에서도 baseline으로 매우 많이 사용됩니다.

**Q2. Decoder에서 업샘플만 하면 충분한가요?**  
A. 보통은 부족합니다. skip connection이 없으면 경계 복원이 약해지는 경우가 많습니다.

**Q3. 학습 loss는 줄는데 마스크 경계가 울퉁불퉁합니다. 왜 그런가요?**  
A. 업샘플 아티팩트, 클래스 불균형, 불충분한 고해상도 특징 전달(weak skip) 가능성을 먼저 확인하세요.

## 초보자가 가장 많이 막히는 3가지 (빠른 처방)

| 막히는 지점 | 흔한 원인 | 먼저 할 조치 |
|---|---|---|
| `torch.cat`에서 shape 에러 | 입력 해상도가 2의 거듭제곱 배수가 아님, encoder/decoder H/W 불일치 | skip concat 직전에 `F.interpolate` 또는 center-crop으로 H/W 정렬 |
| 마스크가 전체적으로 비어 있음 | sigmoid/softmax/threshold 조합 오류 | Binary는 `BCEWithLogitsLoss` + 추론 시 `sigmoid` 1회만 적용 |
| Dice는 오르는데 IoU가 정체 | 경계 오차가 큰데 내부 픽셀만 맞는 경우 | 작은 객체 샘플 비중 확대 + 경계 중심 augmentation 점검 |

## 손실 함수-출력 활성화 매칭표 (실전에서 자주 헷갈림)

| 태스크 | 모델 출력 채널 | 학습 손실 | 추론 활성화 | 비고 |
|---|---:|---|---|---|
| Binary Segmentation | 1 | `BCEWithLogitsLoss` | `sigmoid` | 학습 시 sigmoid 중복 금지 |
| Multi-class Segmentation | C | `CrossEntropyLoss` | `softmax` + `argmax` | GT는 보통 class index mask |
| Binary + 클래스 불균형 큼 | 1 | `BCEWithLogitsLoss + Dice` | `sigmoid` | 작은 전경 객체에 유리 |

## 미니 검증 코드: 출력 shape + 확률 범위 sanity check

```python
import torch

model = UNet(in_channels=3, out_channels=1).eval()
x = torch.randn(2, 3, 256, 256)

with torch.no_grad():
    logits = model(x)                  # (B, 1, H, W)
    probs = torch.sigmoid(logits)      # [0, 1]

print("logits shape:", logits.shape)
print("prob range:", float(probs.min()), float(probs.max()))

assert logits.shape == (2, 1, 256, 256)
assert probs.min() >= 0.0 and probs.max() <= 1.0
```

이 체크만 먼저 통과해도, 초반 디버깅 시간(특히 shape/activation 관련)을 크게 줄일 수 있습니다.

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

- [Transposed Convolution](/ko/docs/components/convolution/transposed-conv) - Upsampling 원리
- [ResNet](/ko/docs/architecture/cnn/resnet) - Skip connection 아이디어
- [Mask R-CNN](/ko/docs/architecture/segmentation/mask-rcnn) - Instance segmentation
- [Segmentation 태스크](/ko/docs/task/segmentation) - 평가 지표
