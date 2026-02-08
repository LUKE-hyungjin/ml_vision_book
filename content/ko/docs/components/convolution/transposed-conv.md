---
title: "Transposed Convolution"
weight: 4
math: true
---

# Transposed Convolution (전치 합성곱)

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d)
{{% /hint %}}

> **한 줄 요약**: Transposed Convolution은 특징 맵의 **공간 해상도를 키우는** 학습 가능한 업샘플링 연산입니다.

## 왜 Transposed Convolution이 필요한가?

### 문제 상황: "줄인 건 좋은데, 다시 키워야 합니다"

CNN의 encoder는 이미지를 점점 작게 만듭니다:

```
입력: 256×256 → Conv+Pool → 128×128 → Conv+Pool → 64×64 → ... → 8×8
```

그런데 다음과 같은 상황에서는 **원래 크기로 되돌려야** 합니다:

**Segmentation** — 모든 픽셀에 레이블을 붙여야 함:
```
입력: 256×256 (이미지)
출력: 256×256 (각 픽셀이 고양이/배경/...)  ← 원래 크기 필요!
```

**이미지 생성 (GAN)** — 작은 벡터에서 큰 이미지를 만들어야 함:
```
입력: 1×1×512 (랜덤 노이즈)
출력: 256×256×3 (생성된 이미지)  ← 키워야 함!
```

**Super Resolution** — 저해상도를 고해상도로:
```
입력: 64×64 (저해상도)
출력: 256×256 (고해상도)  ← 4배 키워야 함!
```

{{< figure src="/images/components/convolution/ko/transposed-conv-need.jpeg" caption="Encoder는 축소, Decoder는 확대 — Transposed Convolution은 Decoder의 핵심" >}}

---

## 동작 원리

### "Convolution의 반대 방향"

이름이 "Transposed"인 이유: 일반 convolution을 행렬로 표현했을 때, 그 행렬의 **전치(transpose)**를 사용하기 때문입니다.

```
일반 Convolution:        Transposed Convolution:
큰 입력 → 작은 출력      작은 입력 → 큰 출력

[4×4] → Conv → [2×2]    [2×2] → TransConv → [4×4]
 축소                      확대
```

{{% hint warning %}}
**주의**: "Deconvolution"이라고 부르기도 하지만, 수학적 역연산(inverse)이 **아닙니다**. 원본 값을 복원하는 것이 아니라, **크기만 복원**합니다. 학습을 통해 적절한 업샘플링을 배웁니다.
{{% /hint %}}

### 연산 과정

Transposed Convolution은 입력의 각 값을 커널과 곱하여 **넓은 영역에 흩뿌리고**, 겹치는 부분을 더합니다:

```
입력 (2×2):          커널 (3×3):
┌───┬───┐            ┌───────────┐
│ 1 │ 2 │            │ a  b  c   │
├───┼───┤            │ d  e  f   │
│ 3 │ 4 │            │ g  h  i   │
└───┴───┘            └───────────┘

Step 1: 입력[0,0]=1 × 커널 → 출력 왼쪽 위에 배치
Step 2: 입력[0,1]=2 × 커널 → 출력 오른쪽 위에 배치 (겹치는 부분은 더함)
Step 3: 입력[1,0]=3 × 커널 → 출력 왼쪽 아래에 배치 (겹치는 부분은 더함)
Step 4: 입력[1,1]=4 × 커널 → 출력 오른쪽 아래에 배치 (겹치는 부분은 더함)
```

{{< figure src="/images/components/convolution/ko/transposed-conv-operation.jpeg" caption="Transposed Convolution: 입력의 각 값이 커널을 통해 넓은 출력 영역에 기여" >}}

---

## 출력 크기 계산

$$
O = (I - 1) \times S - 2P + K + P_{out}
$$

**각 기호의 의미:**
- $O$ : 출력 크기
- $I$ : 입력 크기
- $S$ : 스트라이드 (보통 2 = 2배 확대)
- $P$ : 패딩
- $K$ : 커널 크기
- $P_{out}$ : 출력 패딩 (`output_padding`)

일반 Conv의 공식과 **역관계**입니다:
- Conv: `O = (I - K + 2P) / S + 1`
- TransConv: `O = (I - 1) × S - 2P + K`

### 일반적인 2배 업샘플링 설정

```python
import torch
import torch.nn as nn

# 가장 많이 쓰는 설정: kernel=4, stride=2, padding=1
# O = (I - 1) × 2 - 2×1 + 4 = 2I
deconv = nn.ConvTranspose2d(
    in_channels=256,
    out_channels=128,
    kernel_size=4,     # 짝수 커널
    stride=2,          # 2배 확대
    padding=1          # 출력 크기 조절
)

x = torch.randn(1, 256, 14, 14)
y = deconv(x)
print(f"입력: {x.shape}")  # 입력: torch.Size([1, 256, 14, 14])
print(f"출력: {y.shape}")  # 출력: torch.Size([1, 128, 28, 28])  ← 정확히 2배!

# 또 다른 설정: kernel=2, stride=2, padding=0
deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
y2 = deconv2(x)
print(f"출력2: {y2.shape}")  # 출력2: torch.Size([1, 128, 28, 28])
```

---

## 체커보드 아티팩트

### 문제

Transposed Convolution의 가장 큰 문제: **불균일한 오버랩**으로 격자 패턴이 생깁니다.

```
stride=2, kernel=4일 때 각 출력 위치에 기여하는 입력 수:

┌───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 2 │ 2 │ 2 │ 1 │   ← 가장자리: 1번
├───┼───┼───┼───┼───┼───┤
│ 2 │ 4 │ 4 │ 4 │ 4 │ 2 │   ← 중앙: 4번
├───┼───┼───┼───┼───┼───┤
│ 2 │ 4 │ 4 │ 4 │ 4 │ 2 │
├───┼───┼───┼───┼───┼───┤
│ 1 │ 2 │ 2 │ 2 │ 2 │ 1 │
└───┴───┴───┴───┴───┴───┘

1과 4의 차이 → 밝기 불균형 → 체커보드!
```

{{< figure src="/images/components/convolution/ko/checkerboard-artifact.jpeg" caption="체커보드 아티팩트: 불균일한 오버랩으로 격자 패턴이 생김 (stride가 kernel_size를 나누지 않을 때 심함)" >}}

### 해결책 1: Upsample + Conv (가장 권장)

먼저 크기를 키우고(보간), 그 다음 일반 conv로 특징을 학습:

```python
class UpsampleConv(nn.Module):
    """보간 후 convolution — 체커보드 없음"""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=scale,
            mode='bilinear',
            align_corners=True
        )
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

up_conv = UpsampleConv(256, 128)
x = torch.randn(1, 256, 14, 14)
y = up_conv(x)
print(f"출력: {y.shape}")  # 출력: torch.Size([1, 128, 28, 28])
```

**장점**: 체커보드 없음, 안정적
**단점**: Upsample + Conv 두 단계로 약간 느림

### 해결책 2: Pixel Shuffle (Sub-pixel Convolution)

채널을 공간으로 재배치하는 방법:

```python
class PixelShuffleUp(nn.Module):
    """채널 → 공간 변환 (Super Resolution에서 많이 사용)"""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        # 채널을 scale² 배 늘림
        self.conv = nn.Conv2d(in_ch, out_ch * scale**2, 3, padding=1)
        # 채널 → 공간으로 재배치
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))

ps = PixelShuffleUp(256, 128, scale=2)
x = torch.randn(1, 256, 14, 14)
y = ps(x)
print(f"출력: {y.shape}")  # 출력: torch.Size([1, 128, 28, 28])

# 내부 동작:
# (1, 256, 14, 14) → conv → (1, 512, 14, 14) → shuffle → (1, 128, 28, 28)
#                         512 = 128 × 2²
```

**장점**: 빠름, 체커보드 적음
**단점**: 메모리 사용량 증가 (중간에 채널 폭발)

### 해결책 3: kernel_size = stride (오버랩 균일화)

```python
# kernel=2, stride=2: 오버랩 없음 → 체커보드 없음
deconv_no_overlap = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
# 단점: 커널이 작아서 학습 능력 제한
```

---

## 3가지 방법 비교

```python
import torch
import torch.nn as nn

x = torch.randn(1, 256, 14, 14)

# 방법 1: Transposed Convolution
method1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
y1 = method1(x)

# 방법 2: Upsample + Conv
method2 = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(256, 128, 3, padding=1)
)
y2 = method2(x)

# 방법 3: Pixel Shuffle
method3 = nn.Sequential(
    nn.Conv2d(256, 128 * 4, 3, padding=1),
    nn.PixelShuffle(2)
)
y3 = method3(x)

for i, y in enumerate([y1, y2, y3], 1):
    params = sum(p.numel() for p in [method1, method2, method3][i-1].parameters())
    print(f"방법 {i}: {y.shape}, 파라미터: {params:,}")

# 방법 1: torch.Size([1, 128, 28, 28]), 파라미터: 524,416
# 방법 2: torch.Size([1, 128, 28, 28]), 파라미터: 295,168
# 방법 3: torch.Size([1, 128, 28, 28]), 파라미터: 295,040
```

| 방법 | 체커보드 | 속도 | 품질 | 사용처 |
|------|---------|------|------|--------|
| TransConv | 있을 수 있음 | 빠름 | 보통 | [U-Net](/ko/docs/architecture/segmentation/unet) 전통 |
| Upsample+Conv | **없음** | 보통 | **좋음** | 현대 모델 (권장) |
| Pixel Shuffle | 거의 없음 | **빠름** | 좋음 | Super Resolution |

---

## 실전 사용 예시

### U-Net Decoder

```python
class UNetDecoderBlock(nn.Module):
    """U-Net의 업샘플링 블록"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # 업샘플링 (2배)
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        # Skip connection과 합친 후 conv
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)                       # 업샘플링
        x = torch.cat([x, skip], dim=1)      # Skip connection
        return self.conv(x)

# 사용
block = UNetDecoderBlock(512, 256, 256)
x = torch.randn(1, 512, 14, 14)     # 인코더 출력
skip = torch.randn(1, 256, 28, 28)  # 인코더 중간 특징
y = block(x, skip)
print(f"출력: {y.shape}")  # 출력: torch.Size([1, 256, 28, 28])
```

### GAN Generator

```python
class Generator(nn.Module):
    """DCGAN 스타일 Generator (간략화)"""
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            # 입력: (B, 100, 1, 1)
            nn.ConvTranspose2d(100, 512, 4, 1, 0),   # → (B, 512, 4, 4)
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # → (B, 256, 8, 8)
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # → (B, 128, 16, 16)
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # → (B, 64, 32, 32)
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),      # → (B, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(-1, 100, 1, 1))

gen = Generator()
z = torch.randn(4, 100)
img = gen(z)
print(f"생성 이미지: {img.shape}")  # 생성 이미지: torch.Size([4, 3, 64, 64])
```

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| TransConv 2× | [U-Net](/ko/docs/architecture/segmentation/unet) | Decoder에서 해상도 복원 |
| TransConv 연쇄 | [GAN](/ko/docs/architecture/generative/gan) Generator | 노이즈 → 이미지 생성 |
| Upsample+Conv | [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) | 아티팩트 없는 업샘플링 |
| Pixel Shuffle | Super Resolution (ESPCN) | 효율적 고해상도화 |

---

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d) — Transposed Conv의 정방향 연산
- [Pooling](/ko/docs/components/convolution/pooling) — 다운샘플링 (반대 방향)
- [U-Net](/ko/docs/architecture/segmentation/unet) — Transposed Conv의 대표적 사용처
- [GAN](/ko/docs/architecture/generative/gan) — Generator에서 이미지 업샘플링
