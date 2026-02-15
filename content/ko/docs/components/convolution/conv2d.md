---
title: "2D Convolution"
weight: 1
math: true
---

# 2D Convolution (2차원 합성곱)

{{% hint info %}}
**선수지식**: [행렬](/ko/docs/math/linear-algebra/matrix)
{{% /hint %}}

> **한 줄 요약**: Convolution은 작은 필터를 이미지 위에서 밀어가며 "이 부근에 이런 패턴이 있나?"를 묻는 연산입니다.

## 왜 Convolution이 필요한가?

### 문제 상황: "이미지를 이해하고 싶은데 픽셀이 너무 많습니다"

224×224 RGB 이미지는 **150,528개**의 숫자입니다. 이걸 한꺼번에 처리하면?

```python
# Fully Connected로 이미지 처리
input_size = 224 * 224 * 3  # = 150,528
hidden_size = 4096
params = input_size * hidden_size  # = 616,562,688 (약 6억 개!)
```

문제가 3가지 있습니다:

**1. 파라미터 폭발** — 첫 번째 레이어만 6억 개. GPU 메모리가 터집니다.

**2. 공간 정보 손실** — 픽셀을 일렬로 펴면 "위에 있는 픽셀"과 "아래 있는 픽셀"의 관계가 사라집니다.

**3. 이동 불변성 없음** — 고양이가 사진 왼쪽에 있든 오른쪽에 있든 "고양이"인데, FC는 위치가 바뀌면 완전히 다른 패턴으로 봅니다.

### 해결: "전체를 보지 말고, 작은 창으로 훑자"

돋보기로 책을 읽을 때를 생각해보세요:
- 돋보기(필터)의 크기는 항상 같습니다
- 돋보기를 **한 칸씩 밀며** 전체를 훑습니다
- 어디에서 글자를 보든 같은 돋보기를 사용합니다

이것이 바로 Convolution입니다.

{{< figure src="/images/components/convolution/ko/conv2d-sliding-window.jpeg" caption="필터(커널)가 이미지 위를 슬라이딩하며 특징을 추출하는 과정" >}}

---

## 핵심 원리 2가지: Local Connectivity + Weight Sharing

초보자가 Conv2d를 이해할 때 가장 자주 놓치는 핵심은 아래 2개입니다.

1. **Local Connectivity (국소 연결)**
   - 한 출력 픽셀은 입력 전체가 아니라, 커널이 덮는 **작은 영역(K×K)** 만 봅니다.
   - 그래서 파라미터 수가 폭발하지 않고, 에지/코너 같은 로컬 패턴을 잘 잡습니다.

2. **Weight Sharing (가중치 공유)**
   - 같은 커널(같은 숫자들)을 이미지 모든 위치에 반복 사용합니다.
   - 덕분에 "왼쪽 위의 세로선"과 "오른쪽 아래의 세로선"을 같은 패턴으로 인식할 수 있습니다.

한 줄로 정리하면: **Conv2d는 작은 창(Local)으로 보고, 같은 기준(Shared Weights)으로 전체를 훑습니다.**

## Convolution 연산

### 핵심 아이디어

입력 이미지의 작은 영역과 필터(커널)의 **원소별 곱의 합**을 구합니다.

$$
\text{output}(i, j) = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \text{input}(i+m, j+n) \cdot \text{kernel}(m, n)
$$

**각 기호의 의미:**
- $\text{input}(i+m, j+n)$ : 입력 이미지에서 (i+m, j+n) 위치의 픽셀값
- $\text{kernel}(m, n)$ : 필터의 (m, n) 위치 가중치 (학습으로 결정!)
- $K$ : 커널 크기 (보통 3)
- $\text{output}(i, j)$ : 출력 특징 맵의 (i, j) 위치 값

### 직관적 이해: "패턴 매칭 점수"

```
입력 영역:        커널 (세로선 탐지):    원소별 곱 → 합:
┌───────────┐     ┌───────────┐
│ 0   1   0 │     │-1   2  -1 │     0×(-1) + 1×2 + 0×(-1)
│ 0   1   0 │  ×  │-1   2  -1 │  =  0×(-1) + 1×2 + 0×(-1) = 6
│ 0   1   0 │     │-1   2  -1 │     0×(-1) + 1×2 + 0×(-1)
└───────────┘     └───────────┘
 세로선이 있다!    → 높은 점수(6)
```

```
입력 영역:        같은 커널:            원소별 곱 → 합:
┌───────────┐     ┌───────────┐
│ 1   0   0 │     │-1   2  -1 │     1×(-1) + 0×2 + 0×(-1)
│ 1   0   0 │  ×  │-1   2  -1 │  =  1×(-1) + 0×2 + 0×(-1) = -6
│ 1   0   0 │     │-1   2  -1 │     1×(-1) + 0×2 + 0×(-1)
└───────────┘     └───────────┘
 세로선 위치 다름  → 낮은 점수(-6)
```

커널이 찾는 패턴과 **비슷하면 높은 값**, 다르면 낮은 값이 나옵니다.

### 코드로 확인하기

```python
import torch
import torch.nn as nn

# 수동으로 convolution 계산
input_patch = torch.tensor([[0., 1., 0.],
                             [0., 1., 0.],
                             [0., 1., 0.]])

kernel = torch.tensor([[-1., 2., -1.],
                        [-1., 2., -1.],
                        [-1., 2., -1.]])

result = (input_patch * kernel).sum()
print(f"수동 계산: {result.item()}")  # 수동 계산: 6.0

# PyTorch Conv2d로 같은 결과
conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
conv.weight.data = kernel.reshape(1, 1, 3, 3)

x = input_patch.reshape(1, 1, 3, 3)
y = conv(x)
print(f"Conv2d 결과: {y.item()}")  # Conv2d 결과: 6.0
```

---

## 출력 크기 계산

### 공식

$$
O = \frac{I - K + 2P}{S} + 1
$$

**각 기호의 의미:**
- $O$ : 출력 크기 (Output)
- $I$ : 입력 크기 (Input)
- $K$ : 커널 크기 (Kernel)
- $P$ : 패딩 (Padding) — 입력 가장자리에 0을 채우는 양
- $S$ : 스트라이드 (Stride) — 필터가 한 번에 이동하는 칸 수

{{< figure src="/images/components/convolution/ko/conv2d-output-size.png" caption="입력 크기, 커널, 패딩, 스트라이드에 따른 출력 크기 변화" >}}

### 예시 계산

```python
def output_size(I, K, P, S):
    return (I - K + 2 * P) // S + 1

# 예시 1: 기본 (same padding)
print(output_size(I=32, K=3, P=1, S=1))   # 32 (크기 유지!)

# 예시 2: 패딩 없음
print(output_size(I=32, K=3, P=0, S=1))   # 30 (2 줄어듦)

# 예시 3: 다운샘플링
print(output_size(I=32, K=3, P=1, S=2))   # 16 (절반!)

# 예시 4: AlexNet 첫 번째 레이어
print(output_size(I=224, K=11, P=2, S=4)) # 55
```

---

## 핵심 파라미터 상세

### 1. 커널 크기 (Kernel Size)

| 크기 | 특징 | 사용처 |
|------|------|--------|
| **1×1** | 채널 간 정보 혼합, 차원 축소/확장 | GoogLeNet, ResNet bottleneck |
| **3×3** | 가장 일반적. 2개 쌓으면 5×5와 동일 RF | VGG 이후 표준 |
| **5×5** | 넓은 영역, 파라미터 많음 | AlexNet 초기 레이어 |
| **7×7** | 매우 넓은 영역 | ResNet 첫 레이어 |
| **11×11** | 가장 넓음 | AlexNet 첫 레이어 |

**왜 3×3이 표준이 되었나?**

VGG 논문의 핵심 발견: 3×3 두 개 = 5×5 하나와 같은 영역을 보지만, 파라미터는 더 적고 비선형성은 더 많습니다.

```python
# 5×5 conv: 25C² 파라미터, 비선형 1번
nn.Conv2d(C, C, 5, padding=2)

# 3×3 × 2: 18C² 파라미터 (28% 절약), 비선형 2번
nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(C, C, 3, padding=1),
)
```

### 2. 스트라이드 (Stride)

필터가 한 번에 몇 칸씩 이동하는지:

```python
# Stride 1: 한 칸씩 이동 (기본)
conv_s1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

# Stride 2: 두 칸씩 이동 → 출력 크기 절반 (다운샘플링)
conv_s2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

x = torch.randn(1, 64, 56, 56)
print(conv_s1(x).shape)  # torch.Size([1, 64, 56, 56])
print(conv_s2(x).shape)  # torch.Size([1, 128, 28, 28])
```

### 3. 패딩 (Padding)

입력 가장자리에 값(보통 0)을 추가:

- **padding=0 (valid)**: 출력이 줄어듦. 가장자리 정보 손실.
- **padding=K//2 (same)**: 출력 크기 = 입력 크기 유지. 가장 많이 사용.

```python
# Same padding: 입력 = 출력
conv_same = nn.Conv2d(64, 64, kernel_size=3, padding=1)   # 3//2 = 1
conv_same5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)  # 5//2 = 2
conv_same7 = nn.Conv2d(64, 64, kernel_size=7, padding=3)  # 7//2 = 3
```

### 4. 딜레이션 (Dilation)

커널 원소 사이에 간격을 넣어 **같은 파라미터로 더 넓은 영역**을 봅니다:

```
일반 3×3 (dilation=1):     Dilated 3×3 (dilation=2):
■ ■ ■                      ■ · ■ · ■
■ ■ ■     → 3×3 영역       · · · · ·
■ ■ ■                      ■ · ■ · ■    → 5×5 영역!
                           · · · · ·
                           ■ · ■ · ■
```

$$
O = \frac{I - d(K-1) - 1 + 2P}{S} + 1
$$

- $d$ : dilation rate

```python
# Dilated convolution: 파라미터 9개로 5×5 영역 커버
conv_d2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)

# 주로 Segmentation에서 사용 (DeepLab)
```

---

## 멀티채널 Convolution

### 현실의 Conv2d: 3D 연산

실제로는 입력이 여러 채널(RGB의 3채널, 중간 레이어의 64채널 등)을 가집니다.

{{< figure src="/images/components/convolution/ko/conv2d-multi-channel.jpeg" caption="멀티채널 입력에서의 Convolution: 각 채널별로 다른 커널을 적용하고 결과를 합산" >}}

```
입력: (C_in × H × W)           커널: (C_in × K × K)         출력: (1 × H' × W')
┌──────────────────┐           ┌──────────────────┐
│  채널 1 (H × W)  │           │  커널 1 (K × K)  │
│  채널 2 (H × W)  │     ×     │  커널 2 (K × K)  │     →    1장의 출력
│  채널 3 (H × W)  │           │  커널 3 (K × K)  │
└──────────────────┘           └──────────────────┘

이런 커널 세트가 C_out개 → 출력: (C_out × H' × W')
```

```python
# RGB 입력(3채널)에서 64개 특징 맵 추출
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

# 커널 형태: (C_out, C_in, K, K) = (64, 3, 3, 3)
print(conv.weight.shape)  # torch.Size([64, 3, 3, 3])

x = torch.randn(1, 3, 224, 224)   # (B, C_in, H, W)
y = conv(x)
print(y.shape)  # torch.Size([1, 64, 224, 224])  (B, C_out, H', W')
```

---

## 파라미터 수 계산

```
파라미터 수 = (K × K × C_in + 1) × C_out
                ─────────────   ─     ────
                 커널 크기     bias   필터 개수
```

```python
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

# 수동 계산
manual = (3 * 3 * 64 + 1) * 128
print(f"수동 계산: {manual:,}")  # 수동 계산: 73,856

# PyTorch 확인
pytorch = sum(p.numel() for p in conv.parameters())
print(f"PyTorch: {pytorch:,}")  # PyTorch: 73,856
```

### 실전 파라미터 비교

| 모델 첫 레이어 | 설정 | 파라미터 수 |
|---|---|---|
| AlexNet | Conv2d(3, 96, 11, stride=4) | 34,944 |
| VGG | Conv2d(3, 64, 3, padding=1) | 1,792 |
| ResNet | Conv2d(3, 64, 7, stride=2, padding=3) | 9,472 |

---

## 특수한 Convolution

### 1×1 Convolution

공간은 건드리지 않고 **채널만 혼합/변환**합니다:

```python
# 채널 축소 (bottleneck)
reduce = nn.Conv2d(256, 64, kernel_size=1)   # 256채널 → 64채널
# 파라미터: (1×1×256+1)×64 = 16,448

# 채널 확장
expand = nn.Conv2d(64, 256, kernel_size=1)   # 64채널 → 256채널

x = torch.randn(1, 256, 56, 56)
print(reduce(x).shape)  # torch.Size([1, 64, 56, 56])  공간은 그대로!
```

GoogLeNet과 ResNet의 bottleneck 구조에서 핵심 역할을 합니다.

### Depthwise Separable Convolution

일반 convolution을 두 단계로 분리하여 **파라미터를 획기적으로 줄입니다**:

```
일반 Convolution:
  입력 (C_in × H × W) → 커널 (C_out × C_in × K × K) → 출력 (C_out × H' × W')

Depthwise Separable:
  Step 1: Depthwise — 채널별 독립 연산
    입력 (C_in × H × W) → 커널 (C_in × 1 × K × K) → 중간 (C_in × H' × W')

  Step 2: Pointwise — 1×1 conv로 채널 혼합
    중간 (C_in × H' × W') → 커널 (C_out × C_in × 1 × 1) → 출력 (C_out × H' × W')
```

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        # Step 1: 채널별 독립 convolution (groups=in_ch)
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            padding=kernel_size // 2, groups=in_ch
        )
        # Step 2: 1×1 conv로 채널 혼합
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# 파라미터 비교
conv_normal = nn.Conv2d(64, 128, 3, padding=1)
conv_sep = DepthwiseSeparableConv(64, 128)

params_normal = sum(p.numel() for p in conv_normal.parameters())
params_sep = sum(p.numel() for p in conv_sep.parameters())

print(f"일반: {params_normal:,}")     # 일반: 73,856
print(f"Separable: {params_sep:,}")  # Separable: 8,896
print(f"비율: {params_normal / params_sep:.1f}배 감소")  # 비율: 8.3배 감소
```

MobileNet, EfficientNet 등 경량 모델의 핵심입니다.

---

## 특징 계층 구조

Conv 레이어를 쌓으면 점점 복잡한 패턴을 인식합니다:

{{< figure src="/images/components/convolution/ko/conv2d-feature-hierarchy.jpeg" caption="레이어가 깊어질수록 저수준(에지) → 중수준(텍스처) → 고수준(객체 부분) 특징을 추출" >}}

```
Layer 1-2: 에지, 색상 변화     → "경계선이 있나?"
Layer 3-4: 텍스처, 코너         → "줄무늬인가? 점박이인가?"
Layer 5-7: 객체 부분            → "바퀴인가? 눈인가?"
Layer 8+:  전체 객체            → "자동차인가? 고양이인가?"
```

이것이 **모든 CNN의 핵심 원리**입니다. 단순한 패턴 감지기를 쌓아서 복잡한 패턴을 인식합니다.

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| 3×3 conv | [VGG](/ko/docs/architecture/cnn/vgg), [ResNet](/ko/docs/architecture/cnn/resnet) | 현대 CNN의 표준 블록 |
| 1×1 conv | [ResNet](/ko/docs/architecture/cnn/resnet) bottleneck | 채널 차원 조절 |
| Stride 2 | [ResNet](/ko/docs/architecture/cnn/resnet), 대부분의 CNN | Pooling 대체 다운샘플링 |
| Dilated conv | DeepLab, Segmentation 모델 | 넓은 RF, 해상도 유지 |
| Depthwise Sep. | MobileNet, EfficientNet | 경량화 |
| Transposed conv | [U-Net](/ko/docs/architecture/segmentation/unet), GAN | 업샘플링 ([상세](/ko/docs/components/convolution/transposed-conv)) |

---

## 초보자 실수 방지: Shape 추적 60초 루틴

Conv2d 디버깅의 80%는 shape 확인으로 해결됩니다. 아래 3줄을 습관처럼 찍어보세요.

```python
import torch
import torch.nn as nn

x = torch.randn(2, 3, 32, 32)                 # (B, C, H, W)
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
y = conv(x)

print("input :", x.shape)   # torch.Size([2, 3, 32, 32])
print("weight:", conv.weight.shape)  # torch.Size([16, 3, 3, 3])
print("output:", y.shape)   # torch.Size([2, 16, 16, 16])
```

읽는 법:
- `weight.shape = [C_out, C_in, K, K]`
- 출력 채널 수는 `C_out`으로 결정
- `stride=2`면 보통 공간 크기(H, W)가 절반으로 줄어듦

## 증상 → 원인 빠른 매핑

| 관측 증상 | 가장 흔한 원인 | 먼저 볼 항목 |
|---|---|---|
| `Expected ... to have 3 channels, but got 1` | 입력 채널/레이어 설정 불일치 | `in_channels`, 데이터 전처리(RGB/Grayscale) |
| 출력 크기가 예상보다 작음 | padding 누락 또는 stride 과대 | `padding`, `stride`, 출력 크기 공식 |
| 학습 초반 loss가 NaN | 학습률 과대, mixed precision 불안정 | learning rate, AMP 설정, gradient clipping |
| 파라미터 수가 너무 큼 | 커널/채널 설정 과도 | `C_in`, `C_out`, kernel size 재설계 |

## 수식 ↔ 코드 연결: 출력 크기 사전 점검

실전에서 가장 자주 나는 실수는 **"공식은 맞게 외웠는데 코드에서 정수로 안 떨어지는 경우"**입니다.
아래처럼 레이어 정의 전에 출력 크기를 먼저 계산하면, 중간에 shape 에러로 멈추는 일을 크게 줄일 수 있습니다.

```python
def conv2d_out(i, k=3, p=1, s=1):
    numer = i - k + 2 * p
    assert numer % s == 0, (
        f"출력 크기가 정수가 아닙니다: (I-K+2P)={numer}, stride={s}"
    )
    return numer // s + 1

h = conv2d_out(i=31, k=3, p=1, s=2)
print(h)  # 16
```

체크 포인트:
- `(I - K + 2P)`가 `stride`로 나누어 떨어지는가?
- 다운샘플링(`stride=2`)을 여러 번 쓰면 해상도가 너무 빨리 줄지 않는가?
- Detection/Segmentation에서는 마지막 feature map 해상도가 과도하게 작아지지 않았는가?

## 실무 미니 점검: HWC ↔ CHW 혼동 빠르게 잡기

이미지 라이브러리(OpenCV, PIL, numpy)는 보통 `(H, W, C)` 순서를 사용하지만,
PyTorch `Conv2d`는 `(B, C, H, W)`를 기대합니다.

아래 20초 점검 코드를 데이터 로더 직후에 넣으면, 채널 순서 실수를 거의 바로 잡을 수 있습니다.

```python
import torch

# numpy/PIL에서 온 샘플이라고 가정: (H, W, C)
x_hwc = torch.randn(224, 224, 3)

# Conv2d 입력으로 변환: (C, H, W) -> (B, C, H, W)
x_chw = x_hwc.permute(2, 0, 1).contiguous()
x_bchw = x_chw.unsqueeze(0)

print("HWC :", x_hwc.shape)   # torch.Size([224, 224, 3])
print("BCHW:", x_bchw.shape)  # torch.Size([1, 3, 224, 224])
```

체크 포인트:
- `permute(2, 0, 1)`로 채널 축을 앞으로 이동했는가?
- 배치 차원 `unsqueeze(0)`을 추가했는가?
- 모델의 `in_channels`(예: 1, 3)와 실제 입력 채널 수가 일치하는가?

## 5분 실험: Stride/Padding이 결과에 미치는 영향

"공식은 알겠는데 감이 안 온다"를 줄이려면, 같은 입력에 `stride/padding`만 바꿔 출력 크기를 직접 비교해보는 게 가장 빠릅니다.

```python
import torch
import torch.nn as nn

x = torch.randn(1, 3, 32, 32)

settings = [
    {"k": 3, "s": 1, "p": 1},  # same
    {"k": 3, "s": 2, "p": 1},  # downsample
    {"k": 5, "s": 1, "p": 0},  # valid
]

for cfg in settings:
    conv = nn.Conv2d(3, 8, kernel_size=cfg["k"], stride=cfg["s"], padding=cfg["p"])
    y = conv(x)
    print(cfg, "->", tuple(y.shape))
```

학습 체크:
- `stride=2`일 때 공간 해상도(H, W)가 어떻게 줄어드는지 설명할 수 있는가?
- `padding=0`일 때 가장자리 정보가 왜 줄어드는지 말로 설명할 수 있는가?
- detection/segmentation에서 해상도 축소가 성능에 어떤 영향을 줄지 예측할 수 있는가?

## 초보자 완료 체크 (Level 1 브리지)
아래 5개를 말로 설명하거나 코드로 확인할 수 있으면, Conv2d 기초를 통과한 것입니다.

- [ ] `weight.shape = [C_out, C_in, K, K]` 를 보고 각 축 의미를 설명할 수 있다.
- [ ] 출력 크기 공식 $O = \frac{I-K+2P}{S}+1$ 로 레이어 출력 해상도를 미리 계산할 수 있다.
- [ ] `HWC -> CHW -> BCHW` 변환을 직접 수행하고, 왜 필요한지 설명할 수 있다.
- [ ] `stride/padding`을 바꿨을 때 분류 vs 검출/분할에서 어떤 영향이 나는지 예측할 수 있다.
- [ ] Local Connectivity와 Weight Sharing이 파라미터 절감/일반화에 왜 유리한지 설명할 수 있다.

## 10분 보강 실습 (막힘 해소용)
"공식은 알겠는데 모델에서 자꾸 shape 에러가 난다"는 초보자에게 가장 효과적인 루틴입니다.

1. 랜덤 입력 `(1,3,64,64)`으로 Conv2d를 2개만 쌓아 출력 shape를 손으로 먼저 계산
2. 실제 코드 출력 shape와 비교해 불일치가 있으면 `padding/stride`만 수정
3. 마지막에 `assert y.shape[2] > 0 and y.shape[3] > 0`로 붕괴 여부 확인

핵심은 **학습 데이터를 붙이기 전에 shape 수학을 먼저 맞추는 것**입니다.

## 실무 실패 패턴 추가: `groups`/Depthwise 설정

`Depthwise Separable Conv`를 직접 구현할 때 초보자가 가장 자주 막히는 지점은 `groups` 값입니다.

핵심 규칙:
- 일반 Conv: `groups=1`
- Depthwise Conv: `groups=in_channels` 그리고 `out_channels`는 보통 `in_channels`의 배수

```python
import torch
import torch.nn as nn

x = torch.randn(1, 8, 32, 32)

# 올바른 depthwise 예시
ok = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8)
print(ok(x).shape)  # torch.Size([1, 8, 32, 32])

# 잘못된 예시: groups가 in_channels를 나누지 못함
try:
    bad = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=3)
except Exception as e:
    print("에러:", e)
```

빠른 점검 체크:
- [ ] `in_channels % groups == 0`
- [ ] `out_channels % groups == 0`
- [ ] depthwise 단계와 pointwise(1×1) 단계를 혼동하지 않았는가?

## 시각자료(나노바나나) 프롬프트
- **KO 다이어그램 1 (Stride/Padding 비교)**  
  "다크 테마 배경(#1a1a2e), Conv2d stride/padding 비교 인포그래픽. 동일한 32x32 입력에서 (k=3,s=1,p=1), (k=3,s=2,p=1), (k=5,s=1,p=0) 세 설정을 좌→우 패널로 배치. 각 패널에 입력 격자, 필터 이동 간격, 출력 크기(32x32 / 16x16 / 28x28) 명확히 표시. 한국어 라벨과 화살표, 초보자용 깔끔한 벡터 스타일"

- **KO 다이어그램 2 (HWC→BCHW 변환)**  
  "다크 테마 배경(#1a1a2e), 텐서 차원 변환 다이어그램. HWC(224,224,3) 텐서를 permute(2,0,1) 후 CHW(3,224,224), unsqueeze 후 BCHW(1,3,224,224)로 변환하는 단계별 화살표 표시. 한국어 라벨, 축 이름(C/H/W) 강조, 실수 포인트(채널 축 위치) 경고 아이콘 포함"

## 관련 콘텐츠

- [행렬](/ko/docs/math/linear-algebra/matrix) — Convolution의 행렬 표현 (Toeplitz matrix)
- [Pooling](/ko/docs/components/convolution/pooling) — 다운샘플링 연산
- [Receptive Field](/ko/docs/components/convolution/receptive-field) — 출력이 "보는" 입력 영역
- [Transposed Convolution](/ko/docs/components/convolution/transposed-conv) — 업샘플링 연산
- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — Conv 뒤에 거의 항상 함께 사용
- [AlexNet](/ko/docs/architecture/cnn/alexnet) — Convolution을 깊게 쌓은 최초의 모델
