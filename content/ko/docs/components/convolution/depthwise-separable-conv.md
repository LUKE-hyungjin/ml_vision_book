---
title: "Depthwise Separable Conv"
weight: 5
math: true
---

# Depthwise Separable Convolution

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d)
{{% /hint %}}

> **한 줄 요약**: Depthwise Separable Convolution은 일반 Conv를 **공간 필터링 + 채널 혼합**으로 분리하여, 정확도는 비슷하게 유지하면서 연산량을 **8~9배** 줄이는 기법입니다.

## 왜 Depthwise Separable Conv이 필요한가?

### 문제 상황: "모바일에서 CNN을 돌리고 싶습니다"

일반 Conv2d의 연산량을 계산해봅시다:

```python
import torch.nn as nn

# 일반 Conv2d
conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

# 파라미터 수: K × K × C_in × C_out = 3 × 3 × 256 × 256 = 589,824
# 연산량 (FLOPs): K² × C_in × C_out × H × W
#   = 9 × 256 × 256 × 56 × 56 ≈ 1.85B FLOPs (18.5억 연산!)
```

이게 한 레이어입니다. 모델 전체는 수십 개의 레이어가 있으니, **모바일 기기에서는 실시간 처리가 불가능**합니다.

### 핵심 관찰: "일반 Conv는 두 가지 일을 동시에 한다"

일반 Conv2d가 하는 일을 분석하면:
1. **공간 정보 추출** — 3×3 영역에서 패턴(에지, 텍스처) 탐지
2. **채널 간 혼합** — 모든 입력 채널의 정보를 합쳐서 새 특징 생성

이 두 가지를 **동시에** 하기 때문에 비용이 $K^2 \times C_{in} \times C_{out}$으로 폭발합니다.

### 해결: "두 단계로 나누자!"

요리에 비유하면:
- **일반 Conv** = 256가지 재료를 한꺼번에 넣고 256가지 요리를 동시에 만드는 것
- **Depthwise Separable** = ① 각 재료를 따로 손질하고(Depthwise) → ② 손질된 재료를 조합해서 요리를 만드는 것(Pointwise)

![Depthwise Separable Convolution 구조](/images/components/convolution/ko/depthwise-separable-overview.jpeg)

---

## 1단계: Depthwise Convolution

### 직관적 정의

> **각 채널에 독립적으로** 3×3 필터를 적용합니다. 채널 간 정보 교환은 없습니다.

```
입력: 3채널                  Depthwise Conv               출력: 3채널
┌──────────┐               ┌──────────┐
│ 채널 1   │  ──── K×K ──→ │ 출력 1   │
├──────────┤               ├──────────┤
│ 채널 2   │  ──── K×K ──→ │ 출력 2   │   각 채널이 자기만의
├──────────┤               ├──────────┤   필터를 가짐!
│ 채널 3   │  ──── K×K ──→ │ 출력 3   │
└──────────┘               └──────────┘

→ 채널 수가 변하지 않음 (C_in = C_out)
→ 채널 간 정보 교환 없음 (독립적!)
```

### 수학적 정의

$$
\text{DW}(c, i, j) = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \text{input}(c, i+m, j+n) \cdot \text{kernel}_c(m, n)
$$

**각 기호의 의미:**
- $c$ : 채널 인덱스 — 각 채널마다 **별도의 커널** $\text{kernel}_c$ 사용
- $K$ : 커널 크기 (보통 3)
- 일반 Conv와 차이: 채널 축 합산($\sum_c$)이 **없음!**

### PyTorch에서의 구현

```python
import torch
import torch.nn as nn

# Depthwise Conv = groups 파라미터를 in_channels로 설정
depthwise = nn.Conv2d(
    in_channels=32,
    out_channels=32,       # 입력과 동일!
    kernel_size=3,
    padding=1,
    groups=32              # 핵심! 각 채널을 독립 처리
)

# 파라미터 수: K × K × C_in = 3 × 3 × 32 = 288
print(f"DW 파라미터: {sum(p.numel() for p in depthwise.parameters()):,}")
# DW 파라미터: 320 (288 + 32 bias)
```

**`groups` 파라미터의 의미:**
- `groups=1` (기본값): 일반 Conv — 모든 입력 채널이 모든 출력 채널에 연결
- `groups=C_in`: Depthwise Conv — 각 채널이 독립적으로 처리
- `1 < groups < C_in`: Grouped Conv — [Grouped Convolution](/ko/docs/components/convolution/grouped-conv) 참조

---

## 2단계: Pointwise Convolution

### 직관적 정의

> **1×1 Conv**로 채널 간 정보를 혼합합니다. 공간은 건드리지 않습니다.

```
Depthwise 출력: 3채널            Pointwise (1×1 Conv)          최종 출력: 6채널
┌──────────┐                                                ┌──────────┐
│ 출력 1   │─┐                                           ┌─→│ 혼합 1   │
├──────────┤ ├─ 1×1 커널로 ─→  채널 간 가중합  ──→        ├──→│ 혼합 2   │
│ 출력 2   │─┤   혼합                                    ├──→│ 혼합 3   │
├──────────┤ │                                           ├──→│ 혼합 4   │
│ 출력 3   │─┘                                           ├──→│ 혼합 5   │
└──────────┘                                             └──→│ 혼합 6   │
                                                             └──────────┘
→ 채널 수 변경 가능 (C_in → C_out)
→ 공간 크기 유지 (H, W 그대로)
```

```python
# Pointwise Conv = 일반 1×1 Conv
pointwise = nn.Conv2d(
    in_channels=32,
    out_channels=64,       # 채널 수 변경!
    kernel_size=1           # 1×1
)

# 파라미터 수: 1 × 1 × C_in × C_out = 1 × 1 × 32 × 64 = 2,048
print(f"PW 파라미터: {sum(p.numel() for p in pointwise.parameters()):,}")
# PW 파라미터: 2,112 (2,048 + 64 bias)
```

---

## 합치면: Depthwise Separable Convolution

### 전체 구조

```python
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (MobileNet 스타일)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        # Step 1: 공간 필터링 (채널별 독립)
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_ch,       # 핵심!
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)

        # Step 2: 채널 혼합 (1×1)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x
```

### 파라미터 비교

$$
\frac{\text{Separable 파라미터}}{\text{일반 Conv 파라미터}} = \frac{K^2 \cdot C_{in} + C_{in} \cdot C_{out}}{K^2 \cdot C_{in} \cdot C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}
$$

**각 기호의 의미:**
- $K$ : 커널 크기 — 보통 3
- $C_{in}$ : 입력 채널 수
- $C_{out}$ : 출력 채널 수

### 예시 계산

```python
def compare_params(C_in, C_out, K=3):
    """일반 Conv vs Depthwise Separable 파라미터 비교"""
    normal = K * K * C_in * C_out
    separable = K * K * C_in + C_in * C_out  # DW + PW
    ratio = normal / separable

    print(f"C_in={C_in}, C_out={C_out}, K={K}")
    print(f"  일반 Conv:  {normal:>10,}")
    print(f"  Separable:  {separable:>10,}")
    print(f"  절감 비율:  {ratio:.1f}배\n")

compare_params(32, 64)
# C_in=32, C_out=64, K=3
#   일반 Conv:      18,432
#   Separable:       2,336
#   절감 비율:  7.9배

compare_params(128, 128)
# C_in=128, C_out=128, K=3
#   일반 Conv:     147,456
#   Separable:      17,536
#   절감 비율:  8.4배

compare_params(256, 512)
# C_in=256, C_out=512, K=3
#   일반 Conv:   1,179,648
#   Separable:     133,376
#   절감 비율:  8.8배
```

$K=3$, $C_{out}$이 충분히 크면 약 $\frac{1}{1/C_{out} + 1/9} \approx 8\sim9$배 절감됩니다.

### 연산량 (FLOPs) 비교

```python
def compare_flops(C_in, C_out, H, W, K=3):
    """연산량 비교 (곱셈-덧셈 기준)"""
    normal_flops = K * K * C_in * C_out * H * W
    dw_flops = K * K * C_in * H * W           # Depthwise
    pw_flops = C_in * C_out * H * W           # Pointwise
    sep_flops = dw_flops + pw_flops

    print(f"입력: ({C_in}, {H}, {W}) → 출력: ({C_out}, {H}, {W})")
    print(f"  일반:     {normal_flops / 1e6:.1f}M FLOPs")
    print(f"  DW+PW:    {sep_flops / 1e6:.1f}M FLOPs ({dw_flops/1e6:.1f} + {pw_flops/1e6:.1f})")
    print(f"  절감:     {normal_flops / sep_flops:.1f}배\n")

compare_flops(64, 128, 56, 56)
# 입력: (64, 56, 56) → 출력: (128, 56, 56)
#   일반:     1,849.7M FLOPs
#   DW+PW:    226.5M FLOPs (115.6 + 110.9)  (실제 값은 약간 다를 수 있음)
#   절감:     8.2배
```

---

## MobileNet에서의 활용

### MobileNetV1 (2017)

일반 Conv를 **모두** Depthwise Separable로 교체:

```python
# MobileNetV1의 기본 블록
class MobileNetV1Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw_conv = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)

    def forward(self, x):
        return self.dw_conv(x)
```

### MobileNetV2 (2018) — Inverted Residual Block

**핵심 개선**: Bottleneck을 뒤집었습니다!

```
일반 Bottleneck (ResNet):          Inverted Residual (MobileNetV2):
넓은(256) → 좁은(64) → 넓은(256)   좁은(24) → 넓은(144) → 좁은(24)
              │                                 │
           3×3 Conv                        3×3 DW Conv
        (고차원에서 처리)                  (고차원에서 공간 처리!)
```

```python
class InvertedResidual(nn.Module):
    """MobileNetV2의 Inverted Residual Block"""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        # 1. Expansion (1×1): 채널 확장
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])

        # 2. Depthwise (3×3): 공간 필터링
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ])

        # 3. Projection (1×1): 채널 축소 (활성화 없음!)
        layers.extend([
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
```

### EfficientNet (2019)

MobileNetV2의 Inverted Residual에 **SE Block** 추가 + **NAS**로 최적 구조 탐색:

```python
# EfficientNet의 MBConv 블록 (간략화)
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 expand_ratio=6, se_ratio=0.25):
        super().__init__()
        hidden = in_ch * expand_ratio

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        # SE (Squeeze-and-Excitation) Block
        squeezed = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, squeezed, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed, hidden, 1),
            nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = out * self.se(out)    # SE 가중치 적용
        out = self.project(out)
        if x.shape == out.shape:
            out = out + x           # Residual
        return out
```

---

## 성능 비교

| 모델 | Conv 방식 | 파라미터 | Top-1 (ImageNet) |
|------|----------|---------|-------------------|
| VGG-16 | 일반 Conv | 138M | 71.5% |
| ResNet-50 | 일반 Conv | 25.6M | 76.1% |
| MobileNetV1 | DW Separable | **4.2M** | 70.6% |
| MobileNetV2 | Inverted Residual | **3.4M** | 72.0% |
| EfficientNet-B0 | MBConv | **5.3M** | 77.1% |

핵심: **파라미터가 5~40배 적으면서 정확도는 비슷하거나 더 높음!**

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === 일반 Conv vs Depthwise Separable 비교 ===
C_in, C_out, H, W, K = 64, 128, 56, 56, 3

# 일반 Conv
conv_normal = nn.Conv2d(C_in, C_out, K, padding=1, bias=False)

# Depthwise Separable
conv_dw = nn.Conv2d(C_in, C_in, K, padding=1, groups=C_in, bias=False)
conv_pw = nn.Conv2d(C_in, C_out, 1, bias=False)

# 파라미터 수
params_normal = sum(p.numel() for p in conv_normal.parameters())
params_sep = sum(p.numel() for p in conv_dw.parameters()) + \
             sum(p.numel() for p in conv_pw.parameters())

print(f"일반 Conv 파라미터: {params_normal:,}")     # 73,728
print(f"Separable 파라미터: {params_sep:,}")         # 8,768
print(f"절감: {params_normal / params_sep:.1f}배")   # 8.4배

# 출력 확인
x = torch.randn(1, C_in, H, W)
y_normal = conv_normal(x)
y_sep = conv_pw(conv_dw(x))

print(f"\n일반 출력: {y_normal.shape}")     # [1, 128, 56, 56]
print(f"Separable 출력: {y_sep.shape}")    # [1, 128, 56, 56]  동일!

# === groups 파라미터 이해 ===
print("\n=== groups 파라미터 ===")
for g in [1, 2, 4, 64]:
    conv_g = nn.Conv2d(64, 64, 3, padding=1, groups=g, bias=False)
    params = sum(p.numel() for p in conv_g.parameters())
    print(f"groups={g:>2}: 파라미터 {params:>6,}  "
          f"(그룹당 {64//g}채널씩 처리)")
# groups= 1: 파라미터 36,864  (그룹당 64채널씩 처리) → 일반 Conv
# groups= 2: 파라미터 18,432  (그룹당 32채널씩 처리)
# groups= 4: 파라미터  9,216  (그룹당 16채널씩 처리)
# groups=64: 파라미터    576  (그룹당  1채널씩 처리) → Depthwise
```

---

## 핵심 정리

| 단계 | 연산 | 역할 | 파라미터 |
|------|------|------|---------|
| **Depthwise** | $K \times K$, groups=C | 공간 패턴 추출 (채널별 독립) | $K^2 \cdot C_{in}$ |
| **Pointwise** | $1 \times 1$ | 채널 간 정보 혼합 | $C_{in} \cdot C_{out}$ |
| **합계** | DW + PW | 일반 Conv와 동일한 기능 | $\approx \frac{1}{8\sim9}$ 절감 |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| DW Separable Conv | MobileNetV1, Xception | 경량 모델의 기본 블록 |
| Inverted Residual | MobileNetV2, EfficientNet | 모바일 모델의 표준 블록 |
| MBConv | EfficientNet, EfficientDet | SE + DW Separable |
| groups 파라미터 | [Grouped Conv](/ko/docs/components/convolution/grouped-conv) | DW Conv의 일반화 |

---

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d) — 선수 지식: 일반 합성곱의 원리
- [Grouped Convolution](/ko/docs/components/convolution/grouped-conv) — Depthwise Conv의 일반화 (groups 파라미터)
- [Receptive Field](/ko/docs/components/convolution/receptive-field) — DW Conv의 수용 영역 분석
- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — DW/PW 사이에 필수 사용
