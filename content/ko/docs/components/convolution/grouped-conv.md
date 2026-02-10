---
title: "Grouped Convolution"
weight: 8
math: true
---

# Grouped Convolution (그룹 합성곱)

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d)
{{% /hint %}}

> **한 줄 요약**: Grouped Convolution은 입력 채널을 **G개 그룹으로 나누어** 각 그룹을 독립적으로 처리하여, 파라미터와 연산량을 **G배** 줄이는 기법입니다.

## 왜 Grouped Convolution이 필요한가?

### 역사: GPU 메모리가 부족했습니다

Grouped Convolution은 AlexNet(2012)에서 **실용적인 이유**로 처음 사용되었습니다:

```python
# AlexNet의 문제:
# Conv2 레이어: 입력 96채널, 출력 256채널, 5×5 커널
# 파라미터: 5 × 5 × 96 × 256 = 614,400

# 2012년 GPU: GTX 580 (메모리 3GB)
# → 한 GPU에 모델 전체를 올릴 수 없었음!
```

**해결**: 채널을 반으로 나눠 2개 GPU에 분산

```
일반 Conv (groups=1):
  96채널 ───[5×5 Conv]───→ 256채널     파라미터: 614,400

Grouped Conv (groups=2):
  48채널 ───[5×5 Conv]───→ 128채널     GPU 1   파라미터: 153,600
  48채널 ───[5×5 Conv]───→ 128채널     GPU 2   파라미터: 153,600
  ─────────────────────────────────            합계: 307,200 (절반!)
```

### 놀라운 발견: 그룹이 있으면 성능이 더 좋다!

단순히 메모리 절약을 위한 것이었는데, 연구자들이 발견한 사실:

```
AlexNet의 학습된 필터를 관찰하면:

그룹 1 (GPU 1): 주로 흑백 패턴, 에지, 텍스처 학습
그룹 2 (GPU 2): 주로 컬러 패턴, 색상 변화 학습

→ 자연스럽게 역할 분담이 일어남!
```

이후 ResNeXt 논문에서 **Grouped Conv가 더 효율적**임을 체계적으로 증명했습니다.

---

## 수식

### 일반 Conv (복습)

$$
y_j = \sum_{i=0}^{C_{in}-1} w_{j,i} * x_i
$$

**모든 입력 채널** $i$가 **모든 출력 채널** $j$에 연결됩니다.

### Grouped Conv

입력 $C_{in}$개 채널을 $G$개 그룹으로 나누면:

$$
y_j^{(g)} = \sum_{i=0}^{C_{in}/G - 1} w_{j,i}^{(g)} * x_i^{(g)}, \quad g = 0, 1, ..., G-1
$$

**각 기호의 의미:**
- $G$ : 그룹 수
- $g$ : 그룹 인덱스
- $x^{(g)}$ : $g$번째 그룹의 입력 (채널 $C_{in}/G$개)
- $w^{(g)}$ : $g$번째 그룹의 커널
- $y^{(g)}$ : $g$번째 그룹의 출력 (채널 $C_{out}/G$개)

### 파라미터 비교

| | 일반 Conv | Grouped Conv |
|---|---|---|
| 커널 크기 | $K \times K \times C_{in} \times C_{out}$ | $K \times K \times \frac{C_{in}}{G} \times C_{out}$ |
| **파라미터 수** | $K^2 \cdot C_{in} \cdot C_{out}$ | $\frac{K^2 \cdot C_{in} \cdot C_{out}}{G}$ |
| **절감 비율** | 1× | **$G$배 절감** |

### 직관적 이해

```
일반 Conv (groups=1):                    Grouped Conv (groups=2):

입력 채널:     출력 채널:                입력 채널:     출력 채널:
  ch0 ──┬──→ out0                         ch0 ──┬──→ out0    (그룹 1)
  ch1 ──┤──→ out1                         ch1 ──┘──→ out1
  ch2 ──┤──→ out2          →
  ch3 ──┘──→ out3                         ch2 ──┬──→ out2    (그룹 2)
                                          ch3 ──┘──→ out3
모든 입력이 모든 출력에                   그룹 내에서만 연결!
연결 (완전 연결)                         그룹 간에는 독립
```

---

## PyTorch에서의 사용

```python
import torch
import torch.nn as nn

# === groups 파라미터 ===
# groups=1: 일반 Conv (기본값)
conv_g1 = nn.Conv2d(64, 128, 3, padding=1, groups=1, bias=False)

# groups=2: 2개 그룹
conv_g2 = nn.Conv2d(64, 128, 3, padding=1, groups=2, bias=False)

# groups=4: 4개 그룹
conv_g4 = nn.Conv2d(64, 128, 3, padding=1, groups=4, bias=False)

# groups=64: Depthwise Conv (각 채널이 하나의 그룹)
conv_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)

# 파라미터 비교
for name, conv in [('g=1', conv_g1), ('g=2', conv_g2),
                    ('g=4', conv_g4), ('g=64(DW)', conv_dw)]:
    params = sum(p.numel() for p in conv.parameters())
    print(f"{name:>10}: {params:>8,} 파라미터  "
          f"커널 shape: {list(conv.weight.shape)}")

# g=1:   73,728 파라미터  커널 shape: [128, 64, 3, 3]
# g=2:   36,864 파라미터  커널 shape: [128, 32, 3, 3]  ← C_in/G = 32
# g=4:   18,432 파라미터  커널 shape: [128, 16, 3, 3]
# g=64(DW):  576 파라미터  커널 shape: [64, 1, 3, 3]   ← Depthwise!
```

**제약 조건**: `C_in`과 `C_out`이 모두 `groups`로 나누어떨어져야 합니다.

---

## Grouped Conv의 스펙트럼

`groups` 값에 따라 다양한 Conv가 됩니다:

```
groups=1          groups=G          groups=C_in
   │                 │                 │
   ▼                 ▼                 ▼
일반 Conv    →   Grouped Conv  →   Depthwise Conv
(완전 연결)      (그룹별 연결)      (채널별 독립)

파라미터: 많음  ←────────────→   적음
표현력: 높음    ←────────────→   낮음
```

| groups | 이름 | 파라미터 (C=256, K=3) |
|:---:|---|---:|
| 1 | 일반 Conv | 589,824 |
| 2 | AlexNet 스타일 | 294,912 |
| 32 | ResNeXt 스타일 | 18,432 |
| 256 | Depthwise Conv | 2,304 |

---

## ResNeXt: Grouped Conv의 체계적 활용

### 핵심 아이디어: "넓게 가자" (Cardinality)

ResNet의 bottleneck을 Grouped Conv로 대체합니다:

```
ResNet Bottleneck:              ResNeXt Bottleneck (C=32):

256 ──[1×1, 64]──→ 64          256 ──[1×1, 128]──→ 128
 64 ──[3×3, 64]──→ 64          128 ──[3×3, 128, G=32]──→ 128  ← Grouped!
 64 ──[1×1, 256]──→ 256         128 ──[1×1, 256]──→ 256
        +                              +
       256                             256
```

```python
class ResNeXtBlock(nn.Module):
    """ResNeXt의 Bottleneck Block (C=32, d=4)"""
    def __init__(self, in_ch, cardinality=32, bottleneck_width=4):
        super().__init__()
        # 그룹 수 × 그룹당 너비 = 중간 채널
        mid_ch = cardinality * bottleneck_width  # 32 × 4 = 128

        self.conv = nn.Sequential(
            # 1×1: 채널 변환
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            # 3×3: Grouped Conv (핵심!)
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1,
                      groups=cardinality, bias=False),   # G=32
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            # 1×1: 채널 복원
            nn.Conv2d(mid_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))
```

### Cardinality vs Depth vs Width

ResNeXt 논문의 핵심 발견:

```
같은 파라미터 예산에서 뭘 늘려야 정확도가 올라갈까?

1. Depth (레이어 수) 늘리기 → 한계 있음 (gradient 문제)
2. Width (채널 수) 늘리기 → 효과 있지만 비효율적
3. Cardinality (그룹 수) 늘리기 → 가장 효과적! ✓
```

| 모델 | 파라미터 | Top-1 (ImageNet) |
|------|---------|-------------------|
| ResNet-50 | 25.6M | 75.3% |
| ResNet-101 | 44.5M | 76.4% |
| ResNeXt-50 (32×4d) | 25.0M | **77.8%** |
| ResNeXt-101 (32×4d) | 44.2M | **78.8%** |

**같은 파라미터에서 ResNeXt가 더 높은 정확도!**

---

## ShuffleNet: Channel Shuffle

### 문제: 그룹 간 정보 단절

Grouped Conv의 문제점: 그룹 간에 정보가 흐르지 않습니다.

```
그룹 1: ch0, ch1 → out0, out1     그룹 1의 출력은
그룹 2: ch2, ch3 → out2, out3     그룹 2의 정보를 전혀 모름!
```

### 해결: Channel Shuffle

출력 채널을 **섞어서** 그룹 간 정보를 교환합니다:

```
Grouped Conv 출력:     Channel Shuffle:      다음 Grouped Conv:
┌─ 그룹1: A B ─┐     ┌─ A C ─┐            ┌─ 그룹1: A C ─┐
└─ 그룹2: C D ─┘     └─ B D ─┘            └─ 그룹2: B D ─┘

→ 다음 레이어에서 다른 그룹의 정보를 볼 수 있음!
```

```python
def channel_shuffle(x, groups):
    """채널을 그룹 간에 섞는 연산"""
    B, C, H, W = x.shape
    # (B, C, H, W) → (B, G, C//G, H, W) → (B, C//G, G, H, W) → (B, C, H, W)
    x = x.reshape(B, groups, C // groups, H, W)
    x = x.transpose(1, 2)  # 그룹과 채널 축 교환
    x = x.reshape(B, C, H, W)
    return x

class ShuffleBlock(nn.Module):
    """ShuffleNet의 기본 블록"""
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        mid_ch = out_ch // 4
        self.groups = groups

        # 1×1 Grouped Conv
        self.gconv1 = nn.Conv2d(in_ch, mid_ch, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        # 3×3 Depthwise Conv
        self.dwconv = nn.Conv2d(mid_ch, mid_ch, 3, padding=1,
                                 groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        # 1×1 Grouped Conv
        self.gconv2 = nn.Conv2d(mid_ch, out_ch, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.gconv1(x)))
        out = channel_shuffle(out, self.groups)    # ← 핵심!
        out = self.bn2(self.dwconv(out))
        out = self.bn3(self.gconv2(out))
        return self.relu(out + x)
```

---

## Grouped Conv vs Depthwise Separable Conv

| | Grouped Conv | Depthwise Separable |
|---|---|---|
| 구조 | 채널을 G개 그룹으로 나누어 Conv | DW(G=C) + PW(1×1) |
| 그룹 수 | 자유 (1~C) | 항상 C |
| 채널 혼합 | 그룹 내에서만 | PW에서 전체 혼합 |
| 대표 모델 | AlexNet, ResNeXt | MobileNet, EfficientNet |
| 관계 | Depthwise Conv ⊂ Grouped Conv | Grouped Conv의 극단적 케이스 |

**관계도:**

```
일반 Conv (G=1)
    │
    ├─ Grouped Conv (1 < G < C)  ← ResNeXt
    │       │
    │       └─ Depthwise Conv (G=C)  ← MobileNet의 DW 부분
    │               │
    │               └─ + Pointwise = Depthwise Separable Conv
    │
    └─ 1×1 Conv (채널만 혼합)  ← Pointwise Conv
```

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === Groups에 따른 파라미터와 연산량 ===
print("=== Groups에 따른 변화 ===")
C_in, C_out, K = 256, 256, 3

for G in [1, 2, 4, 8, 16, 32, 64, 256]:
    if C_in % G == 0 and C_out % G == 0:
        conv = nn.Conv2d(C_in, C_out, K, padding=1, groups=G, bias=False)
        params = sum(p.numel() for p in conv.parameters())
        print(f"G={G:>3}: 파라미터 {params:>8,}  "
              f"그룹당 채널: {C_in//G}→{C_out//G}  "
              f"커널: {list(conv.weight.shape)}")

# G=  1: 파라미터  589,824  그룹당 채널: 256→256  커널: [256, 256, 3, 3]
# G=  2: 파라미터  294,912  그룹당 채널: 128→128  커널: [256, 128, 3, 3]
# G=  4: 파라미터  147,456  그룹당 채널: 64→64    커널: [256, 64, 3, 3]
# G= 32: 파라미터   18,432  그룹당 채널: 8→8      커널: [256, 8, 3, 3]
# G=256: 파라미터    2,304  그룹당 채널: 1→1      커널: [256, 1, 3, 3]

# === Channel Shuffle 확인 ===
print("\n=== Channel Shuffle ===")
x = torch.arange(8).float().reshape(1, 8, 1, 1)
print(f"입력 채널: {x.squeeze().tolist()}")

shuffled = channel_shuffle(x, groups=2)
print(f"Shuffle(G=2): {shuffled.squeeze().tolist()}")

shuffled = channel_shuffle(x, groups=4)
print(f"Shuffle(G=4): {shuffled.squeeze().tolist()}")

# 입력 채널: [0, 1, 2, 3, 4, 5, 6, 7]
# Shuffle(G=2): [0, 4, 1, 5, 2, 6, 3, 7]  ← 그룹 교차!
# Shuffle(G=4): [0, 2, 4, 6, 1, 3, 5, 7]

# === ResNet vs ResNeXt 파라미터 비교 ===
print("\n=== ResNet Bottleneck vs ResNeXt Block ===")

# ResNet-50 Bottleneck: 256→64→64→256
resnet_block = nn.Sequential(
    nn.Conv2d(256, 64, 1, bias=False),
    nn.Conv2d(64, 64, 3, padding=1, bias=False),
    nn.Conv2d(64, 256, 1, bias=False),
)

# ResNeXt-50 (32×4d): 256→128→128(G=32)→256
resnext_block = nn.Sequential(
    nn.Conv2d(256, 128, 1, bias=False),
    nn.Conv2d(128, 128, 3, padding=1, groups=32, bias=False),  # G=32
    nn.Conv2d(128, 256, 1, bias=False),
)

p_resnet = sum(p.numel() for p in resnet_block.parameters())
p_resnext = sum(p.numel() for p in resnext_block.parameters())

print(f"ResNet Bottleneck:  {p_resnet:>8,} 파라미터")
print(f"ResNeXt (32×4d):    {p_resnext:>8,} 파라미터")
print(f"차이: {abs(p_resnet - p_resnext):,}")
# 비슷한 파라미터 수에서 ResNeXt가 더 높은 정확도!
```

---

## 핵심 정리

| 항목 | 일반 Conv | Grouped Conv | Depthwise Conv |
|------|----------|-------------|----------------|
| Groups | 1 | G | C_in |
| 그룹당 입력 채널 | C_in | C_in/G | 1 |
| 파라미터 | $K^2 C_{in} C_{out}$ | $\frac{K^2 C_{in} C_{out}}{G}$ | $K^2 C_{in}$ |
| 채널 간 혼합 | 전체 | 그룹 내 | 없음 |
| 대표 모델 | VGG, ResNet | ResNeXt | MobileNet |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| groups=2 | AlexNet | 최초의 Grouped Conv (GPU 분산) |
| groups=32 | ResNeXt | Cardinality로 성능 향상 |
| Channel Shuffle | ShuffleNet | 그룹 간 정보 교환 |
| groups=C | MobileNet | [Depthwise Separable](/ko/docs/components/convolution/depthwise-separable-conv)의 DW 부분 |

---

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d) — 선수 지식: 일반 합성곱
- [Depthwise Separable Conv](/ko/docs/components/convolution/depthwise-separable-conv) — Grouped Conv의 극단적 케이스 (G=C)
- [Receptive Field](/ko/docs/components/convolution/receptive-field) — 그룹 수에 따른 RF 변화
