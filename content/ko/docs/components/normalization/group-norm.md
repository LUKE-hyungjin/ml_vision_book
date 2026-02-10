---
title: "Group Normalization"
weight: 4
math: true
---

# Group Normalization

{{% hint info %}}
**선수지식**: [Batch Normalization](/ko/docs/components/normalization/batch-norm)
{{% /hint %}}

> **한 줄 요약**: Group Normalization은 채널을 **G개 그룹으로 나누어** 각 그룹 내에서 정규화하여, **배치 크기에 무관**하게 안정적으로 작동합니다.

## 왜 Group Normalization이 필요한가?

### 문제 상황: "배치 크기가 작으면 BatchNorm이 불안정합니다"

BatchNorm은 **배치 전체의 평균/분산**을 사용합니다:

```python
# BatchNorm의 통계 계산
mean = x.mean(dim=(0, 2, 3))  # B, H, W 축으로 평균
```

배치 크기가 클 때는 통계가 안정적이지만:

| 배치 크기 | 평균/분산 안정성 | 성능 |
|:---:|:---:|:---:|
| 32 | 안정적 | 좋음 |
| 16 | 약간 불안정 | 약간 저하 |
| 4 | 불안정 | 심각한 저하 |
| **1** | **계산 불가** (분산=0) | **사용 불가** |

**현실 문제**: Detection, Segmentation에서는 고해상도 이미지를 다루기 때문에 GPU 메모리 제약으로 **배치 크기가 1~4**인 경우가 많습니다.

### 해결: "배치 대신 채널 그룹으로 정규화하자!"

```
BatchNorm: 배치 축으로 통계 → 배치 크기 의존
LayerNorm: 모든 채널 축으로 통계 → 채널 간 차이 무시
GroupNorm: 채널을 그룹으로 나눠서 통계 → 배치 무관 + 채널 구조 유지!
```

![정규화 방법 비교](/images/components/normalization/ko/normalization-comparison.jpeg)

---

## 정규화 축 비교

입력 텐서 $(B, C, H, W)$에서 **어떤 축으로 평균/분산을 구하느냐**가 핵심:

```
입력: (B, C, H, W) — B=배치, C=채널, H=높이, W=너비

BatchNorm:     B, H, W 축 평균 → 채널별 1개 통계 (C개)
LayerNorm:     C, H, W 축 평균 → 샘플별 1개 통계 (B개)
InstanceNorm:  H, W 축 평균 → 샘플×채널별 1개 통계 (B×C개)
GroupNorm:     그룹 내 C/G, H, W 축 평균 → 샘플×그룹별 1개 통계 (B×G개)
```

```
    B
    ↑   ┌───────────────────┐
    │   │ C₁  C₂ │ C₃  C₄ │  ← 2개 그룹
    │   │        │         │
    │   │ 그룹1  │  그룹2  │
    │   └───────────────────┘
    └──→ C (×H×W)

BatchNorm:    ■를 B 방향으로 모아서 정규화 (세로줄)
GroupNorm:    ■를 그룹 내에서 정규화 (각 사각형 안)
LayerNorm:    ■를 전체 C에서 정규화 (가로줄 전체)
InstanceNorm: ■를 각 채널에서 정규화 (한 칸씩)
```

---

## 수식

### Group Normalization

채널 $C$를 $G$개 그룹으로 나누고, 각 그룹 내에서 정규화:

$$
\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_{b,g}}{\sqrt{\sigma_{b,g}^2 + \epsilon}}
$$

$$
y_{b,c,h,w} = \gamma_c \cdot \hat{x}_{b,c,h,w} + \beta_c
$$

여기서 $g = \lfloor c / (C/G) \rfloor$ (채널 $c$가 속한 그룹 인덱스)

**평균과 분산은 그룹 내에서 계산:**

$$
\mu_{b,g} = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in \text{group } g} \sum_{h,w} x_{b,c,h,w}
$$

$$
\sigma_{b,g}^2 = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in \text{group } g} \sum_{h,w} (x_{b,c,h,w} - \mu_{b,g})^2
$$

**각 기호의 의미:**
- $G$ : 그룹 수 (보통 32)
- $g$ : 채널 $c$가 속한 그룹 인덱스
- $\mu_{b,g}$ : 샘플 $b$, 그룹 $g$ 내의 평균 — **배치 축 없음!**
- $\gamma_c, \beta_c$ : 채널별 학습 가능한 스케일/시프트 (BatchNorm과 동일)

### 특수 케이스

| $G$ 값 | 동작 | 이름 |
|:---:|---|---|
| $G = 1$ | 모든 채널을 하나의 그룹 | **Layer Norm** |
| $G = C$ | 각 채널이 하나의 그룹 | **Instance Norm** |
| $1 < G < C$ | 채널을 그룹으로 나눔 | **Group Norm** |

---

## 구현

```python
import torch
import torch.nn as nn

# === PyTorch GroupNorm ===
gn = nn.GroupNorm(
    num_groups=32,        # G: 그룹 수
    num_channels=256,     # C: 채널 수 (C % G == 0이어야 함)
)

x = torch.randn(2, 256, 56, 56)  # (B, C, H, W) — 배치 크기 2도 OK!
y = gn(x)
print(f"입력: {x.shape} → 출력: {y.shape}")


# === 수동 구현 ===
class ManualGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        assert num_channels % num_groups == 0
        self.G = num_groups
        self.C = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        B, C, H, W = x.shape

        # 채널을 그룹으로 reshape
        x = x.reshape(B, self.G, C // self.G, H, W)

        # 그룹 내에서 평균/분산 (채널, H, W 축)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)

        # 정규화
        x = (x - mean) / torch.sqrt(var + self.eps)

        # 원래 shape으로 복원
        x = x.reshape(B, C, H, W)

        # 스케일/시프트
        return self.gamma[None, :, None, None] * x + self.beta[None, :, None, None]


# === 동작 확인 ===
mgn = ManualGroupNorm(32, 256)
y_manual = mgn(x)
print(f"수동 구현 출력: {y_manual.shape}")

# 배치 크기 1에서도 정상 작동!
x_single = torch.randn(1, 256, 56, 56)
y_single = gn(x_single)
print(f"배치 1: {y_single.shape}")  # 문제 없음!
```

---

## 그룹 수 선택

### 기본값: G=32

```python
# 대부분의 논문과 구현에서 G=32 사용
gn = nn.GroupNorm(32, num_channels)
```

### 그룹 수에 따른 성능

| 그룹 수 $G$ | 그룹당 채널 (C=256) | 특징 |
|:---:|:---:|---|
| 1 | 256 | = LayerNorm (채널 차이 무시) |
| 4 | 64 | 그룹이 너무 크면 채널 간 구분 약함 |
| **32** | **8** | **최적 (논문 기준)** |
| 64 | 4 | 그룹 내 통계 불안정 |
| 256 | 1 | = InstanceNorm (공간 정보만) |

논문 실험 결과 **$G=32$가 가장 안정적**이며, 채널 수가 변해도 $G=32$를 유지합니다.

---

## BatchNorm vs GroupNorm 비교

### 배치 크기에 따른 성능

```
ImageNet Top-1 Error (%) — ResNet-50

배치 크기    BatchNorm    GroupNorm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   32         23.6         24.1      ← BN 약간 우세
   16         23.7         24.1
    8         24.4         24.1      ← 비슷해짐
    4         26.0         24.2      ← GN 우세!
    2         28.2         24.2      ← GN 압도적!
    1          ---         24.3      ← BN 사용 불가
```

**핵심**: 배치 크기 ≥ 32이면 BN, < 16이면 GN이 유리합니다.

### 코드에서의 교체

```python
# BatchNorm → GroupNorm 교체는 매우 간단!

# 기존: BatchNorm
conv_bn = nn.Sequential(
    nn.Conv2d(256, 256, 3, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
)

# 변경: GroupNorm
conv_gn = nn.Sequential(
    nn.Conv2d(256, 256, 3, padding=1, bias=False),
    nn.GroupNorm(32, 256),    # G=32
    nn.ReLU(inplace=True),
)
```

### 주요 차이점

| | BatchNorm | GroupNorm |
|---|---|---|
| 정규화 축 | Batch, H, W | 그룹 내 C/G, H, W |
| 배치 의존 | O (배치 크기 ≥ 16 필요) | **X (배치 1도 OK)** |
| Train/Eval 차이 | O (running stats) | **X (동일 동작)** |
| 학습 파라미터 | $\gamma, \beta$ (C개씩) | $\gamma, \beta$ (C개씩) |
| 추가 버퍼 | running_mean, running_var | **없음** |

---

## 사용처

### Detection / Segmentation

고해상도 이미지로 인해 배치 크기가 작은 경우:

```python
# Mask R-CNN, FCOS 등에서 backbone의 BN → GN 교체
# detectron2 기본 설정
class FPNWithGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(32, 256),
        )
```

### DDPM / Diffusion 모델

U-Net의 정규화:

```python
# 대부분의 Diffusion 모델은 GroupNorm 사용
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.conv1(nn.SiLU()(self.norm1(x)))
        h = self.conv2(nn.SiLU()(self.norm2(h)))
        return x + h
```

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === 배치 크기에 따른 안정성 비교 ===
print("=== 배치 크기에 따른 통계 안정성 ===")

C = 64
bn = nn.BatchNorm2d(C)
gn = nn.GroupNorm(32, C)
bn.train()

for B in [32, 8, 2, 1]:
    x = torch.randn(B, C, 16, 16)
    try:
        y_bn = bn(x)
        bn_ok = "OK"
    except:
        bn_ok = "FAIL"
    y_gn = gn(x)
    print(f"B={B:>2}: BN={bn_ok}, GN=OK (var={y_gn.var().item():.3f})")

# === GroupNorm의 특수 케이스 ===
print("\n=== 특수 케이스 ===")
x = torch.randn(2, 64, 8, 8)

gn_1 = nn.GroupNorm(1, 64)      # G=1 → LayerNorm
gn_32 = nn.GroupNorm(32, 64)    # G=32 → GroupNorm
gn_64 = nn.GroupNorm(64, 64)    # G=64 → InstanceNorm

# LayerNorm과 비교
ln = nn.LayerNorm([64, 8, 8])

y_gn1 = gn_1(x)
y_ln = ln(x)
# G=1 GroupNorm ≈ LayerNorm (미세한 차이는 파라미터 shape 때문)

print(f"G=1 (≈LayerNorm):     평균={y_gn1.mean():.4f}, 분산={y_gn1.var():.4f}")
print(f"G=32 (GroupNorm):      평균={gn_32(x).mean():.4f}, 분산={gn_32(x).var():.4f}")
print(f"G=64 (≈InstanceNorm): 평균={gn_64(x).mean():.4f}, 분산={gn_64(x).var():.4f}")

# === Train/Eval 차이 없음 확인 ===
print("\n=== Train vs Eval ===")
gn = nn.GroupNorm(32, 64)
x = torch.randn(4, 64, 8, 8)

gn.train()
y_train = gn(x)

gn.eval()
y_eval = gn(x)

diff = (y_train - y_eval).abs().max().item()
print(f"Train vs Eval 최대 차이: {diff:.6f}")  # 0.000000 — 완전 동일!
```

---

## 핵심 정리

| 항목 | BatchNorm | GroupNorm | LayerNorm | InstanceNorm |
|------|-----------|-----------|-----------|--------------|
| 정규화 단위 | 채널 | **그룹** | 전체 | 채널 |
| 배치 의존 | O | **X** | X | X |
| 최적 사용처 | CNN (큰 배치) | **CNN (작은 배치)** | Transformer | Style Transfer |
| 그룹 수 $G$ | - | **32** (기본) | 1 | C |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| GroupNorm (G=32) | Mask R-CNN, FCOS | 작은 배치에서 안정적 |
| GroupNorm | DDPM U-Net, Stable Diffusion | Diffusion 모델의 표준 |
| GroupNorm | ResNeXt | Grouped Conv와 자연스럽게 결합 |

---

## 관련 콘텐츠

- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — 선수 지식: 배치 기반 정규화
- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — GroupNorm의 특수 케이스 (G=1)
- [Instance Normalization](/ko/docs/components/normalization/instance-norm) — GroupNorm의 특수 케이스 (G=C)
- [RMSNorm](/ko/docs/components/normalization/rms-norm) — LayerNorm의 간소화 버전
