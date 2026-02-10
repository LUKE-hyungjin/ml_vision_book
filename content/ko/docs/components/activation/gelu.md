---
title: "GELU"
weight: 4
math: true
---

# GELU (Gaussian Error Linear Unit)

{{% hint info %}}
**선수지식**: [ReLU](/ko/docs/components/activation/relu) | [확률분포](/ko/docs/math/probability/distribution)
{{% /hint %}}

> **한 줄 요약**: GELU는 입력에 **표준정규분포의 CDF를 곱하여**, 입력 크기에 따라 **확률적으로 뉴런을 활성화**하는 Transformer의 표준 활성화 함수입니다.

![활성화 함수 그래프 비교](/images/components/activation/ko/activation-comparison.png)

## 왜 GELU가 필요한가?

### 문제 상황: "ReLU의 경계가 너무 딱딱합니다"

```python
# ReLU: x=0을 기준으로 딱 잘림
# x = 0.01 → 출력 0.01 (활성화)
# x = -0.01 → 출력 0.00 (비활성화)
# 0.02 차이인데 하나는 살고 하나는 죽음!

# 문제: Transformer처럼 부드러운 표현이 필요한 모델에서
# 이런 "칼로 자르는" 비선형성은 최적이 아님
```

### 해결: "입력 크기에 따라 확률적으로 활성화하자!"

ReLU는 0이냐 아니냐의 **이진 결정**이지만, GELU는 **부드러운 확률적 결정**입니다:

```
ReLU:  "x > 0? → 살린다 / 죽인다"        (0 또는 1, 칼같이)
GELU:  "x가 클수록 살릴 확률 높다"        (0~1 사이, 부드럽게)

       x = 3.0:  GELU ≈ 3.0 × 0.999 ≈ 3.0   (거의 확실히 활성화)
       x = 0.0:  GELU = 0.0 × 0.5   = 0.0    (50% 확률로 활성화)
       x = -3.0: GELU ≈ -3.0 × 0.001 ≈ 0.0   (거의 확실히 비활성화)
```

Dropout에 비유하면:
- **Dropout**: 모든 뉴런을 **같은 확률 p**로 무작위 비활성화
- **GELU**: 각 뉴런을 **입력 크기에 비례한 확률**로 비활성화

---

## 수식

### GELU

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

여기서 $\Phi(x)$는 **표준정규분포의 CDF** (누적분포함수):

$$
\Phi(x) = P(X \leq x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

**각 기호의 의미:**
- $x$ : 입력값
- $\Phi(x)$ : $x$ 이하의 확률 — "이 입력이 얼마나 큰가?"의 확률적 측정
- $\text{erf}$ : 오차 함수 (error function)
- $x \cdot \Phi(x)$ : 입력에 "활성화 확률"을 곱함

### 직관적 이해

```
GELU(x) = x × Φ(x) = x × P(X ≤ x)

  x = +∞:  Φ(+∞) = 1.0  → GELU(x) = x × 1.0 = x     (그대로 통과)
  x = 0:   Φ(0)  = 0.5  → GELU(0) = 0 × 0.5 = 0      (절반만 통과)
  x = -∞:  Φ(-∞) = 0.0  → GELU(x) = x × 0.0 = 0      (차단)
```

### 근사 공식

정확한 GELU는 $\text{erf}$ 함수가 필요하지만, 빠른 근사가 가능합니다:

$$
\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]
$$

또는 더 간단한 Sigmoid 근사:

$$
\text{GELU}(x) \approx x \cdot \sigma(1.702x)
$$

### ReLU vs GELU 비교

| 특성 | ReLU | GELU |
|------|------|------|
| 수식 | $\max(0, x)$ | $x \cdot \Phi(x)$ |
| $x=0$에서 | 미분 불연속 | **매끄러움** |
| 음수 영역 | 정확히 0 | **약간 음수** (비단조) |
| 미분 | 0 또는 1 | 연속적 |
| 계산 비용 | 매우 빠름 | 약간 느림 |

GELU의 독특한 성질: $x \approx -0.17$에서 최솟값 ≈ $-0.17$을 가집니다. 즉, 약간의 음수 출력을 허용합니다.

---

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === PyTorch GELU ===
gelu = nn.GELU()
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"입력: {x.tolist()}")
print(f"GELU: {gelu(x).tolist()}")
# GELU: [-0.0036, -0.1587, 0.0, 0.8413, 2.9964]


# === 근사 모드 ===
gelu_tanh = nn.GELU(approximate='tanh')  # tanh 근사 (더 빠름)
gelu_none = nn.GELU(approximate='none')  # 정확한 erf (기본값)


# === 수동 구현 ===
def gelu_exact(x):
    """정확한 GELU (erf 사용)"""
    return x * 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def gelu_tanh_approx(x):
    """Tanh 근사 GELU (GPT-2 스타일)"""
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)
    ))

def gelu_sigmoid_approx(x):
    """Sigmoid 근사 GELU"""
    return x * torch.sigmoid(1.702 * x)


# === Transformer FFN에서의 사용 ===
class TransformerFFN(nn.Module):
    """Transformer Feed-Forward Network"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

# ViT, GPT, BERT 모두 이 패턴!
ffn = TransformerFFN(768)
x = torch.randn(4, 196, 768)  # (B, T, D)
y = ffn(x)  # (4, 196, 768)
```

---

## 사용 모델

### GELU를 사용하는 주요 모델

| 모델 | 연도 | GELU 근사 | 비고 |
|------|------|-----------|------|
| **BERT** | 2018 | tanh 근사 | GELU를 처음 널리 사용 |
| **GPT-2** | 2019 | tanh 근사 | |
| **ViT** | 2020 | exact (erf) | Vision Transformer |
| **GPT-3** | 2020 | tanh 근사 | |
| **DeiT** | 2021 | exact | |
| **Swin Transformer** | 2021 | exact | |

### GELU → SiLU 전환 트렌드

2023년 이후 LLM에서는 GELU 대신 **Swish/SiLU**로 전환하는 추세입니다:

| 모델 | 활성화 | 이유 |
|------|--------|------|
| BERT, GPT-2, ViT | GELU | 초기 Transformer 표준 |
| LLaMA, Qwen, Mistral | **SiLU** | 더 단순, SwiGLU와 조합 |

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === ReLU vs GELU 비교 ===
print("=== ReLU vs GELU 비교 ===")
x = torch.linspace(-3, 3, 13)
relu_y = F.relu(x)
gelu_y = F.gelu(x)

for xi, ry, gy in zip(x, relu_y, gelu_y):
    print(f"x={xi:>5.1f}: ReLU={ry:>6.3f}, GELU={gy:>6.3f}")
# 차이점: GELU는 음수에서도 약간의 값이 있고, 양수에서도 약간 줄어듦


# === 근사 정확도 비교 ===
print("\n=== 근사 정확도 ===")
x = torch.randn(10000)
exact = F.gelu(x, approximate='none')
tanh_approx = F.gelu(x, approximate='tanh')
diff = (exact - tanh_approx).abs()
print(f"tanh 근사 최대 오차: {diff.max():.6f}")
print(f"tanh 근사 평균 오차: {diff.mean():.6f}")
# 매우 작은 오차 → 실용적으로 동일


# === GELU의 비단조성 확인 ===
print("\n=== GELU 최솟값 ===")
x = torch.linspace(-1, 0, 1000, requires_grad=True)
y = F.gelu(x)
min_idx = y.argmin()
print(f"GELU 최솟값: x={x[min_idx].item():.3f}, y={y[min_idx].item():.4f}")
# 약 x=-0.17에서 y≈-0.17 → 약간의 음수 허용!


# === Transformer FFN 파라미터 수 ===
print("\n=== Transformer FFN ===")
dim = 768
ffn = nn.Sequential(
    nn.Linear(dim, dim * 4),
    nn.GELU(),
    nn.Linear(dim * 4, dim),
)
params = sum(p.numel() for p in ffn.parameters())
print(f"FFN 파라미터: {params:,}")
# Linear(768→3072): 768×3072 + 3072 = 2,362,368
# GELU: 파라미터 0개
# Linear(3072→768): 3072×768 + 768 = 2,360,064
# 총: 4,722,432
```

---

## 핵심 정리

| 항목 | 내용 |
|------|------|
| **수식** | $x \cdot \Phi(x)$ |
| **출력 범위** | $[\approx -0.17, \infty)$ |
| **핵심 아이디어** | 확률적 뉴런 활성화 |
| **ReLU 대비 장점** | 매끄러움, 음수 약간 허용 |
| **계산 비용** | ReLU보다 약간 느림 |
| **주 사용처** | Transformer (ViT, GPT, BERT) |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| Transformer FFN | [ViT](/ko/docs/architecture/transformer/vit), GPT, BERT | 표준 활성화 함수 |
| tanh 근사 | GPT-2, BERT 구현 | 계산 속도 최적화 |
| GELU → SiLU 전환 | LLaMA, Qwen | 최신 LLM 트렌드 |

---

## 관련 콘텐츠

- [ReLU](/ko/docs/components/activation/relu) — 선수 지식: 가장 기본적인 활성화 함수
- [확률분포](/ko/docs/math/probability/distribution) — 선수 지식: 정규분포 CDF
- [Swish/SiLU](/ko/docs/components/activation/swish-silu) — GELU의 대안 (최신 LLM)
- [Sigmoid](/ko/docs/components/activation/sigmoid) — GELU의 sigmoid 근사와 관련
