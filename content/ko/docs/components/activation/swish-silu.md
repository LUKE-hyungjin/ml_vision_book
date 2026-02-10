---
title: "Swish / SiLU"
weight: 5
math: true
---

# Swish / SiLU (Sigmoid Linear Unit)

{{% hint info %}}
**선수지식**: [Sigmoid](/ko/docs/components/activation/sigmoid)
{{% /hint %}}

> **한 줄 요약**: Swish (= SiLU)는 입력에 **자기 자신의 Sigmoid를 곱한** $x \cdot \sigma(x)$ 형태로, ReLU보다 부드럽고 GELU와 비슷한 성능을 보이는 **최신 LLM의 표준 활성화 함수**입니다.

![활성화 함수 그래프 비교](/images/components/activation/ko/activation-comparison.png)

## 왜 Swish/SiLU가 필요한가?

### 문제 상황: "GELU는 좋지만 수식이 복잡합니다"

```python
# GELU: x × Φ(x)  ← 정규분포 CDF 필요 (erf 또는 tanh 근사)
# 구현이 복잡하고, 하드웨어 최적화가 어려움

# 필요한 것: GELU만큼 좋지만 더 단순한 수식
```

### 해결: "x에 sigmoid(x)를 곱하자!"

```
Swish(x) = x × σ(x)     ← sigmoid 하나면 됨!

ReLU:   max(0, x)        → 불연속, 음수 = 0
GELU:   x × Φ(x)         → 매끄러움, erf 필요
Swish:  x × σ(x)         → 매끄러움, sigmoid만 필요!
```

**Google Brain의 자동 탐색 결과**: 2017년 Google이 강화학습으로 활성화 함수를 자동 탐색했을 때, $x \cdot \sigma(\beta x)$가 발견되었습니다. 이를 **Swish**라 명명했고, $\beta=1$인 특수 경우를 PyTorch에서는 **SiLU**라 부릅니다.

---

## 수식

### Swish / SiLU

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

### 일반화된 Swish

$$
\text{Swish}_\beta(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}
$$

**각 기호의 의미:**
- $x$ : 입력값
- $\sigma(x)$ : Sigmoid 함수 — 0~1 사이의 "게이트 값"
- $\beta$ : 조절 파라미터 (보통 $\beta=1$로 고정)
- $x \cdot \sigma(x)$ : 입력에 자기 자신의 "게이트"를 곱함

### β에 따른 변화

$$
\beta \to \infty: \quad \text{Swish}_\beta(x) \to \text{ReLU}(x)
$$
$$
\beta = 0: \quad \text{Swish}_0(x) = x/2 \quad \text{(선형)}
$$
$$
\beta = 1: \quad \text{SiLU}(x) \quad \text{(표준)}
$$

```
β → ∞:  σ(βx) → step function → Swish → ReLU
β = 1:   SiLU (표준, 대부분의 모델)
β → 0:   σ(βx) → 0.5 → Swish → x/2 (선형)
```

### 미분

$$
\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))
$$

- 양수 영역: 미분값 > 1이 될 수 있음 (ReLU의 1과 달리)
- $x \approx -1.28$에서 미분 = 0 (최솟값)

### GELU와의 비교

| | GELU | SiLU (Swish) |
|---|------|------|
| 수식 | $x \cdot \Phi(x)$ | $x \cdot \sigma(x)$ |
| "게이트" | 정규분포 CDF | Sigmoid |
| 최솟값 | $\approx -0.17$ | $\approx -0.28$ |
| 최솟값 위치 | $x \approx -0.17$ | $x \approx -1.28$ |
| 계산 | erf 또는 tanh 근사 | sigmoid만 |
| 성능 | 거의 동일 | 거의 동일 |

두 함수의 그래프는 거의 겹칩니다. 실질적인 차이는 미미합니다.

---

## SwiGLU: 최신 LLM의 표준

### GLU (Gated Linear Unit)

```python
# GLU: 입력의 절반을 게이트로 사용
def glu(x, W, V, b, c):
    return (W @ x + b) * sigmoid(V @ x + c)
```

### SwiGLU = Swish + GLU

$$
\text{SwiGLU}(x) = \text{SiLU}(W_1 x) \otimes (W_2 x)
$$

```python
class SwiGLU(nn.Module):
    """LLaMA, Qwen, Mistral의 FFN"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        #              ^^^^^^^^^^^^^^^^     ^^^^^^^^
        #              SiLU(gate)     ×     up projection
```

**왜 SwiGLU가 좋은가?**

| FFN 방식 | 수식 | 사용 모델 |
|----------|------|----------|
| 표준 FFN | $W_2 \cdot \text{GELU}(W_1 x)$ | BERT, GPT-2, ViT |
| SwiGLU FFN | $W_2 \cdot (\text{SiLU}(W_1 x) \otimes W_3 x)$ | **LLaMA, Qwen, Mistral** |

SwiGLU는 파라미터가 50% 더 많지만 ($W_3$ 추가), 같은 연산량에서 **더 좋은 성능**을 보여 최신 LLM의 표준이 되었습니다.

---

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === PyTorch SiLU ===
silu = nn.SiLU()
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"입력: {x.tolist()}")
print(f"SiLU: {silu(x).tolist()}")
# SiLU: [-0.1423, -0.2689, 0.0, 0.7311, 2.8577]


# === 함수형 사용 ===
y = F.silu(x)  # nn.SiLU()과 동일


# === 수동 구현 ===
def manual_silu(x):
    return x * torch.sigmoid(x)

def manual_swish(x, beta=1.0):
    return x * torch.sigmoid(beta * x)


# === LLaMA 스타일 SwiGLU FFN ===
class LLaMAFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # 2/3 × 4d ≈ 2.67d (LLaMA 스타일)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

ffn = LLaMAFFN(4096)
x = torch.randn(1, 2048, 4096)  # (B, T, D)
y = ffn(x)
print(f"\nLLaMA FFN: {x.shape} → {y.shape}")


# === EfficientNet 스타일 (MBConv에서 SiLU 사용) ===
class MBConvBlock(nn.Module):
    """EfficientNet의 기본 블록 (간략화)"""
    def __init__(self, in_ch, out_ch, expand_ratio=6):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.dw = nn.Conv2d(mid_ch, mid_ch, 3, padding=1,
                            groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.silu(self.bn1(self.expand(x)))
        out = self.silu(self.bn2(self.dw(out)))
        out = self.bn3(self.project(out))
        return out
```

---

## 사용 모델

| 모델 | 연도 | 활성화 | 구조 |
|------|------|--------|------|
| **EfficientNet** | 2019 | Swish (SiLU) | MBConv 블록 |
| **EfficientNetV2** | 2021 | SiLU | Fused-MBConv |
| **LLaMA 1/2/3** | 2023-24 | SiLU + SwiGLU | FFN |
| **Qwen 1/2** | 2023-24 | SiLU + SwiGLU | FFN |
| **Mistral / Mixtral** | 2023-24 | SiLU + SwiGLU | FFN |
| **Gemma** | 2024 | GeGLU (GELU+GLU) | FFN |

---

## 코드로 확인하기

```python
import torch
import torch.nn.functional as F

# === ReLU vs GELU vs SiLU 비교 ===
print("=== ReLU vs GELU vs SiLU ===")
x = torch.linspace(-3, 3, 13)
for xi in x:
    r = F.relu(xi).item()
    g = F.gelu(xi).item()
    s = F.silu(xi).item()
    print(f"x={xi:>5.1f}: ReLU={r:>6.3f}, GELU={g:>6.3f}, SiLU={s:>6.3f}")


# === β에 따른 Swish 변화 ===
print("\n=== Swish β 변화 ===")
x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
for beta in [0.1, 0.5, 1.0, 2.0, 10.0]:
    y = x * torch.sigmoid(beta * x)
    vals = [f"{v:.3f}" for v in y.tolist()]
    print(f"β={beta:>4.1f}: {vals}")
# β 커지면 → ReLU에 가까워짐


# === SiLU의 비단조성 (음수 영역) ===
print("\n=== SiLU 최솟값 ===")
x = torch.linspace(-3, 0, 1000)
y = F.silu(x)
min_idx = y.argmin()
print(f"SiLU 최솟값: x={x[min_idx].item():.3f}, y={y[min_idx].item():.4f}")
# 약 x=-1.28에서 y≈-0.28


# === GELU vs SiLU 차이 ===
print("\n=== GELU vs SiLU 차이 ===")
x = torch.randn(100000)
gelu_out = F.gelu(x)
silu_out = F.silu(x)
diff = (gelu_out - silu_out).abs()
print(f"평균 차이: {diff.mean():.4f}")
print(f"최대 차이: {diff.max():.4f}")
# 차이가 매우 작음 → 실질적으로 비슷


# === SwiGLU vs 표준 FFN 파라미터 비교 ===
print("\n=== FFN 파라미터 비교 ===")
dim = 4096
hidden = dim * 4  # 표준 FFN

# 표준 FFN: W1(dim→4dim) + W2(4dim→dim)
standard_params = dim * hidden + hidden * dim
# SwiGLU: W1(dim→h) + W2(h→dim) + W3(dim→h), h = 2/3 × 4dim
h = int(dim * 8 / 3)
swiglu_params = dim * h * 2 + h * dim  # W1 + W3 + W2

print(f"표준 FFN:  {standard_params:>12,}  (hidden={hidden})")
print(f"SwiGLU:    {swiglu_params:>12,}  (hidden={h})")
print(f"비율:      {swiglu_params / standard_params:.2f}x")
```

---

## 핵심 정리

| 항목 | 내용 |
|------|------|
| **수식** | $x \cdot \sigma(x)$ |
| **출력 범위** | $[\approx -0.28, \infty)$ |
| **핵심 아이디어** | 자기 게이팅 (self-gating) |
| **GELU 대비** | 더 단순한 수식, 비슷한 성능 |
| **SwiGLU** | SiLU + GLU 조합, 최신 LLM 표준 |
| **주 사용처** | EfficientNet, LLaMA, Qwen, Mistral |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| SiLU in CNN | EfficientNet | MBConv의 활성화 |
| SwiGLU FFN | LLaMA, Qwen, Mistral | 최신 LLM의 FFN 표준 |
| β 조절 | Neural Architecture Search | 활성화 함수 자동 탐색 |

---

## 관련 콘텐츠

- [Sigmoid](/ko/docs/components/activation/sigmoid) — 선수 지식: SiLU = x × sigmoid(x)
- [ReLU](/ko/docs/components/activation/relu) — SiLU가 ReLU를 대체하는 맥락
- [GELU](/ko/docs/components/activation/gelu) — Transformer의 활성화 함수 (비교 대상)
