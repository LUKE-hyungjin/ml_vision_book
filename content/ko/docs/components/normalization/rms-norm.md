---
title: "RMSNorm"
weight: 3
math: true
---

# RMSNorm (Root Mean Square Normalization)

{{% hint info %}}
**선수지식**: [Layer Normalization](/ko/docs/components/normalization/layer-norm)
{{% /hint %}}

> **한 줄 요약**: RMSNorm은 LayerNorm에서 **평균 빼기(mean centering)를 제거**하고 RMS만으로 정규화하여, 동등한 성능에서 **15~20% 빠른** 경량화된 정규화입니다.

## 왜 RMSNorm이 필요한가?

### 문제 상황: "LLM이 커지면서 LayerNorm 비용이 무시할 수 없습니다"

```python
# LLaMA-70B 모델
# 80개 Transformer 블록 × 블록당 LayerNorm 2개 = 160번의 LayerNorm
# 각 LayerNorm에서:
#   1. mean 계산  ← 비용
#   2. variance 계산  ← 비용
#   3. 정규화 + scale/shift

# 질문: mean centering이 정말 필요한가?
```

### 핵심 발견: "LayerNorm의 성공은 re-scaling이지, re-centering이 아니다"

연구자들이 LayerNorm의 두 연산을 분리해서 실험했습니다:

```
LayerNorm = re-centering (평균 빼기) + re-scaling (분산으로 나누기)

실험 결과:
  LayerNorm 전체:     성능 100% (기준)
  re-scaling만:       성능 ~100% (거의 동일!)  ← RMSNorm
  re-centering만:     성능 하락
```

**결론**: 평균을 빼는 것은 거의 기여하지 않으므로, 제거해도 됩니다!

### 해결: "평균을 빼지 말고, RMS로만 나누자!"

```
LayerNorm:  (x - mean) / std × γ + β    → mean, var 둘 다 계산
RMSNorm:    x / RMS(x) × γ              → RMS만 계산 (mean, β 제거!)
```

---

## 수식

### LayerNorm (복습)

$$
\text{LayerNorm}(x_i) = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i
$$

연산: mean 계산 → variance 계산 → centering → scaling → affine

### RMSNorm

$$
\text{RMSNorm}(x_i) = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i
$$

여기서:

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon}
$$

**각 기호의 의미:**
- $\text{RMS}(\mathbf{x})$ : Root Mean Square — 원소 제곱의 평균의 제곱근
- $D$ : feature 차원 크기
- $\gamma_i$ : 학습 가능한 스케일 파라미터
- $\beta$ : **없음!** (bias 제거)

### 무엇이 제거되었나?

| 연산 | LayerNorm | RMSNorm |
|------|-----------|---------|
| mean ($\mu$) 계산 | O | **X** |
| $x - \mu$ (centering) | O | **X** |
| variance ($\sigma^2$) 계산 | O | **X** |
| RMS 계산 | X | **O** |
| $\gamma$ (scale) | O | O |
| $\beta$ (shift) | O | **X** |

**핵심 차이**: LayerNorm은 mean과 variance를 **별도로** 계산하지만, RMSNorm은 $\sum x_i^2$ **하나만** 계산합니다.

### 수학적 관계

RMS와 std의 관계:

$$
\text{RMS}(\mathbf{x})^2 = \frac{1}{D}\sum x_i^2 = \mu^2 + \sigma^2
$$

- $\mu = 0$이면: $\text{RMS} = \sigma$ → LayerNorm과 동일!
- LayerNorm은 먼저 $\mu=0$으로 만든 후 $\sigma$로 나누지만
- RMSNorm은 그냥 $\sqrt{\mu^2 + \sigma^2}$로 나눔 → 한 번에 처리

---

## 구현

```python
import torch
import torch.nn as nn

# === RMSNorm 구현 ===
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ만, β 없음!

    def forward(self, x):
        # RMS 계산: sqrt(mean(x²))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 정규화 + 스케일
        return self.weight * (x / rms)


# === 사용 예시 ===
rms_norm = RMSNorm(dim=4096)
x = torch.randn(4, 2048, 4096)  # (B, T, D)
y = rms_norm(x)
print(f"입력: {x.shape} → 출력: {y.shape}")


# === LayerNorm과 비교 ===
ln = nn.LayerNorm(4096)
rn = RMSNorm(4096)

# 파라미터 수 비교
ln_params = sum(p.numel() for p in ln.parameters())
rn_params = sum(p.numel() for p in rn.parameters())
print(f"LayerNorm 파라미터: {ln_params:,}")  # 8,192 (γ 4096 + β 4096)
print(f"RMSNorm 파라미터:   {rn_params:,}")  # 4,096 (γ 4096만)
```

---

## 속도 비교

### 이론적 분석

```
LayerNorm 연산:
  1. mean(x)           → 1회 reduction
  2. x - mean          → 1회 elementwise
  3. var(x - mean)     → 1회 reduction
  4. (x - mean) / std  → 1회 elementwise
  5. γ * x + β         → 2회 elementwise
  총: 2회 reduction + 4회 elementwise

RMSNorm 연산:
  1. mean(x²)          → 1회 reduction
  2. x / rms           → 1회 elementwise
  3. γ * x             → 1회 elementwise
  총: 1회 reduction + 2회 elementwise
```

**reduction이 줄어든 것이 핵심**: GPU에서 reduction은 비싸고, elementwise는 저렴합니다.

### 실측 벤치마크

```python
import time

def benchmark(norm_layer, x, n_iter=1000):
    """간단한 속도 벤치마크"""
    # Warmup
    for _ in range(100):
        _ = norm_layer(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iter):
        _ = norm_layer(x)
    torch.cuda.synchronize()
    return (time.time() - start) / n_iter * 1000  # ms

# GPU에서 실행 시 (결과는 하드웨어에 따라 다름)
# dim = 4096, seq_len = 2048, batch = 4
# LayerNorm: ~0.15ms
# RMSNorm:   ~0.12ms  (약 20% 빠름)
```

---

## 사용 모델

### LLaMA (Meta)

```python
# LLaMA의 Transformer Block (간략화)
class LLaMABlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention_norm = RMSNorm(dim)    # LayerNorm 대신 RMSNorm!
        self.ffn_norm = RMSNorm(dim)          # 여기도!
        self.attention = MultiHeadAttention(dim, n_heads)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        # Pre-RMSNorm (Pre-LN과 같은 위치)
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

### 주요 사용 모델 목록

| 모델 | 정규화 | 차원 |
|------|--------|------|
| **LLaMA 1/2/3** (Meta) | RMSNorm | 4096~8192 |
| **Qwen 1/2** (Alibaba) | RMSNorm | 4096~8192 |
| **Mistral / Mixtral** | RMSNorm | 4096 |
| **Gemma** (Google) | RMSNorm | 2048~3072 |
| GPT-2/3 (OpenAI) | LayerNorm | 768~12288 |
| ViT (Google) | LayerNorm | 768~1024 |
| BERT (Google) | LayerNorm | 768 |

**추세**: 2023년 이후 대규모 LLM은 거의 **RMSNorm을 표준으로** 채택하고 있습니다.

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === LayerNorm vs RMSNorm 출력 비교 ===
print("=== LayerNorm vs RMSNorm ===")

dim = 64
ln = nn.LayerNorm(dim, elementwise_affine=False)  # γ, β 없이 비교
rn = RMSNorm(dim)
rn.weight.data.fill_(1.0)  # γ=1로 통일

x = torch.randn(4, 10, dim)

y_ln = ln(x)
y_rn = rn(x)

print(f"LayerNorm — 평균: {y_ln.mean():.4f}, 분산: {y_ln.var():.4f}")
print(f"RMSNorm   — 평균: {y_rn.mean():.4f}, 분산: {y_rn.var():.4f}")
# RMSNorm: 평균이 정확히 0이 아님 (centering 안 하므로)

# === 평균이 0에 가까울 때 두 방법은 거의 동일 ===
print("\n=== 평균 ≈ 0일 때 ===")
x_zero_mean = x - x.mean(dim=-1, keepdim=True)  # 수동으로 centering

y_ln_zm = ln(x_zero_mean)
y_rn_zm = rn(x_zero_mean)
rn_weight_adjusted = x_zero_mean / torch.sqrt(x_zero_mean.pow(2).mean(-1, keepdim=True) + 1e-6)

diff = (y_ln_zm - rn_weight_adjusted).abs().mean()
print(f"평균=0일 때 LN vs RMS 차이: {diff:.6f}")  # 거의 0!

# === 파라미터 수 절약 (대형 모델) ===
print("\n=== 파라미터 절약 ===")
for dim in [768, 4096, 8192]:
    ln_p = dim * 2  # γ + β
    rn_p = dim * 1  # γ만
    print(f"dim={dim:>5}: LN={ln_p:>6,}, RMSNorm={rn_p:>6,}, "
          f"절약={ln_p - rn_p:,}")

# dim=  768: LN= 1,536, RMSNorm=  768, 절약=768
# dim= 4096: LN= 8,192, RMSNorm=4,096, 절약=4,096
# dim= 8192: LN=16,384, RMSNorm=8,192, 절약=8,192
# LLaMA-70B: 160개 norm × 절약 → 상당한 파라미터/연산 절약
```

---

## 핵심 정리

| 항목 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 정규화 방식 | $(x - \mu) / \sigma$ | $x / \text{RMS}$ |
| Mean centering | O | **X (제거)** |
| Bias ($\beta$) | O | **X (제거)** |
| 파라미터 | $2D$ ($\gamma + \beta$) | $D$ ($\gamma$만) |
| 속도 | 기준 | **15~20% 빠름** |
| 성능 | 기준 | **동등** |
| 주 사용처 | Transformer, ViT | **최신 LLM** |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| RMSNorm | LLaMA, Qwen, Mistral, Gemma | 최신 LLM의 표준 정규화 |
| Pre-RMSNorm | LLaMA 스타일 | Pre-LN + RMSNorm 조합 |
| QK-Norm | Gemma 2 | Attention 내 RMSNorm 적용 |

---

## 관련 콘텐츠

- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — 선수 지식: RMSNorm의 원본
- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — CNN의 정규화
- [Group Normalization](/ko/docs/components/normalization/group-norm) — CNN에서 BN의 대안
