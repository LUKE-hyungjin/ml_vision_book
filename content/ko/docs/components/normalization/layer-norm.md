---
title: "Layer Normalization"
weight: 2
math: true
---

# Layer Normalization (레이어 정규화)

{{% hint info %}}
**선수지식**: [Batch Normalization](/ko/docs/components/normalization/batch-norm)
{{% /hint %}}

> **한 줄 요약**: Layer Normalization은 **각 샘플 내에서** 모든 feature를 정규화하여, 배치 크기에 무관하게 작동하는 **Transformer의 표준 정규화**입니다.

## 왜 Layer Normalization이 필요한가?

### 문제 상황: "BatchNorm을 Transformer에 쓸 수 없습니다"

BatchNorm은 CNN에서 잘 작동하지만, Transformer/RNN에서는 문제가 있습니다:

**1. 시퀀스 길이가 다릅니다**

```python
# NLP: 문장마다 길이가 다름
batch = [
    "고양이",                    # 길이 3
    "귀여운 고양이가 잠을 잔다",    # 길이 12
    "나",                       # 길이 1
]
# BatchNorm은 같은 위치(시간축)에서 평균을 구하는데...
# 위치 4~12는 일부 샘플에만 존재!
```

**2. 배치 간 독립이어야 합니다**

```python
# Inference 시 배치 크기 = 1인 경우가 많음
# BatchNorm: 배치 1개로는 통계가 무의미
# LayerNorm: 샘플 내에서 계산하므로 문제 없음
```

**3. Train/Eval 불일치 없어야 합니다**

```
BatchNorm: training → 배치 통계, eval → running 통계 (다를 수 있음!)
LayerNorm: training = eval (항상 동일 연산)
```

### 해결: "배치가 아닌 feature 축으로 정규화하자!"

```
BatchNorm: "이 채널이 전체 배치에서 어떤 분포인가?" → 배치 축
LayerNorm: "이 샘플의 모든 feature가 어떤 분포인가?" → feature 축
```

---

## 수식

### Layer Normalization

입력 $\mathbf{x} = (x_1, x_2, ..., x_D)$에서 (D = feature 차원):

$$
\mu = \frac{1}{D} \sum_{i=1}^{D} x_i
$$

$$
\sigma^2 = \frac{1}{D} \sum_{i=1}^{D} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma_i \cdot \hat{x}_i + \beta_i
$$

**각 기호의 의미:**
- $D$ : feature 차원 크기 (Transformer에서는 hidden_dim, 예: 768)
- $\mu, \sigma^2$ : **이 샘플 내에서** 계산 — 배치와 무관!
- $\gamma_i, \beta_i$ : feature 위치별 학습 가능한 파라미터 (총 D개씩)

### BatchNorm vs LayerNorm 정규화 축

```
입력: (B, T, D) — Batch, Time(sequence), Dimension

BatchNorm:  B, T 축으로 평균 → D개 통계 (차원별 1개)
LayerNorm:  D 축으로 평균   → B×T개 통계 (토큰별 1개)
```

```
        D (feature 차원)
        ─────────→
    B │ ┌─────────────┐
    ↓ │ │  token 1    │ ← LayerNorm: 이 한 줄에서 평균
      │ │  token 2    │
      │ │  token 3    │
      │ └─────────────┘
        ↑
        BatchNorm: 이 한 열에서 평균 (배치 전체)
```

---

## 구현

```python
import torch
import torch.nn as nn

# === PyTorch LayerNorm ===
ln = nn.LayerNorm(normalized_shape=768)  # 마지막 차원 크기

x = torch.randn(32, 196, 768)  # (B, T, D) — ViT: 196 패치, 768 차원
y = ln(x)

# 학습 파라미터
print(f"gamma: {ln.weight.shape}")  # [768]
print(f"beta: {ln.bias.shape}")     # [768]
# 버퍼 없음! (running stats 불필요)


# === 수동 구현 ===
class ManualLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 마지막 축(feature)에서만 통계 계산
        mean = x.mean(dim=-1, keepdim=True)           # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (..., 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# === 2D 이미지에 대한 LayerNorm ===
# CNN에서 사용할 때: (B, C, H, W) → 정규화 축: (C, H, W)
ln_2d = nn.LayerNorm([256, 56, 56])  # normalized_shape에 여러 차원 지정
x_2d = torch.randn(4, 256, 56, 56)
y_2d = ln_2d(x_2d)
```

---

## Pre-LN vs Post-LN

Transformer에서 LayerNorm의 **위치**가 학습 안정성을 크게 좌우합니다.

### Post-LN (원래 Transformer, 2017)

```python
# Attention → Add → LayerNorm
def post_ln_block(x, attn, ffn, ln1, ln2):
    x = ln1(x + attn(x))     # Residual 후 정규화
    x = ln2(x + ffn(x))
    return x
```

### Pre-LN (현대적, 안정적)

```python
# LayerNorm → Attention → Add
def pre_ln_block(x, attn, ffn, ln1, ln2):
    x = x + attn(ln1(x))     # 정규화 후 Attention → Residual
    x = x + ffn(ln2(x))
    return x
```

### 비교

| | Post-LN | Pre-LN |
|---|---|---|
| 구조 | $\text{LN}(x + \text{Attn}(x))$ | $x + \text{Attn}(\text{LN}(x))$ |
| 학습 안정성 | 불안정 (warmup 필수) | **안정적** |
| 최종 성능 | 약간 더 좋을 수 있음 | 약간 낮을 수 있음 |
| 사용 모델 | 원래 Transformer | **GPT, ViT, LLaMA 등 대부분** |

```python
# Pre-LN Transformer Block (현대적 구현)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Pre-LN: 정규화 → 연산 → Residual
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x
```

---

## BatchNorm vs LayerNorm 상세 비교

| | BatchNorm | LayerNorm |
|---|---|---|
| 정규화 축 | Batch + 공간 | Feature |
| 통계 개수 | $C$개 | $B \times T$개 |
| 배치 의존 | **O** | X |
| Train/Eval 차이 | **O** (running stats) | X |
| 버퍼 필요 | O (running_mean/var) | **X** |
| 주 사용처 | **CNN** | **Transformer** |
| 정규화 효과 | 배치 노이즈 → 정규화 | 없음 |

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === 배치 크기 무관 확인 ===
print("=== 배치 크기에 따른 동작 ===")
ln = nn.LayerNorm(64)

for B in [32, 4, 1]:
    x = torch.randn(B, 10, 64)
    y = ln(x)
    print(f"B={B:>2}: 출력 평균={y.mean():.4f}, 분산={y.var():.4f}")
# 모든 배치 크기에서 안정적!

# === Train vs Eval 동일 확인 ===
print("\n=== Train vs Eval ===")
ln = nn.LayerNorm(64)
x = torch.randn(4, 10, 64)

ln.train()
y_train = ln(x)

ln.eval()
y_eval = ln(x)

diff = (y_train - y_eval).abs().max().item()
print(f"Train vs Eval 차이: {diff:.6f}")  # 0.000000!

# === BatchNorm vs LayerNorm 비교 ===
print("\n=== BN vs LN 정규화 축 비교 ===")
B, C, H, W = 4, 3, 4, 4
x = torch.randn(B, C, H, W) * 5 + 10  # 편향된 입력

bn = nn.BatchNorm2d(C)
ln = nn.LayerNorm([C, H, W])
bn.train()

y_bn = bn(x)
y_ln = ln(x)

# BN: 채널별 정규화 → 각 채널의 평균 ≈ 0
print("BatchNorm (채널별 평균):")
for c in range(C):
    print(f"  채널 {c}: 평균={y_bn[:, c].mean():.4f}")

# LN: 샘플별 정규화 → 각 샘플의 평균 ≈ 0
print("LayerNorm (샘플별 평균):")
for b in range(B):
    print(f"  샘플 {b}: 평균={y_ln[b].mean():.4f}")
```

---

## 핵심 정리

| 항목 | 내용 |
|------|------|
| **정규화 축** | Feature (마지막 D개 차원) |
| **통계** | 토큰/샘플별 $\mu, \sigma^2$ |
| **배치 의존** | X — 배치 1도 OK |
| **Train/Eval 차이** | X — 동일 동작 |
| **주 사용처** | Transformer (ViT, GPT, BERT) |
| **현대적 사용** | Pre-LN (Residual 전에 정규화) |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| Pre-LN | ViT, GPT, LLaMA | 학습 안정성 |
| Post-LN | 원래 Transformer | 역사적 의미 |
| LN → RMSNorm 교체 | LLaMA, Qwen | 연산 효율화 |
| LN in CNN | ConvNeXt | Transformer 스타일 CNN |

---

## 관련 콘텐츠

- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — 선수 지식: 배치 기반 정규화
- [RMSNorm](/ko/docs/components/normalization/rms-norm) — LayerNorm의 경량화 버전
- [Group Normalization](/ko/docs/components/normalization/group-norm) — CNN에서 BN의 대안
- [Self-Attention](/ko/docs/components/attention/self-attention) — LayerNorm이 사용되는 핵심 구조
