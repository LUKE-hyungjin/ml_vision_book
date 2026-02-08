---
title: "Layer Normalization"
weight: 2
math: true
---

# Layer Normalization

## 개요

Layer Normalization은 각 샘플 내에서 feature 축을 따라 정규화합니다. Transformer의 표준 정규화 방법입니다.

## Batch Norm vs Layer Norm

| | Batch Norm | Layer Norm |
|---|---|---|
| 정규화 축 | Batch | Feature (Hidden) |
| 의존성 | 배치 크기 의존 | 배치 무관 |
| 주 사용처 | CNN | Transformer, RNN |

```
입력: (B, T, D) - Batch, Time, Dimension

BatchNorm: B, T 축으로 평균 → 채널별 정규화
LayerNorm: D 축으로 평균 → 샘플별 정규화
```

## 수식

입력 x = (x₁, x₂, ..., x_D):

$$
\mu = \frac{1}{D} \sum_{i=1}^D x_i
$$

$$
\sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

## 구현

```python
import torch
import torch.nn as nn

# PyTorch LayerNorm
ln = nn.LayerNorm(normalized_shape=768)  # 마지막 차원 크기

x = torch.randn(32, 196, 768)  # (B, T, D)
y = ln(x)

# 수동 구현
class ManualLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

## Pre-LN vs Post-LN

**Post-LN** (원래 Transformer):
```python
x = x + Attention(LayerNorm(x))  # 잘못된 표현
# 실제:
x = LayerNorm(x + Attention(x))
```

**Pre-LN** (현대적, 안정적):
```python
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Pre-LN 장점:
- 더 안정적인 학습
- Warmup 없이도 학습 가능
- 대부분의 최신 모델 사용

## 관련 콘텐츠

- [Batch Norm](/ko/docs/components/normalization/batch-norm)
- [RMSNorm](/ko/docs/components/normalization/rms-norm)
- [Self-Attention](/ko/docs/components/attention/self-attention)
