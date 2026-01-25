---
title: "Self-Attention"
weight: 1
math: true
---

# Self-Attention

## 개요

Self-Attention은 시퀀스 내 모든 위치 쌍 간의 관계를 계산하여 각 위치의 표현을 업데이트합니다.

## Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- **Q (Query)**: "무엇을 찾을까?"
- **K (Key)**: "여기 뭐가 있지?"
- **V (Value)**: "실제 값은 이거야"
- **√d_k**: 스케일링 (gradient 안정화)

## 직관적 이해

문장 "The cat sat on the mat" 에서:
- "sat"이 Query
- 모든 단어가 Key
- Attention scores: ["The": 0.1, "cat": 0.4, "sat": 0.1, "on": 0.1, "the": 0.1, "mat": 0.2]
- "cat"에 높은 가중치 → "sat"의 의미 파악에 중요

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Sequence Length, Channels

        # QKV 계산
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(out)

# 사용
x = torch.randn(32, 196, 768)  # (B, 14*14 patches, dim)
attn = SelfAttention(768)
out = attn(x)  # (32, 196, 768)
```

## Multi-Head Attention

여러 개의 attention head를 병렬로 사용:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

각 head는 서로 다른 관계 학습:
- Head 1: 인접 관계
- Head 2: 먼 거리 관계
- Head 3: 의미적 유사성
- ...

## 계산 복잡도

Self-Attention: O(N² · d)
- N: 시퀀스 길이
- d: 임베딩 차원

**문제**: N이 크면 메모리/연산 폭발

**해결책**:
- Flash Attention: 메모리 효율적 구현
- Window Attention: 로컬 윈도우만 계산
- Linear Attention: O(N) 근사

## Flash Attention

```python
# PyTorch 2.0+
from torch.nn.functional import scaled_dot_product_attention

# 자동으로 Flash Attention 사용 (GPU, 조건 충족 시)
attn_output = scaled_dot_product_attention(q, k, v)
```

## 관련 콘텐츠

- [Cross-Attention](/ko/docs/math/attention/cross-attention)
- [Positional Encoding](/ko/docs/math/attention/positional-encoding)
- [Layer Normalization](/ko/docs/math/normalization/layer-norm)
