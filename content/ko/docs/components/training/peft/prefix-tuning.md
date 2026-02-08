---
title: "Prefix Tuning"
weight: 4
math: true
---

# Prefix Tuning

## 개요

Prefix Tuning은 각 Transformer 층에 학습 가능한 "가상 토큰"을 추가하여 모델을 조정합니다.

## 핵심 아이디어

```
일반 입력:  [토큰1, 토큰2, 토큰3, ...]
Prefix:    [P1, P2, ..., Pm, 토큰1, 토큰2, 토큰3, ...]
                 ↑
           학습 가능한 가상 토큰
```

## Prompt Tuning vs Prefix Tuning

| | Prompt Tuning | Prefix Tuning |
|---|--------------|---------------|
| 적용 위치 | 입력 임베딩만 | 모든 층의 key, value |
| 파라미터 | 매우 적음 | 더 많음 |
| 성능 | 대형 모델에서 좋음 | 더 안정적 |

## 수식

각 층에서:
$$
\text{Attention}(Q, [P_K; K], [P_V; V])
$$

- P_K, P_V: 학습 가능한 prefix (각 층마다 다름)
- [;]: concatenation

## 구현

```python
import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    def __init__(self, num_layers, num_heads, head_dim, prefix_length=20):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers

        # 각 층의 prefix key, value
        # (num_layers, 2, prefix_length, num_heads * head_dim)
        embed_dim = num_heads * head_dim

        # MLP를 통해 reparameterization (안정성)
        self.prefix_embedding = nn.Embedding(prefix_length, embed_dim)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Tanh(),
            nn.Linear(embed_dim * 2, num_layers * 2 * embed_dim)
        )

    def forward(self, batch_size):
        # Prefix indices
        prefix_ids = torch.arange(self.prefix_length, device=self.prefix_embedding.weight.device)

        # Embedding -> MLP
        prefix_embed = self.prefix_embedding(prefix_ids)  # (prefix_len, embed_dim)
        prefix = self.prefix_mlp(prefix_embed)  # (prefix_len, num_layers * 2 * embed_dim)

        # Reshape for each layer
        prefix = prefix.view(
            self.prefix_length,
            self.num_layers,
            2,  # key, value
            -1
        ).permute(1, 2, 0, 3)  # (num_layers, 2, prefix_len, embed_dim)

        # Expand for batch
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        return prefix  # (batch, num_layers, 2, prefix_len, embed_dim)
```

## Attention에 적용

```python
class AttentionWithPrefix(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x, prefix_kv=None):
        B, L, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Prefix 추가
        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv  # (B, prefix_len, D)
            K = torch.cat([prefix_k, K], dim=1)
            V = torch.cat([prefix_v, V], dim=1)

        # Multi-head attention
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)
```

## PEFT 라이브러리

```python
from peft import PrefixTuningConfig, get_peft_model

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prefix_projection=True  # MLP reparameterization
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 983,040 || all params: 6,739,415,040 || 0.01%
```

## P-Tuning v2

Prefix Tuning의 변형으로, 더 깊은 층에 적용:

```python
config = PrefixTuningConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=20,
    encoder_hidden_size=512,  # 더 큰 hidden
    prefix_projection=True
)
```

## 장단점

**장점**:
- 파라미터 매우 적음
- 태스크별 prefix만 저장
- 모델 가중치 완전 보존

**단점**:
- 시퀀스 길이 약간 증가
- 학습 불안정할 수 있음

## 관련 콘텐츠

- [LoRA](/ko/docs/components/training/peft/lora)
- [Self-Attention](/ko/docs/components/attention/self-attention)
- [Transformer](/ko/docs/architecture/transformer)
