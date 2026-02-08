---
title: "Transformer"
weight: 5
bookCollapseSection: true
math: true
---

# Transformer

## 개요

- **논문**: Attention Is All You Need (2017)
- **저자**: Vaswani et al. (Google)
- **핵심 기여**: Self-attention으로 RNN 없이 시퀀스 처리

원래 NLP를 위해 설계되었으나, Vision 분야에서도 혁신을 가져왔습니다.

---

## 핵심 아이디어: Self-Attention

모든 위치 간의 관계를 직접 모델링:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q (Query)**: 현재 위치에서 찾고 싶은 것
- **K (Key)**: 다른 위치들의 특성
- **V (Value)**: 실제 정보

### CNN과 비교

| 측면 | CNN | Transformer |
|------|-----|-------------|
| 수용 영역 | 지역적 (kernel size) | 전역적 (모든 위치) |
| 관계 모델링 | 암시적 | 명시적 (attention) |
| 위치 정보 | 구조에 내재 | Positional encoding 필요 |
| 계산량 | O(n) | O(n²) |

---

## 구조

### Encoder-Decoder 구조

```
┌─────────────────────┐     ┌─────────────────────┐
│       Encoder       │     │       Decoder       │
├─────────────────────┤     ├─────────────────────┤
│  Multi-Head Attn    │     │  Masked Multi-Head  │
│         ↓           │     │         ↓           │
│  Add & Norm         │────→│  Cross Attention    │
│         ↓           │     │         ↓           │
│  Feed Forward       │     │  Add & Norm         │
│         ↓           │     │         ↓           │
│  Add & Norm         │     │  Feed Forward       │
└─────────────────────┘     │         ↓           │
        × N                 │  Add & Norm         │
                            └─────────────────────┘
                                    × N
```

### Multi-Head Attention

여러 관점에서 attention 수행:

```python
# 8개 head, 각 head는 64차원
MultiHead(Q, K, V) = Concat(head_1, ..., head_8) @ W_O
where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

### Positional Encoding

위치 정보 주입:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

---

## Vision에서의 Transformer

### 핵심 모델들

| 모델 | 연도 | 입력 처리 | 특징 |
|------|------|----------|------|
| [ViT](/ko/docs/architecture/transformer/vit) | 2020 | Image → Patches | 순수 Transformer |
| DeiT | 2021 | Distillation | 효율적 학습 |
| Swin | 2021 | Shifted Windows | 계층적 구조 |
| [DiT](/ko/docs/architecture/transformer/dit) | 2023 | Diffusion + Transformer | 이미지 생성 |

---

## 구현 예시

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Concat and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

---

## Transformer의 영향

### Vision 분야

- Image Classification: ViT, DeiT, Swin
- Object Detection: DETR, Deformable DETR
- Segmentation: [SAM](/ko/docs/architecture/segmentation/sam), SegFormer
- Generation: [DiT](/ko/docs/architecture/transformer/dit)

### 장점

- 전역적 context 이해
- 확장성 (scaling law)
- 사전학습 효과적

### 단점

- O(n²) 계산량
- 많은 데이터 필요
- 위치 정보 취약

---

## 관련 콘텐츠

- [Attention](/ko/docs/components/attention) - Attention 수식 상세
- [ViT](/ko/docs/architecture/transformer/vit) - Vision Transformer
- [DiT](/ko/docs/architecture/transformer/dit) - Diffusion Transformer
- [CNN 기초](/ko/docs/architecture/cnn) - 대조되는 접근 방식
