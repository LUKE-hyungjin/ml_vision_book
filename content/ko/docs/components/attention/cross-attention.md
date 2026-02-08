---
title: "Cross-Attention"
weight: 2
math: true
---

# Cross-Attention

## 개요

Cross-Attention은 Query와 Key/Value가 다른 소스에서 오는 attention입니다. 두 모달리티(텍스트↔이미지) 간 정보 교환에 사용됩니다.

## Self vs Cross Attention

| | Self-Attention | Cross-Attention |
|---|---|---|
| Q 소스 | 자기 자신 | A 시퀀스 |
| K, V 소스 | 자기 자신 | B 시퀀스 |
| 용도 | 내부 관계 학습 | 외부 조건 반영 |

## 수식

$$
\text{CrossAttention}(Q_A, K_B, V_B) = \text{softmax}\left(\frac{Q_A K_B^T}{\sqrt{d_k}}\right)V_B
$$

- Q: 대상 시퀀스 (예: 이미지 특징)
- K, V: 조건 시퀀스 (예: 텍스트 임베딩)

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        """
        x: (B, N, query_dim) - 이미지 특징
        context: (B, M, context_dim) - 텍스트 임베딩
        """
        B, N, C = x.shape

        # Q from x, K/V from context
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# 사용 예: Stable Diffusion
image_features = torch.randn(4, 4096, 320)  # 64x64 latent
text_embeddings = torch.randn(4, 77, 768)   # CLIP text

cross_attn = CrossAttention(320, 768, num_heads=8)
out = cross_attn(image_features, text_embeddings)
```

## 사용 사례

### 1. Stable Diffusion

텍스트 조건을 이미지 생성에 반영:

```
이미지 노이즈 (Query) + 텍스트 프롬프트 (Key, Value)
→ 텍스트에 맞는 이미지 생성
```

### 2. VLM (Vision-Language Models)

이미지 질문 답변:

```
질문 토큰 (Query) + 이미지 특징 (Key, Value)
→ 이미지 기반 답변 생성
```

### 3. Transformer Decoder

원문 → 번역:

```
번역 토큰 (Query) + 원문 인코딩 (Key, Value)
→ 번역 결과
```

## Cross-Attention Map 시각화

어떤 텍스트가 이미지 어느 부분에 대응하는지 확인:

```python
def visualize_cross_attention(attn_map, text_tokens, image_size=(64, 64)):
    """
    attn_map: (num_heads, N_image, N_text)
    """
    import matplotlib.pyplot as plt

    # 평균 attention (heads 평균)
    attn = attn_map.mean(0)  # (N_image, N_text)
    H, W = image_size

    for i, token in enumerate(text_tokens):
        plt.subplot(1, len(text_tokens), i+1)
        plt.imshow(attn[:, i].reshape(H, W))
        plt.title(token)
```

## 관련 콘텐츠

- [Self-Attention](/ko/docs/components/attention/self-attention)
- [Positional Encoding](/ko/docs/components/attention/positional-encoding)
- [Layer Normalization](/ko/docs/components/normalization/layer-norm)
