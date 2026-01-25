---
title: "Positional Encoding"
weight: 3
math: true
---

# Positional Encoding

## 개요

Attention은 순서를 알지 못합니다. Positional Encoding은 위치 정보를 주입하여 순서를 인식하게 합니다.

## 왜 필요한가?

Self-Attention의 특성:
- 입력 순서를 바꿔도 (각 위치의) 출력 동일
- "I love you" ≠ "you love I" 구분 못함

## Sinusoidal Positional Encoding

원래 Transformer 논문에서 제안:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

- pos: 위치 인덱스
- i: 차원 인덱스
- d: 임베딩 차원

### 특징
- 학습 파라미터 없음
- 임의의 길이 처리 가능
- 상대적 위치 표현 가능: PE(pos+k)는 PE(pos)의 선형 변환

```python
import torch
import math

def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe  # (max_len, d_model)

# 사용
pe = sinusoidal_positional_encoding(1000, 512)
x = x + pe[:seq_len]  # 입력에 더하기
```

## Learnable Positional Encoding

학습 가능한 임베딩:

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

**장점**: 데이터에 맞게 최적화
**단점**: 학습 시 본 길이까지만 처리 가능

## 2D Positional Encoding (ViT)

이미지 패치의 2D 위치:

```python
class ViTPositionalEncoding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        # num_patches = (H // patch_size) * (W // patch_size)
        # +1 for [CLS] token
        self.pe = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

    def forward(self, x):
        return x + self.pe
```

## Rotary Positional Encoding (RoPE)

최신 모델(LLaMA, Qwen 등)에서 사용:

```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**장점**:
- 상대 위치 자연스럽게 인코딩
- 임의의 길이 확장 가능
- 계산 효율적

## 비교

| 방법 | 학습 | 길이 외삽 | 사용 |
|------|------|----------|------|
| Sinusoidal | X | O | 원본 Transformer |
| Learnable | O | X | BERT, ViT |
| RoPE | X | O | LLaMA, 최신 LLM |
| ALiBi | X | O | BLOOM |

## 관련 콘텐츠

- [Self-Attention](/ko/docs/math/attention/self-attention)
- [Cross-Attention](/ko/docs/math/attention/cross-attention)
