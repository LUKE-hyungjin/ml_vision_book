---
title: "Positional Encoding"
weight: 3
math: true
---

# Positional Encoding

{{% hint info %}}
**선수지식**: [Self-Attention](/ko/docs/components/attention/self-attention)
{{% /hint %}}

## 한 줄 요약
> **Attention은 순서를 모르므로, 위치 정보를 별도로 주입해야 한다.**

## 왜 필요한가?

### 문제: Attention은 순서를 무시한다

Self-Attention의 본질은 **집합(set) 연산**입니다. 입력의 순서를 바꿔도 (각 위치의) 출력은 동일합니다.

{{< figure src="/images/components/attention/ko/positional-encoding-sinusoidal.jpeg" caption="Sinusoidal Positional Encoding 패턴 — 낮은 차원일수록 빠르게 진동" >}}

```
Self-Attention에 "내가 너를 좋아한다" 입력:
  → 각 단어가 다른 단어와의 관계를 계산

Self-Attention에 "너가 나를 좋아한다" 입력:
  → 같은 단어들이므로 같은 관계를 계산
  → 결과가 동일! (순서 정보 없음)
```

**문제**: "내가 너를 좋아한다"와 "너가 나를 좋아한다"는 완전히 다른 의미인데, Attention만으로는 구분할 수 없습니다.

### CNN은 왜 이런 문제가 없나?

CNN의 커널은 **순서대로 슬라이딩**합니다. 1번째 위치에서 계산한 것과 3번째 위치에서 계산한 것은 다른 위치에서 나온 결과이므로, 위치 정보가 자연스럽게 포함됩니다.

Attention은 모든 위치를 한꺼번에 보기 때문에, **위치 정보를 별도로 주입**해야 합니다.

### 해결: 위치 정보 더하기

아이디어는 간단합니다. 각 위치에 **고유한 패턴**을 더해주면 됩니다:

```
입력 = 단어 임베딩 + 위치 인코딩

위치 0: [0.2, -0.5, 0.8] + [sin(0), cos(0), sin(0)] = [0.2, 0.5, 0.8]
위치 1: [0.1, 0.3, -0.2] + [sin(1), cos(1), sin(1)] = [0.94, 0.84, 0.64]
위치 2: [0.7, 0.1, 0.4] + [sin(2), cos(2), sin(2)] = [1.61, -0.32, 1.31]

→ 같은 단어라도 위치에 따라 다른 값을 가짐
→ Attention이 순서를 구분할 수 있음
```

---

## 방법 1: Sinusoidal Positional Encoding

원래 Transformer 논문("Attention Is All You Need", 2017)에서 제안한 방법입니다.

### 수식

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**각 기호의 의미:**
- $pos$: 위치 인덱스 (0, 1, 2, ...)
- $i$: 차원 인덱스 (0, 1, 2, ...)
- $d$: 전체 임베딩 차원
- $10000$: 주파수를 결정하는 상수

### 직관적 이해

**시계 비유**: 시, 분, 초침을 생각해봅시다.

```
시침: 느리게 회전 (12시간에 1바퀴) → 큰 시간 단위
분침: 중간 속도 (1시간에 1바퀴) → 중간 시간 단위
초침: 빠르게 회전 (1분에 1바퀴) → 작은 시간 단위

→ 시, 분, 초를 합치면 모든 시각을 고유하게 표현 가능!
```

Sinusoidal PE도 마찬가지입니다:
- **낮은 차원 (i 작음)**: 빠르게 진동 → 인접한 위치 구분
- **높은 차원 (i 큼)**: 느리게 진동 → 먼 위치 구분
- 모든 차원을 합치면 → **각 위치가 고유한 패턴**

### 장점

| 장점 | 설명 |
|------|------|
| **학습 불필요** | 수학 공식으로 생성 → 파라미터 0개 |
| **임의의 길이** | 학습 시 본 적 없는 긴 시퀀스도 처리 가능 |
| **상대 위치 표현** | $PE(pos+k)$는 $PE(pos)$의 선형 변환 → 상대 위치 학습 가능 |

### 구현

```python
import torch
import math

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Sinusoidal Positional Encoding

    max_len: 최대 시퀀스 길이 (예: 1000)
    d_model: 임베딩 차원 (예: 512)
    """
    pe = torch.zeros(max_len, d_model)

    # 위치 인덱스: [0, 1, 2, ..., max_len-1]
    position = torch.arange(0, max_len).unsqueeze(1).float()

    # 주파수: 차원이 높을수록 느리게 진동
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    # 짝수 차원: sin, 홀수 차원: cos
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe  # (max_len, d_model)

# 사용
pe = sinusoidal_positional_encoding(1000, 512)
x = x + pe[:seq_len]  # 입력 임베딩에 위치 정보 더하기
```

---

## 방법 2: Learnable Positional Encoding

BERT, ViT에서 사용하는 방법입니다. 위치 인코딩을 **학습**합니다.

### 아이디어

"수학 공식 대신, 모델이 데이터에서 최적의 위치 표현을 알아서 배우게 하자"

### 구현

```python
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # 학습 가능한 파라미터 (nn.Parameter)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x: (B, N, d_model)
        return x + self.pe[:, :x.size(1)]

# 사용
pe = LearnablePositionalEncoding(max_len=197, d_model=768)
x = pe(x)  # 위치 정보 더하기
```

### 장단점

| | 장점 | 단점 |
|---|------|------|
| **Learnable** | 데이터에 최적화 | 학습 시 본 길이까지만 처리 가능 |
| **Sinusoidal** | 임의 길이 처리 | 데이터 적응 불가 |

---

## 방법 3: 2D Positional Encoding (ViT)

이미지는 1D 시퀀스가 아니라 2D 격자입니다. ViT는 이미지를 패치로 나누고, 각 패치에 위치 인코딩을 적용합니다.

```
224×224 이미지 → 16×16 패치로 나누면 → 14×14 = 196개 패치
+ [CLS] 토큰 1개 = 197개 위치
```

### 구현

```python
class ViTPositionalEncoding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        # num_patches = (H // patch_size) * (W // patch_size)
        # +1: [CLS] 토큰을 위한 위치
        self.pe = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

    def forward(self, x):
        # x: (B, 1+num_patches, d_model)  — [CLS] + patches
        return x + self.pe

# ViT-Base: 14×14 = 196 패치, 768 차원
vit_pe = ViTPositionalEncoding(num_patches=196, d_model=768)
```

ViT의 Positional Encoding을 시각화하면, 인접한 패치끼리 유사한 값을 가지는 2D 격자 패턴이 나타납니다.

---

## 방법 4: Rotary Positional Encoding (RoPE)

최신 LLM(LLaMA, Qwen, Gemma 등)에서 표준으로 사용되는 방법입니다.

### 핵심 아이디어

기존 방법: 입력에 위치 정보를 **더하기** (additive)
RoPE: Q와 K를 위치에 따라 **회전** (rotary)

```
기존: Q + PE(pos), K + PE(pos)
RoPE: Rotate(Q, pos), Rotate(K, pos)
```

### 장점

| 장점 | 설명 |
|------|------|
| **상대 위치 자연스럽게 인코딩** | 두 위치의 내적이 자동으로 상대 거리에 의존 |
| **임의의 길이 확장 가능** | 학습 시 본 적 없는 긴 시퀀스도 처리 가능 |
| **계산 효율적** | 간단한 회전 연산만 필요 |

### 구현

```python
def rotate_half(x):
    """벡터의 앞/뒤 절반을 교환하며 부호 변경"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    """
    RoPE 적용: Q와 K를 위치에 따라 회전

    cos, sin: 위치별 회전 각도 (미리 계산)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 사용 (LLaMA 스타일)
# cos, sin = precompute_freqs(seq_len, head_dim)
# q, k = apply_rope(q, k, cos, sin)
```

### 직관적 이해

2D 벡터를 각도 $\theta$만큼 회전시키는 것을 생각해봅시다:

$$
\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}
$$

RoPE는 이것을 고차원으로 확장한 것입니다. 각 위치마다 다른 회전 각도를 적용하면, 두 위치의 내적이 자동으로 **상대 거리**에만 의존하게 됩니다.

---

## 비교 정리

| 방법 | 학습 필요 | 길이 외삽 | 상대 위치 | 대표 모델 |
|------|---------|----------|----------|----------|
| **Sinusoidal** | X | O | 간접적 | 원본 Transformer |
| **Learnable** | O | X | X | BERT, ViT |
| **RoPE** | X | O | O (자연스럽게) | LLaMA, Qwen, Gemma |
| **ALiBi** | X | O | O (편향 추가) | BLOOM |

### 어떤 것을 쓸까?

- **비전 (ViT)**: Learnable PE (이미지 크기가 고정적이므로)
- **LLM (텍스트)**: RoPE (가변 길이 텍스트 처리에 유리)
- **학습/교육**: Sinusoidal (이해하기 가장 쉬움)

---

## 정리

| 질문 | 답 |
|------|---|
| 왜 필요한가? | Attention은 집합 연산이라 순서를 모름 |
| 어떻게 해결? | 각 위치에 고유한 패턴을 더해줌 |
| Sinusoidal은? | sin/cos로 만든 고정 패턴 — 학습 불필요, 임의 길이 |
| Learnable은? | 학습으로 최적화 — ViT에서 사용 |
| RoPE는? | Q/K를 위치에 따라 회전 — 최신 LLM 표준 |

## 관련 콘텐츠

- [Self-Attention](/ko/docs/components/attention/self-attention) — Positional Encoding이 필요한 이유
- [Cross-Attention](/ko/docs/components/attention/cross-attention) — 서로 다른 시퀀스 연결
- [ViT](/ko/docs/architecture/transformer/vit) — 2D Positional Encoding 사용
