---
title: "Positional Encoding"
weight: 4
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

### 왜 10000인가?

$10000^{2i/d}$는 각 차원의 **파장(wavelength)**을 결정합니다:

```
차원 i=0:   파장 = 2π × 10000^(0/512)    = 2π      ≈ 6.28     (가장 빠른 진동)
차원 i=128: 파장 = 2π × 10000^(256/512)  = 2π×100  ≈ 628      (중간 속도)
차원 i=255: 파장 = 2π × 10000^(510/512)  = 2π×9770 ≈ 61,400   (가장 느린 진동)
```

**파장 범위**: $2\pi$ ~ $2\pi \times 10000$

- **너무 작은 상수** (예: 100): 느린 차원의 파장이 짧아져 → 먼 위치를 구분하기 어려움
- **너무 큰 상수** (예: 1,000,000): 빠른 차원 외에는 거의 변하지 않아 → 인접 위치 구분 어려움
- **10000**: 일반적인 시퀀스 길이(수백~수천)에서 **인접 위치도, 먼 위치도** 잘 구분되는 경험적 최적값

```python
import torch
import math

# 상수에 따른 PE 변화 비교
d = 512
for base in [100, 10000, 1000000]:
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(base) / d))
    min_freq = div[-1].item()
    max_freq = div[0].item()
    print(f"base={base:>10}: 주파수 범위 [{min_freq:.6f}, {max_freq:.1f}]")
    # base=100:      [0.010000, 1.0] — 좁은 범위
    # base=10000:    [0.000100, 1.0] — 넓은 범위 ✓
    # base=1000000:  [0.000001, 1.0] — 너무 넓음 (대부분 0에 가까움)
```

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

### 상대 위치 표현이 가능한 이유 (수학적 증명)

$PE(pos+k)$가 $PE(pos)$의 **선형 변환**이라는 것은, sin/cos의 덧셈 정리에서 나옵니다:

$$
\sin(\omega \cdot (pos+k)) = \sin(\omega \cdot pos)\cos(\omega \cdot k) + \cos(\omega \cdot pos)\sin(\omega \cdot k)
$$

$$
\cos(\omega \cdot (pos+k)) = \cos(\omega \cdot pos)\cos(\omega \cdot k) - \sin(\omega \cdot pos)\sin(\omega \cdot k)
$$

이것을 행렬로 쓰면:

$$
\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}
$$

→ 오른쪽의 회전 행렬은 **$k$에만 의존**하고 $pos$에 무관합니다.
→ 모델이 이 선형 변환을 학습하면, **상대 거리 $k$**에 기반한 패턴을 파악할 수 있습니다.

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

### 해상도 변경 시: Position Interpolation

ViT를 224×224로 학습했는데 384×384로 추론하면? 패치 수가 196 → 576으로 늘어납니다.

```
224×224 (학습): 14×14 = 196 패치 → PE 196개 학습됨
384×384 (추론): 24×24 = 576 패치 → PE 576개 필요!

해결: 14×14 PE를 24×24로 2D 보간 (bicubic interpolation)
```

```python
import torch.nn.functional as F

def interpolate_pos_encoding(pos_embed, old_size=14, new_size=24):
    """
    학습된 위치 인코딩을 새 해상도에 맞게 보간

    pos_embed: (1, 1+196, d_model) — [CLS] + 14×14 패치
    """
    cls_token = pos_embed[:, :1]        # [CLS]는 그대로
    patch_embed = pos_embed[:, 1:]      # (1, 196, d_model)

    # 2D로 reshape → bicubic 보간 → 다시 1D
    patch_embed = patch_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(patch_embed, size=(new_size, new_size), mode='bicubic')
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_size**2, -1)

    return torch.cat([cls_token, patch_embed], dim=1)  # (1, 1+576, d_model)
```

이 기법 덕분에 ViT는 학습 때와 다른 해상도에서도 추론이 가능합니다. DeiT-III, BEiT v2 등에서 표준적으로 사용됩니다.

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

{{< figure src="/images/components/attention/ko/rope-rotation-visualization.png" caption="RoPE: 위치에 따라 벡터를 회전 — 내적이 상대 거리 (n-m)에만 의존" >}}

2D 벡터를 각도 $\theta$만큼 회전시키는 것을 생각해봅시다:

$$
\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}
$$

RoPE는 이것을 고차원으로 확장한 것입니다. 각 위치마다 다른 회전 각도를 적용하면, 두 위치의 내적이 자동으로 **상대 거리**에만 의존하게 됩니다.

### 왜 상대 위치가 자연스럽게 인코딩되나?

2D에서 증명해봅시다. 위치 $m$의 Q와 위치 $n$의 K가 있을 때:

$$
q_m = R(\theta \cdot m) \cdot q, \quad k_n = R(\theta \cdot n) \cdot k
$$

여기서 $R(\alpha)$는 회전 행렬입니다. 내적을 계산하면:

$$
q_m^T k_n = q^T R(\theta \cdot m)^T R(\theta \cdot n) k = q^T R(\theta \cdot (n - m)) k
$$

→ 내적 결과가 **절대 위치** $m, n$이 아니라 **상대 거리** $(n-m)$에만 의존합니다!

```python
import torch

# RoPE의 상대 위치 특성 수치 검증
def rope_2d(x, pos, theta=10000.0):
    """2D RoPE 적용"""
    angle = pos / theta
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    x0, x1 = x[0], x[1]
    return torch.stack([x0 * cos_a - x1 * sin_a, x0 * sin_a + x1 * cos_a])

q = torch.randn(2)
k = torch.randn(2)

# 위치 (3, 7)과 (10, 14) → 상대 거리 동일 (4)
dot1 = rope_2d(q, 3) @ rope_2d(k, 7)
dot2 = rope_2d(q, 10) @ rope_2d(k, 14)
print(f"pos(3,7):   {dot1:.4f}")
print(f"pos(10,14): {dot2:.4f}")
# 두 값이 동일! → 상대 거리만 중요
```

### RoPE 주파수 미리 계산

실제 구현에서는 주파수를 미리 계산하여 재사용합니다:

```python
def precompute_rope_freqs(seq_len, head_dim, base=10000.0):
    """RoPE의 cos/sin 테이블 미리 계산"""
    # 각 차원 쌍의 주파수 (Sinusoidal PE와 같은 원리)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # 위치 × 주파수 → 각도
    positions = torch.arange(seq_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, head_dim/2)

    # cos, sin 테이블
    cos = torch.cos(angles).repeat(1, 2)  # (seq_len, head_dim)
    sin = torch.sin(angles).repeat(1, 2)
    return cos, sin

# LLaMA: head_dim=128, max_seq_len=4096
cos, sin = precompute_rope_freqs(4096, 128)
# cos.shape = (4096, 128) — 한 번 계산 후 캐싱
```

---

## 방법 5: ALiBi (Attention with Linear Biases)

BLOOM에서 사용하는 방법으로, 위치 인코딩을 **입력에 더하는 대신** attention score에 **편향(bias)**을 더합니다.

### 핵심 아이디어

```
기존 PE:  입력 = X + PE(pos)  → Q, K에 위치 정보 포함
ALiBi:    입력 = X            → attention score에 직접 거리 패널티 추가

scores = Q × K^T - m × |i - j|
                   ↑
           거리에 비례하는 패널티 (head별 다른 기울기 m)
```

### 수식

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} - m_h \cdot |i - j|\right) V
$$

**각 기호의 의미:**
- $Q, K, V$: Query, Key, Value 행렬
- $d_k$: key 차원 수 (스케일링에 사용)
- $i, j$: Query 위치와 Key 위치 인덱스
- $|i - j|$: 토큰 간 절대 거리
- $m_h$: $h$번째 head의 기울기 (head마다 다른 거리 패널티)

- $m_h$는 보통 head별로 기하급수적으로 감소시킵니다 (예: $m_h \in \{2^{-8/n}, 2^{-16/n}, \ldots\}$).
- **실전 주의(디코더 causal)**: 보통은 미래 토큰을 마스크하므로 유효한 구간에서 $j \le i$이고, 이때 $|i-j|$ 대신 $(i-j)$를 그대로 써도 동일하게 동작합니다. 구현에 따라 부호/방향이 다를 수 있으니 `causal mask`와 함께 점검하세요.

```
8개 Head의 기울기 m:
  Head 1: m = 1/2    (가까운 것만 봄)
  Head 2: m = 1/4
  Head 3: m = 1/8
  ...
  Head 8: m = 1/256  (먼 것도 잘 봄)

→ 각 head가 자연스럽게 "시야 범위"가 다름
```

### Head별 slope 생성 (실무에서 자주 쓰는 형태)

ALiBi는 head마다 다른 기울기 $m_h$를 써서, 어떤 head는 근거리 위주로 보고 어떤 head는 원거리까지 보도록 만듭니다.

```python
import torch

def build_alibi_slopes(n_heads: int) -> torch.Tensor:
    """head별 ALiBi slope (큰 값=head가 근거리 위주)

    반환 shape: (n_heads,)
    """
    # ALiBi 논문 구현에서 널리 쓰이는 power-of-two 기반 스케일
    # n_heads가 2의 거듭제곱이 아니어도 단조 감소 slope를 구성
    start = 2 ** (-8.0 / n_heads)
    return torch.tensor([start ** (i + 1) for i in range(n_heads)], dtype=torch.float32)

# 예시: 8 heads
slopes = build_alibi_slopes(8)
print(slopes)  # tensor([0.9170, 0.8409, ..., 0.5000])
```

> 구현마다 slope 생성식은 조금씩 다를 수 있지만, **head마다 단조 감소 기울기**를 주는 원칙은 동일합니다.

### 장점

| 장점 | 설명 |
|------|------|
| **구현 매우 간단** | attention score에 행렬 하나 더하기 |
| **학습 파라미터 0개** | $m$은 고정값 |
| **길이 외삽 우수** | 학습 시 1K → 추론 시 8K도 안정적 |
| **메모리 절약** | 별도 PE 벡터 저장 불필요 |

### 최소 구현 스케치

```python
# scores: (batch, heads, q_len, k_len)
# distance: |i-j| 행렬, shape (q_len, k_len)
# slopes: head별 기울기, shape (heads, 1, 1)

scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores - slopes * distance  # ALiBi bias
scores = scores + causal_mask         # 필요 시 인과 마스크 유지
attn = torch.softmax(scores, dim=-1)
out = attn @ v
```

---

## 비교 정리

{{< figure src="/images/components/attention/ko/positional-encoding-methods-comparison.jpeg" caption="Positional Encoding 방법 비교 — Sinusoidal, Learnable, RoPE, ALiBi의 발전 과정" >}}

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

## 실무 체크포인트

1. **패딩 토큰 처리**: Positional Encoding을 더하더라도, attention mask에서 padding 위치를 반드시 차단해야 합니다.
2. **길이 외삽 전략**: 긴 문장이 자주 나오면 Learnable PE보다 RoPE/ALiBi가 보통 더 안정적입니다.
3. **해상도 변경(ViT)**: 입력 해상도를 바꿀 때는 Position Interpolation을 함께 적용해야 성능 하락을 줄일 수 있습니다.
4. **디버깅 순서**: 학습 불안정 시 `mask 방향 → PE shape → 최대 길이 초과 여부` 순으로 확인하면 빠릅니다.
5. **KV 캐시 오프셋(추론)**: RoPE를 캐시와 함께 쓸 때는 새 토큰의 위치 인덱스를 `현재 시퀀스 길이` 기준으로 이어 붙여야 합니다. 오프셋이 어긋나면 문맥 일관성이 급격히 떨어질 수 있습니다.
6. **RoPE base 일관성(train/serve)**: 학습 때 사용한 `rope_theta`(예: 10000, 500000)를 추론 서버에서도 동일하게 맞추세요. base가 다르면 같은 위치 인덱스여도 회전 각도가 달라져 품질이 눈에 띄게 흔들릴 수 있습니다.

## 관련 콘텐츠

- [Self-Attention](/ko/docs/components/attention/self-attention) — Positional Encoding이 필요한 이유
- [Cross-Attention](/ko/docs/components/attention/cross-attention) — 서로 다른 시퀀스 연결
- [ViT](/ko/docs/architecture/transformer/vit) — 2D Positional Encoding 사용
