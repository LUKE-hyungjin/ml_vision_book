---
title: "Multi-Head Attention"
weight: 2
math: true
---

# Multi-Head Attention

{{% hint info %}}
**선수지식**: [Self-Attention](/ko/docs/components/attention/self-attention)
{{% /hint %}}

## 한 줄 요약
> **하나의 Attention 대신 여러 개의 "전문가"가 각각 다른 관점에서 관계를 파악한 뒤, 결과를 합치는 메커니즘**

## 왜 필요한가?

### 문제 상황: 한 번의 Attention으로는 부족하다

"고양이가 매트 위에 앉아 있다"라는 문장을 분석할 때, 우리는 **동시에 여러 관점**에서 이해합니다:

```
관점 1 (문법): "앉아 있다"의 주어는 "고양이가"
관점 2 (위치): "앉아 있다"의 장소는 "매트 위에"
관점 3 (의미): "고양이"와 "매트"는 가정 용품-반려동물 관계
```

하지만 Single-Head Attention은 **하나의 가중치 행렬**로 이 모든 관계를 표현해야 합니다. 마치 한 사람이 문법, 의미, 위치를 **동시에** 하나의 숫자로 표현하려는 것과 같습니다.

### 비유: 팀 분석

| 접근 | 비유 | 한계 |
|------|------|------|
| **Single-Head** | 만능 분석가 1명 | 한 관점에 치우침 |
| **Multi-Head** | 전문가 8명 팀 | 각자 다른 패턴 발견 후 종합 |

---

## 수식

### Self-Attention (복습)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Multi-Head Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**각 기호의 의미:**
- $h$ : Head 수 (예: ViT-Base는 12개)
- $W_i^Q, W_i^K, W_i^V$ : 각 Head의 **독립적인** 프로젝션 행렬
- $W^O$ : 모든 Head의 결과를 합치는 **출력 프로젝션**
- $d_k = d_{\text{model}} / h$ : 각 Head의 차원

### 핵심: 차원 분배

전체 차원을 Head 수로 나눠서, **총 연산량은 Single-Head와 동일**합니다:

```
Single-Head:
  Q, K, V 각각: (N, 768) → Attention → (N, 768)
  총 연산: N² × 768

Multi-Head (12 heads):
  각 Head: (N, 64) → Attention → (N, 64)   [768 ÷ 12 = 64]
  12개 Head 병렬 실행 → Concat → (N, 768)
  총 연산: 12 × (N² × 64) = N² × 768  ← 같음!
```

**공짜 점심이 아닙니다** — 같은 연산량으로 더 풍부한 표현을 얻는 것입니다.

---

## 단계별 이해

{{< figure src="/images/components/attention/ko/multi-head-dimension-split.png" caption="Multi-Head Attention: 768차원을 12개 Head × 64차원으로 분리하여 병렬 Attention 수행" >}}

### 1단계: 입력을 Head별로 분리

```
입력 X: (Batch, N, 768)

QKV 프로젝션:
  QKV = X × W_qkv  → (B, N, 768×3)

Head별 분리 (reshape):
  Q: (B, 12, N, 64)   ← 12개 Head, 각 64차원
  K: (B, 12, N, 64)
  V: (B, 12, N, 64)
```

### 2단계: 각 Head가 독립적으로 Attention 수행

```
Head 0: Q₀K₀ᵀ/√64 → softmax → ×V₀  → (B, 1, N, 64)
Head 1: Q₁K₁ᵀ/√64 → softmax → ×V₁  → (B, 1, N, 64)
...
Head 11: Q₁₁K₁₁ᵀ/√64 → softmax → ×V₁₁ → (B, 1, N, 64)
```

### 3단계: 결과 합치기

```
Concat: (B, 12, N, 64) → reshape → (B, N, 768)
출력 프로젝션: (B, N, 768) × W_O → (B, N, 768)
```

---

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (효율적 구현)"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V를 한번에 계산 (메모리/속도 효율)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        x: (B, N, embed_dim) — 입력 시퀀스
        mask: (B, 1, 1, N) 또는 None — 어텐션 마스크 (옵션)
        """
        B, N, C = x.shape

        # QKV 계산 후 Head별로 분리
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention score 계산
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, N)

        # 마스크 적용 (옵션)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 가중합 + Head 합치기
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# 사용 예시: ViT-Base 설정
mha = MultiHeadAttention(embed_dim=768, num_heads=12)
x = torch.randn(32, 196, 768)  # 배치 32, 14×14 패치, 768차원
out = mha(x)
print(out.shape)  # torch.Size([32, 196, 768])
```

### PyTorch 내장 사용

```python
# PyTorch 2.0+에서는 내장 함수가 Flash Attention도 자동 적용
mha = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
out, attn_weights = mha(x, x, x)  # Self-Attention
```

---

## Head가 실제로 다른 패턴을 학습하는가?

### ViT에서 관찰된 패턴

{{< figure src="/images/components/attention/ko/multi-head-pattern-specialization.jpeg" caption="학습된 ViT의 Head별 Attention 패턴 — 로컬, 구조적, 장거리, 의미적 관계를 각 Head가 분담" >}}

학습된 ViT의 Attention Head를 시각화하면, 각 Head가 **서로 다른 패턴**에 특화됩니다:

```
Head 1: "가까운 패치에 집중"  → CNN의 로컬 필터와 유사
Head 2: "같은 행/열에 집중"  → 수평/수직 패턴 포착
Head 3: "멀리 있는 패치에 집중" → 전역 관계 학습
Head 4: "색상 유사한 패치에 집중" → 의미적 그룹핑
```

### 확인 코드

```python
def visualize_attention_heads(model, image, num_heads_to_show=4):
    """각 Head의 Attention 패턴 시각화"""
    import matplotlib.pyplot as plt

    # Attention weights 추출 (모델에 따라 방법이 다름)
    with torch.no_grad():
        # ViT의 마지막 블록에서 attention weights 추출
        attn_weights = model.get_attention_weights(image)
        # attn_weights: (B, heads, N, N)

    fig, axes = plt.subplots(1, num_heads_to_show, figsize=(4*num_heads_to_show, 4))

    # CLS 토큰(0번)이 다른 패치에 주는 attention
    for i in range(num_heads_to_show):
        attn_map = attn_weights[0, i, 0, 1:]  # CLS → 패치들
        attn_map = attn_map.reshape(14, 14)    # 14×14 격자로
        axes[i].imshow(attn_map.cpu(), cmap='viridis')
        axes[i].set_title(f'Head {i}')
        axes[i].axis('off')

    plt.suptitle('각 Head의 Attention 패턴')
    plt.tight_layout()
    plt.show()
```

---

## Attention Mask

### Causal Mask (자기회귀 모델)

GPT 같은 자기회귀 모델에서는 미래 토큰을 참조하면 안 됩니다:

```python
def causal_mask(seq_len):
    """미래 토큰을 가리는 마스크"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # 상삼각 = -inf (참조 불가)

# 사용
mask = causal_mask(10)
# [[0, -inf, -inf, ...],
#  [0,  0,  -inf, ...],
#  [0,  0,   0,   ...], ...]
# → 각 위치는 자기 자신과 이전 위치만 볼 수 있음
```

### Padding Mask

배치 내에서 시퀀스 길이가 다를 때, 패딩 토큰을 무시:

```python
def padding_mask(lengths, max_len):
    """패딩 위치를 가리는 마스크"""
    mask = torch.arange(max_len).expand(len(lengths), max_len)
    mask = mask < lengths.unsqueeze(1)
    return mask  # True = 유효, False = 패딩

# lengths = [5, 3, 7] → max_len = 7
# [[T, T, T, T, T, F, F],
#  [T, T, T, F, F, F, F],
#  [T, T, T, T, T, T, T]]
```

---

## 실전 설정값

### 주요 모델별 설정

| 모델 | embed_dim | num_heads | head_dim | 파라미터 |
|------|-----------|-----------|----------|---------|
| ViT-Small | 384 | 6 | 64 | ~22M |
| ViT-Base | 768 | 12 | 64 | ~86M |
| ViT-Large | 1024 | 16 | 64 | ~307M |
| Swin-T | 96→768 | 3→24 | 32 | ~28M |
| GPT-2 | 768 | 12 | 64 | ~117M |

**관찰**: head_dim은 대부분 **64**입니다. 모델이 커지면 Head 수가 늘어나지 차원이 커지지 않습니다.

### 왜 head_dim = 64인가?

- **너무 작으면** (32): 각 Head의 표현력이 부족
- **너무 크면** (128): Head 수가 줄어 다양한 패턴 학습 어려움
- **64**: 표현력과 다양성의 균형점

---

## Multi-Head vs Single-Head 성능 비교

```python
# 같은 파라미터 수로 비교
single = MultiHeadAttention(embed_dim=768, num_heads=1)   # head_dim=768
multi  = MultiHeadAttention(embed_dim=768, num_heads=12)  # head_dim=64

# 파라미터 수 확인
single_params = sum(p.numel() for p in single.parameters())
multi_params = sum(p.numel() for p in multi.parameters())
print(f"Single-Head: {single_params:,} params")
print(f"Multi-Head:  {multi_params:,} params")
# 둘 다 동일! → Multi-Head가 항상 더 좋은 성능
```

**실험 결과 (ViT 논문 등):**
- Multi-Head가 Single-Head보다 **일관되게 우수**
- Head 수를 늘리면 성능 향상 → 하지만 head_dim이 너무 작아지면 역효과
- 최적점: head_dim ≈ 64, 모델 크기에 따라 Head 수 조절

---

## 핵심 정리

| 질문 | 답 |
|------|---|
| 왜 여러 Head? | 하나의 Attention으로는 다양한 관계 포착이 어려움 |
| 연산량은? | Single-Head와 **동일** (차원을 나누므로) |
| 각 Head가 다른 것을 학습? | Yes — 로컬, 전역, 의미적 패턴 등이 분리됨 |
| head_dim은? | 대부분 64 (표현력과 다양성의 균형) |
| Mask는? | Causal(자기회귀), Padding(배치 정렬)에 사용 |

## 관련 콘텐츠

- [Self-Attention](/ko/docs/components/attention/self-attention) — Multi-Head의 기반
- [Cross-Attention](/ko/docs/components/attention/cross-attention) — Multi-Head Cross-Attention
- [Window Attention](/ko/docs/components/attention/window-attention) — 윈도우 기반 효율적 Attention
- [Flash Attention](/ko/docs/components/attention/flash-attention) — GPU 메모리 최적화
- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — Transformer 블록 구성
- [ViT](/ko/docs/architecture/transformer/vit) — Multi-Head Attention의 대표적 사용
