---
title: "Self-Attention"
weight: 1
math: true
---

# Self-Attention

{{% hint info %}}
**선수지식**: [행렬](/ko/docs/math/linear-algebra/matrix) | [미적분 기초](/ko/docs/math/calculus/basics)
{{% /hint %}}

## 한 줄 요약
> **시퀀스 내 모든 위치 쌍의 관계를 계산해서, 각 위치의 표현을 "맥락을 고려한 표현"으로 업데이트하는 메커니즘**

## 왜 필요한가?

### 문제 상황

문장 "고양이가 매트 위에 앉았다"를 이해할 때, 우리는 자연스럽게 이렇게 생각합니다:
- "앉았다" → **누가?** → "고양이가" (주어-동사 관계)
- "앉았다" → **어디에?** → "매트 위에" (장소 관계)

이것이 바로 **Self-Attention**입니다. 문장 안의 각 단어가 다른 모든 단어를 둘러보며 "나와 관련 있는 단어가 뭐지?"를 파악하는 것입니다.

### CNN으로는 왜 안 되나?

CNN의 커널은 **고정된 크기의 창문**입니다. 3x3 커널이면 인접한 3개만 봅니다.

```
문장: [고양이가] [매트] [위에] [앉았다]

CNN (커널 3):
"앉았다"가 볼 수 있는 범위: [매트, 위에, 앉았다]
→ "고양이가"는 보이지 않음!
→ 더 깊은 층을 쌓아야 먼 단어를 볼 수 있음

Self-Attention:
"앉았다"가 볼 수 있는 범위: [고양이가, 매트, 위에, 앉았다] 전부!
→ 한 번에 모든 단어와의 관계 파악
```

**핵심 차이**: CNN은 "가까운 것만", Self-Attention은 "모든 것을 한 번에" 봅니다.

---

## QKV: 도서관 비유

Self-Attention을 이해하는 가장 좋은 비유는 **도서관에서 책을 찾는 과정**입니다.

{{< figure src="/images/components/attention/ko/cnn-vs-attention.png" caption="CNN vs Attention: 제한된 시야 vs 전역적 시야" >}}

| 역할 | 도서관 비유 | 의미 |
|------|-----------|------|
| **Q** (Query) | 검색어 — "고양이에 대한 책 찾아줘" | "나는 무엇을 찾고 있나?" |
| **K** (Key) | 책 제목/태그 — "이 책은 동물 관련이에요" | "나는 어떤 정보를 가지고 있나?" |
| **V** (Value) | 책 내용 — 실제 텍스트 | "실제로 전달할 정보" |

**검색 과정:**
1. **Query와 Key를 비교** → "이 책이 내 검색어와 얼마나 관련 있나?" (유사도 점수)
2. **Softmax** → 점수를 확률로 변환 (합이 1이 되도록)
3. **확률 × Value** → 관련도 높은 책의 내용을 더 많이 가져옴

---

## 수식: 단계별 이해

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

복잡해 보이지만, 3단계로 나누면 간단합니다:

{{< figure src="/images/components/attention/ko/self-attention-score-matrix.png" caption="Attention Score 행렬 — 각 단어 쌍의 유사도" >}}

### 1단계: 유사도 계산 — $QK^T$

$$
\text{scores} = QK^T
$$

- $Q$: (시퀀스 길이 N) × (차원 d) 행렬
- $K^T$: (차원 d) × (시퀀스 길이 N) 행렬
- 결과: (N × N) 행렬 — **모든 위치 쌍의 유사도**

```
예시: 4개 단어 문장

         고양이가  매트  위에  앉았다
고양이가  [ 0.8    0.1   0.0   0.3 ]
매트      [ 0.1    0.7   0.5   0.2 ]
위에      [ 0.0    0.5   0.6   0.4 ]
앉았다    [ 0.3    0.2   0.4   0.5 ]

→ "앉았다" 행을 보면: 고양이가(0.3), 매트(0.2), 위에(0.4), 앉았다(0.5)
```

### 2단계: 스케일링 — $\div \sqrt{d_k}$

$$
\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}
$$

**왜 나누나?**

$d_k$가 크면 내적 값도 커집니다. 값이 너무 크면 softmax가 한 곳에 99%를 몰아주는 "극단적인" 분포가 됩니다. 이러면 gradient가 거의 0이 되어 학습이 멈춥니다.

```
d_k = 64일 때:
  스케일링 전: [32, 1, 2]  → softmax → [1.00, 0.00, 0.00] (거의 one-hot)
  스케일링 후: [4, 0.125, 0.25] → softmax → [0.95, 0.02, 0.03] (부드러운 분포)
```

$\sqrt{d_k}$로 나누면 값의 분산이 1 근처로 유지됩니다.

### 3단계: Softmax + Value 가중합

$$
\text{output} = \text{softmax}(\text{scaled\_scores}) \times V
$$

- softmax: 각 행을 확률로 변환 (합 = 1)
- 확률 × V: **중요한 단어의 정보를 더 많이 반영**

```
"앉았다"의 attention weights: [0.35, 0.15, 0.20, 0.30]

출력 = 0.35 × V(고양이가) + 0.15 × V(매트) + 0.20 × V(위에) + 0.30 × V(앉았다)
→ "고양이가 앉았다"라는 맥락이 반영된 새로운 표현
```

---

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """단일 헤드 Self-Attention (이해용)"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 입력에서 Q, K, V를 만드는 가중치 행렬
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        # x: (Batch, 시퀀스길이, 임베딩차원)
        Q = self.W_q(x)  # "무엇을 찾을까?"
        K = self.W_k(x)  # "여기 뭐가 있지?"
        V = self.W_v(x)  # "실제 정보는 이거야"

        # 1단계: 유사도 계산
        scores = Q @ K.transpose(-2, -1)  # (B, N, N)

        # 2단계: 스케일링 + softmax
        attn_weights = F.softmax(scores / self.scale, dim=-1)  # (B, N, N)

        # 3단계: 가중합
        output = attn_weights @ V  # (B, N, embed_dim)

        return output

# 사용 예시
x = torch.randn(2, 5, 64)      # 배치 2, 시퀀스 5, 차원 64
attn = SelfAttention(64)
out = attn(x)                   # (2, 5, 64) — 맥락이 반영된 표현
print(out.shape)                # torch.Size([2, 5, 64])
```

---

## Multi-Head Attention

### 왜 여러 개의 Head가 필요한가?

한 사람이 문장을 분석하면 한 가지 관점만 봅니다. 하지만 **여러 전문가**가 각자 다른 관점에서 분석하면 더 풍부한 이해가 가능합니다.

<!-- TODO: Multi-Head Attention 이미지 추가 예정 -->

예를 들어 "고양이가 매트 위에 앉았다"에서:
- **Head 1**: 문법 관계 → "앉았다"의 주어는 "고양이가"
- **Head 2**: 위치 관계 → "앉았다"의 장소는 "매트 위에"
- **Head 3**: 의미 유사성 → "고양이"와 "매트"는 관련 있음

### 수식

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

- 각 head는 독립적으로 Attention 수행
- 결과를 이어붙인 뒤(Concat), 선형 변환($W^O$)으로 합침

### 구현

```python
class MultiHeadSelfAttention(nn.Module):
    """실제 사용되는 Multi-Head Self-Attention"""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 768 // 8 = 96

        # Q, K, V를 한번에 계산 (효율)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # 출력 프로젝션
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape

        # Q, K, V 계산 후 head별로 분리
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention (head별 독립 수행)
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)

        # 가중합 후 head 합치기
        out = (attn @ v)                               # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)     # (B, N, embed_dim)

        return self.proj(out)

# 사용: ViT-Base 설정
x = torch.randn(32, 196, 768)    # 배치 32, 14×14 패치, 차원 768
mha = MultiHeadSelfAttention(768, num_heads=12)
out = mha(x)                      # (32, 196, 768)
```

**파라미터 수 비교:**

| 구성 | 계산 | 파라미터 수 |
|------|------|-----------|
| QKV 프로젝션 | 768 × (768×3) | 1,769,472 |
| 출력 프로젝션 | 768 × 768 | 589,824 |
| **합계** | | **~2.4M** |

---

## 계산 복잡도

### $O(N^2 \cdot d)$ 문제

Self-Attention의 핵심 연산은 $QK^T$입니다. N개 위치의 모든 쌍을 비교하므로 $N^2$입니다.

| 입력 | N | N² | 메모리 (FP32) |
|------|---|----|----|
| 문장 (100 토큰) | 100 | 10,000 | ~40 KB |
| ViT (14×14 패치) | 196 | 38,416 | ~150 KB |
| 고해상도 (32×32) | 1,024 | 1,048,576 | ~4 MB |
| 매우 긴 시퀀스 | 16,384 | 268M | ~1 GB |

N이 커지면 메모리가 **제곱으로** 증가합니다. 이것이 긴 시퀀스를 다루기 어려운 이유입니다.

### 해결책

| 방법 | 핵심 아이디어 | 복잡도 |
|------|-------------|--------|
| **Flash Attention** | GPU 메모리 최적화 (tiling) | $O(N^2)$ 이지만 실제로 2-4배 빠름 |
| **Window Attention** | 로컬 윈도우만 계산 (Swin Transformer) | $O(N \cdot W^2)$ |
| **Linear Attention** | kernel trick으로 근사 | $O(N \cdot d)$ |

### Flash Attention 사용법

```python
# PyTorch 2.0+ (가장 실용적인 해결책)
from torch.nn.functional import scaled_dot_product_attention

# GPU에서 자동으로 Flash Attention 적용
# 수학적으로 동일한 결과, 메모리만 효율적
attn_output = scaled_dot_product_attention(q, k, v)
```

Flash Attention은 수학적 결과는 동일하면서 메모리 사용량을 $O(N^2)$에서 $O(N)$으로 줄입니다. 속도도 2-4배 빨라집니다.

---

## 정리: Self-Attention의 핵심

| 질문 | 답 |
|------|---|
| 무엇을 하나? | 시퀀스의 모든 위치 간 관계를 계산 |
| 왜 필요한가? | CNN의 고정된 시야를 넘어 전역 관계 파악 |
| Q, K, V란? | 검색어, 태그, 실제 내용 (도서관 비유) |
| 왜 $\sqrt{d_k}$로 나누나? | 값이 너무 커지는 것 방지 → gradient 안정화 |
| Multi-Head는 왜? | 여러 관점에서 동시에 분석 |
| 단점은? | $O(N^2)$ 복잡도 → Flash Attention으로 해결 |

## 관련 콘텐츠

- [Cross-Attention](/ko/docs/components/attention/cross-attention) — 서로 다른 두 시퀀스를 연결
- [Positional Encoding](/ko/docs/components/attention/positional-encoding) — 순서 정보 주입
- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — Transformer 블록의 필수 요소
- [ViT](/ko/docs/architecture/transformer/vit) — 이미지에 Self-Attention을 적용한 모델
