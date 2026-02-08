---
title: "Cross-Attention"
weight: 2
math: true
---

# Cross-Attention

{{% hint info %}}
**선수지식**: [Self-Attention](/ko/docs/components/attention/self-attention)
{{% /hint %}}

## 한 줄 요약
> **서로 다른 두 시퀀스(예: 이미지와 텍스트)를 연결하여, 한쪽이 다른 쪽의 정보를 선택적으로 가져오는 메커니즘**

## 왜 필요한가?

### 문제 상황

Stable Diffusion에서 "귀여운 고양이가 모자를 쓴 그림"을 생성한다고 합시다. 모델은 두 가지를 동시에 처리해야 합니다:

1. **이미지**: 현재 생성 중인 그림 (노이즈에서 점점 선명해지는 중)
2. **텍스트**: 사용자의 프롬프트 "귀여운 고양이가 모자를 쓴"

이미지의 각 영역이 텍스트의 어떤 단어와 관련 있는지 알아야 합니다:
- 머리 부분 → "모자" 단어에 집중
- 얼굴 부분 → "고양이", "귀여운"에 집중
- 배경 부분 → 특별히 집중할 단어 없음

이것이 **Cross-Attention**입니다 — 한 세계(이미지)가 다른 세계(텍스트)에 **질문**하는 것입니다.

### Self-Attention과의 차이

Self-Attention은 **혼잣말**입니다. 자기 안에서 관계를 파악합니다.
Cross-Attention은 **대화**입니다. 다른 상대에게 질문하고 답을 받습니다.

```
Self-Attention:
  이미지 → [Q, K, V 모두 이미지에서] → 이미지 내부 관계 파악
  "이 패치가 저 패치와 비슷하네"

Cross-Attention:
  이미지 → [Q만 이미지에서], 텍스트 → [K, V는 텍스트에서]
  "이 이미지 영역이 어떤 텍스트 단어와 관련 있지?"
```

---

## 비유: 통역사

{{< figure src="/images/components/attention/ko/cross-attention-image-text.png" caption="Cross-Attention: 이미지 패치와 텍스트 토큰의 연결" >}}

Cross-Attention은 **통역사**와 같습니다.

| 역할 | 통역 비유 | Stable Diffusion 예시 |
|------|---------|---------------------|
| **Q** (질문) | 한국어 화자가 묻는 질문 | 이미지의 각 영역: "나는 무엇을 그려야 해?" |
| **K** (키워드) | 영어 문서의 제목/태그 | 텍스트 토큰: "cute", "cat", "hat" |
| **V** (내용) | 영어 문서의 실제 내용 | 텍스트 임베딩의 실제 정보 |
| **Attention** | 통역사가 관련 문서를 찾아 번역 | 이미지 영역과 텍스트 토큰의 매칭 |

---

## 수식

### Self-Attention (복습)

$$
\text{SelfAttn}(X) = \text{softmax}\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}}\right)(XW_V)
$$

Q, K, V **모두 같은 X**에서 나옵니다.

### Cross-Attention

$$
\text{CrossAttn}(X, C) = \text{softmax}\left(\frac{(XW_Q)(CW_K)^T}{\sqrt{d_k}}\right)(CW_V)
$$

**각 기호의 의미:**
- $X$: 대상 시퀀스 (이미지 특징) — **Query의 출처**
- $C$: 조건 시퀀스 (텍스트 임베딩) — **Key, Value의 출처**
- $W_Q, W_K, W_V$: 학습 가능한 가중치 행렬
- $d_k$: Key 차원 (스케일링용)

**핵심 차이: Q는 X에서, K와 V는 C에서 온다.**

### 단계별 이해

```
입력:
  X (이미지): 4096개 패치 × 320차원
  C (텍스트): 77개 토큰 × 768차원

1단계: Q = X × W_Q  → (4096, 320)   "이미지가 질문"
       K = C × W_K  → (77, 320)     "텍스트가 태그 제공"
       V = C × W_V  → (77, 320)     "텍스트가 정보 제공"

2단계: scores = Q × K^T  → (4096, 77)
       "이미지의 각 패치가 77개 텍스트 토큰과 얼마나 관련 있나?"

3단계: attn = softmax(scores / √d_k)  → (4096, 77)
       "관련도를 확률로 변환"

4단계: output = attn × V  → (4096, 320)
       "텍스트 정보를 이미지 각 위치에 주입"
```

**주의**: Attention 행렬이 (4096 × 77)입니다. Self-Attention의 (4096 × 4096)보다 훨씬 작습니다! Cross-Attention은 일반적으로 Self-Attention보다 연산이 가벼습니다.

---

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """Cross-Attention: 두 시퀀스를 연결"""

    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q는 대상(이미지)에서, K/V는 조건(텍스트)에서
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)   # context → query 차원으로 변환
        self.to_v = nn.Linear(context_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        """
        x:       (B, N, query_dim)   - 이미지 특징 (질문하는 쪽)
        context: (B, M, context_dim) - 텍스트 임베딩 (답하는 쪽)
        """
        B, N, C = x.shape

        # Q from image, K/V from text
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: 이미지의 각 영역이 텍스트의 어떤 토큰에 집중하는가?
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, M)
        attn = F.softmax(attn, dim=-1)

        # 텍스트 정보를 이미지에 주입
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# 사용 예시: Stable Diffusion의 Cross-Attention
image_features = torch.randn(4, 4096, 320)  # 64×64 latent, 320차원
text_embeddings = torch.randn(4, 77, 768)   # CLIP 텍스트 (77 토큰, 768차원)

cross_attn = CrossAttention(
    query_dim=320,     # 이미지 차원
    context_dim=768,   # 텍스트 차원 (CLIP)
    num_heads=8
)
out = cross_attn(image_features, text_embeddings)
print(out.shape)  # torch.Size([4, 4096, 320])
```

---

## 실제 사용 사례

### 1. Stable Diffusion — 텍스트 → 이미지 생성

Stable Diffusion의 U-Net 내부에서 Cross-Attention이 작동합니다:

```
프롬프트: "a cute cat wearing a hat"

이미지 생성 과정:
  ┌─────────────────────────────┐
  │ U-Net Block                 │
  │   ├── Self-Attention        │  이미지 내부 관계 (패치끼리)
  │   ├── Cross-Attention       │  텍스트 ↔ 이미지 연결
  │   └── FFN                   │  비선형 변환
  └─────────────────────────────┘

Cross-Attention에서:
  머리 영역 → "hat"에 집중 → 모자를 그림
  얼굴 영역 → "cat", "cute"에 집중 → 귀여운 고양이 얼굴
  전체 영역 → "wearing"에 집중 → 착용 포즈
```

### 2. VLM (Vision-Language Models) — 이미지 질문 답변

```
질문: "이 사진에서 빨간색 물체는 무엇인가요?"

Cross-Attention에서:
  질문 토큰 (Query)  ×  이미지 특징 (Key, Value)
  "빨간색" → 이미지의 빨간 영역에 집중
  "물체" → 해당 영역의 의미 파악
  → 답변: "소방차입니다"
```

### 3. Transformer Decoder — 번역

```
원문 (영어): "I love cats"
번역 (한국어): "나는 고양이를 좋아한다"

Cross-Attention에서:
  번역 토큰 (Query)  ×  원문 인코딩 (Key, Value)
  "고양이를" → "cats"에 집중
  "좋아한다" → "love"에 집중
```

---

## Cross-Attention Map 시각화

Cross-Attention의 가중치를 시각화하면 "텍스트의 어떤 단어가 이미지의 어느 부분에 대응하는지" 볼 수 있습니다.

```python
import matplotlib.pyplot as plt

def visualize_cross_attention(attn_map, text_tokens, image_size=(64, 64)):
    """
    Cross-Attention Map 시각화

    attn_map:    (num_heads, N_image, N_text) - attention weights
    text_tokens: 텍스트 토큰 리스트 (예: ["a", "cute", "cat", "hat"])
    image_size:  이미지의 H, W (latent 공간 기준)
    """
    # 모든 head의 평균
    attn = attn_map.mean(0)  # (N_image, N_text)
    H, W = image_size

    fig, axes = plt.subplots(1, len(text_tokens), figsize=(3*len(text_tokens), 3))
    for i, token in enumerate(text_tokens):
        ax = axes[i] if len(text_tokens) > 1 else axes
        # 각 텍스트 토큰에 대한 이미지 영역별 attention
        ax.imshow(attn[:, i].reshape(H, W), cmap='hot')
        ax.set_title(token, fontsize=12)
        ax.axis('off')

    plt.suptitle('Cross-Attention Map: 텍스트 → 이미지 대응', fontsize=14)
    plt.tight_layout()
    plt.show()

# 사용 예시
# attn_map = ...  # 모델에서 추출한 attention weights
# visualize_cross_attention(attn_map, ["a", "cute", "cat", "wearing", "hat"])
```

이 시각화를 통해:
- "cat" 토큰이 이미지의 고양이 영역에 높은 값을 가지는지
- "hat" 토큰이 모자 영역에 대응하는지
확인할 수 있습니다.

---

## Self vs Cross Attention 비교

| | Self-Attention | Cross-Attention |
|---|---|---|
| **Q 출처** | 자기 자신 (X) | 대상 시퀀스 (X) |
| **K, V 출처** | 자기 자신 (X) | 조건 시퀀스 (C) |
| **Attention 크기** | N × N | N × M (보통 더 작음) |
| **용도** | 내부 관계 학습 | 외부 조건 반영 |
| **비유** | 혼잣말 | 대화 (질문-답변) |
| **대표 모델** | ViT, GPT | Stable Diffusion, VLM |

---

## 정리

| 질문 | 답 |
|------|---|
| Self와 뭐가 다른가? | Q는 A에서, K/V는 B에서 → 두 시퀀스 연결 |
| 왜 필요한가? | 서로 다른 모달리티(이미지↔텍스트) 연결 |
| 어디서 쓰이나? | Stable Diffusion, VLM, 번역 |
| 연산량은? | N×M (보통 Self-Attention의 N×N보다 가벼움) |

## 관련 콘텐츠

- [Self-Attention](/ko/docs/components/attention/self-attention) — Cross-Attention의 기반
- [Positional Encoding](/ko/docs/components/attention/positional-encoding) — 위치 정보 주입
- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — Transformer 블록 구성
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) — Cross-Attention의 대표적 사용
- [CLIP](/ko/docs/architecture/multimodal/clip) — 이미지-텍스트 연결의 기반
