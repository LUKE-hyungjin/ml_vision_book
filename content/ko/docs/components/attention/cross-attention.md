---
title: "Cross-Attention"
weight: 3
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

### 마스크를 포함한 형태 (실전)

패딩 토큰을 제외하려면 softmax 전에 마스크를 더합니다:

$$
\text{CrossAttn}(X, C, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

- $M$ : 마스크 행렬 (유효 토큰은 0, 패딩 토큰은 매우 작은 값, 예: $-10^4$ 또는 $-\infty$)
- 실전 구현에서는 보통 $M \in \mathbb{R}^{B \times 1 \times 1 \times M}$ 형태로 broadcast해 score $(B, h, N, M)$에 더합니다.
- softmax 이후 패딩 위치의 가중치는 0에 가깝게 됩니다.

### bool 마스크 관례(실무에서 꼭 통일)

PyTorch 코드베이스마다 bool 의미가 달라 디버깅 시간이 크게 낭비됩니다. Cross-Attention에서는 아래처럼 **입력 마스크를 먼저 정규화**해 두면 안전합니다.

- 외부 입력은 `True=유효` 또는 `True=차단` 둘 다 허용
- 내부 계산은 `True=허용(keep)`으로 통일
- 점수 마스킹 시에는 `~keep`을 `masked_fill`에 사용

```python
def normalize_keep_mask(mask: torch.Tensor, *, true_means_keep: bool) -> torch.Tensor:
    """
    mask: (B, M) bool
    return: keep mask (B, M), True=허용
    """
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    return mask if true_means_keep else ~mask

# 예시: tokenizer에서 True=padding(차단)으로 준 경우
raw_mask = padding_mask_bool                     # True=차단
keep = normalize_keep_mask(raw_mask, true_means_keep=False)
score_mask = keep[:, None, None, :]              # (B,1,1,M)
scores = scores.masked_fill(~score_mask, torch.finfo(scores.dtype).min)
```

이 규칙 하나만 팀 내에서 고정해도 `dim`, `broadcast`, `~mask` 실수를 크게 줄일 수 있습니다.

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

### 직관 한 번 더: 왜 Q는 이미지, K/V는 텍스트일까?

- **Q(이미지)**는 "이 위치에 무엇을 그려야 하지?"라는 질문입니다.
- **K(텍스트)**는 "이 단어가 어떤 의미 태그를 갖는지"를 나타내는 색인입니다.
- **V(텍스트)**는 실제로 주입할 의미 정보입니다.

즉, 이미지는 질문을 만들고 텍스트는 답변 후보를 제공합니다. 그래서 Cross-Attention은 단순 결합이 아니라 **조건(condition) 주입 레이어**로 동작합니다.

### 구현에서 자주 놓치는 포인트

1. **텍스트 padding mask 적용**
   - 배치 내 문장 길이가 다르면, 실제 단어가 아닌 padding 토큰에 attention이 새어 들어갈 수 있습니다.
   - Cross-Attention score에 mask를 더해(보통 `-inf`) padding 위치를 softmax에서 제외합니다.

2. **$\sqrt{d_k}$ 스케일링 유지**
   - 스케일링을 빼먹으면 score 분산이 커져 softmax가 너무 뾰족해지고, 학습이 불안정해집니다.
   - 특히 head 차원이 큰 설정(예: 128)에서 영향이 큽니다.

3. **mixed precision에서 softmax 안정성 확인**
   - FP16/BF16에서 큰 음수 mask와 score가 섞이면 NaN이 생길 수 있습니다.
   - 실무에서는 score를 FP32로 올려 softmax 후 원래 dtype으로 내리는 패턴을 자주 씁니다.

### 빠른 shape 점검 체크리스트

디버깅 시 아래 세 줄만 먼저 출력해도 오류 원인의 80%를 찾을 수 있습니다.

- `q.shape == (B, h, N, d)`
- `k.shape == (B, h, M, d)`
- `mask.shape == (B, 1, 1, M)` (mask를 쓸 때)

이 조건이 맞으면 score shape은 자동으로 `(B, h, N, M)`가 됩니다. 이때 마지막 축 `M`에 softmax를 적용해야 "각 이미지 위치가 어떤 텍스트 토큰을 볼지"가 올바르게 계산됩니다.

### softmax 축 실수 체크 (자주 나는 버그)

Cross-Attention에서 가장 흔한 버그는 `dim`을 잘못 주는 것입니다.

```python
# 정답: 텍스트 축(M)으로 정규화
attn = F.softmax(scores, dim=-1)   # scores: (B, h, N, M)

# 오답 예시: 이미지 축(N)으로 정규화
bad = F.softmax(scores, dim=-2)
```

- `dim=-1`이면: 각 이미지 위치가 "어떤 텍스트 토큰을 볼지"를 고릅니다.
- `dim=-2`이면: 각 텍스트 토큰이 "어떤 이미지 위치를 볼지"처럼 해석이 바뀌어, 의도한 conditioning이 깨질 수 있습니다.

빠른 검증 팁:
- `attn.sum(dim=-1)`은 거의 1이어야 합니다.
- 마스크된 텍스트 위치의 평균 가중치는 0에 가까워야 합니다.

### 실전: Causal + Padding 마스크를 함께 쓸 때

디코더 Cross-Attention/Hybrid 블록에서는 두 마스크를 동시에 다루는 경우가 있습니다.
핵심은 **score와 브로드캐스트 가능한 shape로 맞춘 뒤, 불리언 마스크를 한 번에 적용**하는 것입니다.

```python
# scores: (B, h, N, M)
# causal_mask:  (N, M) bool, True=허용 / False=차단
# padding_mask: (B, M) bool, True=유효 / False=padding

allow = causal_mask[None, None, :, :] & padding_mask[:, None, None, :]
# allow: (B, 1, N, M) -> scores로 broadcast 가능

scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
```

- `masked_fill` 이전에 `allow.shape`를 출력해 `(B, 1, N, M)`인지 확인하면 디버깅 시간이 크게 줄어듭니다.
- `torch.finfo(dtype).min` 사용은 FP16/BF16에서 하드코딩 `-1e9`보다 안전한 편입니다.

### 계산 복잡도 비교

Cross-Attention의 연산량을 Self-Attention과 비교해봅시다:

| 연산 | Self-Attention | Cross-Attention |
|------|---------------|-----------------|
| **Attention 행렬** | $N \times N$ | $N \times M$ |
| **FLOPs (QK^T)** | $O(N^2 \cdot d)$ | $O(N \cdot M \cdot d)$ |
| **메모리** | $O(N^2)$ | $O(N \cdot M)$ |

**Stable Diffusion 실제 수치** (64×64 latent, CLIP 77 토큰, d=320):

| | Self-Attention | Cross-Attention | 비율 |
|---|---|---|---|
| **Attention 행렬** | 4096 × 4096 = 16.8M | 4096 × 77 = 315K | **53배 작음** |
| **메모리 (FP16)** | 33.6 MB | 630 KB | **53배 절약** |
| **FLOPs** | ~10.7 GFLOPs | ~200 MFLOPs | **53배 적음** |

→ Cross-Attention은 Self-Attention 대비 매우 가볍습니다. 전체 U-Net 연산의 대부분은 Self-Attention이 차지합니다.

### 차원 불일치 처리

{{< figure src="/images/components/attention/ko/cross-attention-dimension-gradient.jpeg" caption="Cross-Attention의 차원 투영(768→320)과 Gradient 흐름 — CLIP Frozen 시 gradient 차단" >}}

Cross-Attention의 실전 과제 중 하나는 **두 시퀀스의 차원이 다른 경우**입니다:

```
이미지: (B, 4096, 320)   — 320차원
텍스트: (B, 77, 768)     — 768차원 (CLIP)

Q = X × W_Q ∈ ℝ^(320×320)  → (B, 4096, 320)
K = C × W_K ∈ ℝ^(768×320)  → (B, 77, 320)     ← 768→320 변환!
V = C × W_V ∈ ℝ^(768×320)  → (B, 77, 320)     ← 768→320 변환!

→ W_K와 W_V가 "768차원 텍스트 공간"을 "320차원 이미지 공간"으로 투영
→ Q와 K의 내적이 가능하려면 마지막 차원(d_k)이 반드시 일치해야 함
```

이 투영(projection)이 Cross-Attention의 핵심 학습 대상입니다. 서로 다른 모달리티의 **공유 표현 공간(shared representation space)**을 학습하는 것입니다.

### Gradient 흐름

Cross-Attention의 gradient는 **양쪽 시퀀스 모두**에 흐릅니다:

```
순전파:
  X → W_Q → Q ─┐
                ├→ Attention Score → Output → Loss
  C → W_K → K ─┘
  C → W_V → V ─────────────────────┘

역전파:
  ∂Loss/∂W_Q ← Q를 통해 (이미지 측 학습)
  ∂Loss/∂W_K ← K를 통해 (텍스트→이미지 매핑 학습)
  ∂Loss/∂W_V ← V를 통해 (텍스트 정보 전달 학습)

  ∂Loss/∂X ← gradient가 이미지로 흐름
  ∂Loss/∂C ← gradient가 텍스트 인코더로 흐름 (frozen이 아닌 경우)
```

**중요**: Stable Diffusion에서 CLIP 텍스트 인코더는 보통 **frozen** (학습하지 않음)입니다. 따라서 $\partial Loss / \partial C$는 계산되지만 CLIP 가중치 업데이트에는 사용되지 않습니다. U-Net의 $W_Q$, $W_K$, $W_V$만 학습됩니다.

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
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)

        # Q는 대상(이미지)에서, K/V는 조건(텍스트)에서
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)   # context → query 차원으로 변환
        self.to_v = nn.Linear(context_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, context, context_mask=None):
        """
        x:            (B, N, query_dim)   - 이미지 특징 (질문하는 쪽)
        context:      (B, M, context_dim) - 텍스트 임베딩 (답하는 쪽)
        context_mask: (B, M) bool, True=유효 토큰 / False=padding
        """
        B, N, C = x.shape

        # Q from image, K/V from text
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: 이미지의 각 영역이 텍스트의 어떤 토큰에 집중하는가?
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, M)

        # padding 토큰(mask=False)은 softmax에서 제외
        if context_mask is not None:
            mask = context_mask[:, None, None, :]  # (B,1,1,M)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # mixed precision 안정성을 위해 softmax는 FP32에서 계산
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)

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

## 흔한 실패 패턴과 디버깅

### 1. Attention Collapse — 모든 Query가 같은 Key에 집중

```
정상:
  머리 영역 → "hat"       (0.7)
  얼굴 영역 → "cat"       (0.6)
  배경 영역 → 균일 분포    (0.15, 0.15, ...)

비정상 (Collapse):
  머리 영역 → "cat"       (0.9)
  얼굴 영역 → "cat"       (0.9)
  배경 영역 → "cat"       (0.8)
  → 모든 영역이 하나의 토큰에만 집중!
```

**원인**: 학습 초기 learning rate가 너무 크거나, 텍스트 인코딩의 특정 토큰 norm이 비정상적으로 큰 경우.

**진단**: Attention Map의 엔트로피를 모니터링합니다:

```python
import torch

def attention_entropy(attn_weights):
    """Attention 분포의 엔트로피 (높을수록 균일)"""
    # attn_weights: (B, heads, N, M)
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)
    return entropy.mean()

# 정상: 엔트로피 ~2-4 (77개 토큰 기준)
# Collapse: 엔트로피 ~0.1-0.5
```

### 2. 텍스트-이미지 불일치 — 프롬프트가 반영되지 않음

"빨간 자동차"를 요청했는데 파란 자동차가 생성되는 경우:
- Cross-Attention의 **"빨간" 토큰 attention weight가 낮은** 것이 원인
- Classifier-Free Guidance (CFG) scale을 높여 텍스트 조건을 강화하면 개선

### 3. 차원 불일치 버그

```python
# 흔한 실수: context_dim과 query_dim 혼동
cross_attn = CrossAttention(
    query_dim=320,      # 이미지 차원
    context_dim=320,    # ← 틀림! CLIP은 768차원
    num_heads=8
)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (77x768 and 320x320)
```

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
