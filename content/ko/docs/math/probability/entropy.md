---
title: "엔트로피"
weight: 6
math: true
---

# 엔트로피 (Entropy)

{{% hint info %}}
**선수지식**: [확률분포](/ko/docs/math/probability/distribution), [기댓값](/ko/docs/math/probability/expectation)
{{% /hint %}}

## 한 줄 요약

> **"얼마나 불확실한가?"를 숫자로 측정하는 방법**

---

## 왜 엔트로피를 배워야 하나요?

### 문제 상황 1: Cross-Entropy Loss가 뭔가요?

```python
loss = nn.CrossEntropyLoss()
output = model(images)
loss_value = loss(output, labels)  # 이게 대체 뭘 계산하는 거지?
```

**정답**: 모델 예측의 **"불확실성"**을 측정하는 겁니다!
- 엔트로피를 모르면 → Cross-Entropy Loss 이해 불가

### 문제 상황 2: 모델이 "확신"하는 정도를 어떻게 측정할까요?

```python
# 두 모델의 예측
model_A: [0.98, 0.01, 0.01]  # 확신함
model_B: [0.34, 0.33, 0.33]  # 불확실함

# 어떤 모델이 더 확신하는지 수치화하려면?
```

**정답**: 엔트로피로 측정!
- model_A 엔트로피 낮음 → 확신함
- model_B 엔트로피 높음 → 불확실함

### 문제 상황 3: Label Smoothing은 왜 쓰나요?

```python
# 일반 학습
target = [0, 0, 1, 0]  # one-hot

# Label Smoothing
target = [0.025, 0.025, 0.925, 0.025]  # 왜 이렇게?
```

**정답**: 타겟의 **엔트로피를 높여서** 과신(overconfidence)을 방지합니다!

---

## 1. 정보량 (Information): 놀라움의 크기

### 핵심 직관

> "확률이 낮은 일이 일어나면 더 놀랍다 = 정보가 많다"

**예시**:
```
"내일 해가 뜬다" → 당연함 → 정보량 적음
"내일 눈이 5m 온다" → 놀라움 → 정보량 많음
```

### 수학적 정의

$$
I(x) = -\log P(x) = \log \frac{1}{P(x)}
$$

**왜 로그인가?**
1. 확률이 낮을수록 → 정보량이 높아야 함 (역비례)
2. 독립 사건의 정보량은 더해져야 함 (곱 → 합)

### 예시 계산

```
공정한 동전 앞면:
I = -log₂(0.5) = log₂(2) = 1 bit

주사위 1 나옴:
I = -log₂(1/6) = log₂(6) ≈ 2.58 bits

로또 1등 당첨 (확률 1/8,145,060):
I = -log₂(1/8,145,060) ≈ 23 bits
```

**시각화**:
```
정보량 I(x)
    │
    │ \
    │  \
    │   \
    │    ╲
    │     ──────────
    └────────────────→ P(x)
    0               1

확률이 낮을수록 정보량이 큼!
```

![정보량](/images/probability/ko/information-quantity.svg)

---

## 2. 엔트로피 = 평균 정보량

### 정의

> "분포 전체의 불확실성" = 정보량의 기댓값

$$
H(X) = \mathbb{E}[-\log P(X)] = -\sum_x P(x) \log P(x)
$$

### 직관적 해석

```
엔트로피 = "결과를 맞추기 위해 평균적으로 필요한 질문 수"

예시: 8개 중 하나 맞추기 (균등분포)
Q1: "1~4 중에 있어?" → 반 제거
Q2: "1~2 중에 있어?" → 또 반 제거
Q3: "1이야?" → 정답!

→ 평균 3번 질문 필요 = log₂(8) = 3 bits
→ 균등분포의 엔트로피 = log₂(8) = 3
```

### 엔트로피의 극단적 경우

```
결정적 분포 (엔트로피 = 0)        균등 분포 (엔트로피 = 최대)

P(x)                             P(x)
  │                                │
1 ├─█                           ──┼──────────
  │                                │ █ █ █ █
  └──┴──┴──┴──→ x                  └──┴──┴──┴──→ x
     결과가 뻔함                     뭐가 나올지 모름
     H = 0                          H = log(n)
```

---

## 3. 중요한 분포의 엔트로피

### 베르누이 분포

$$
H(p) = -p \log p - (1-p) \log(1-p)
$$

```
H(p)
  │      ╭───╮
1 ├─────╱     ╲─────
  │    ╱       ╲
  │   ╱         ╲
  │  ╱           ╲
0 ├─╱─────────────╲─→ p
  0      0.5      1

• p = 0.5일 때 최대 (가장 불확실)
• p = 0 또는 1일 때 0 (확정적)
```

![베르누이 엔트로피](/images/probability/ko/bernoulli-entropy.svg)

**딥러닝 적용**: 이진 분류에서 예측 확신도 측정

### 카테고리 분포

$$
H = -\sum_{i=1}^{K} p_i \log p_i
$$

최대 엔트로피: $H_{max} = \log K$ (균등분포일 때)

**딥러닝 적용**: 분류 모델의 출력 불확실성

### 가우시안 분포 (미분 엔트로피)

$$
H = \frac{1}{2} \log(2\pi e \sigma^2)
$$

**해석**: 분산이 클수록 엔트로피가 높음 (더 퍼져있으니까)

---

## 4. Cross-Entropy = 딥러닝의 핵심 Loss

### 정의

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

**해석**:
- $p$: 실제 분포 (정답)
- $q$: 예측 분포 (모델)
- Cross-Entropy: "정답 p를 예측 q로 표현할 때의 평균 비트 수"

### 왜 Cross-Entropy인가?

```
Cross-Entropy = Entropy + KL Divergence

H(p, q) = H(p) + D_KL(p || q)
          ─────   ────────────
          상수     최소화 대상
          (레이블)  (p와 q를 같게!)

H(p)는 정답 분포 → 학습으로 바뀌지 않음 → 상수
D_KL(p||q)를 줄이면 → q가 p에 가까워짐 → Cross-Entropy 감소

결론: Cross-Entropy 최소화 = 예측을 정답에 맞추기!
```

![엔트로피와 Cross-Entropy](/images/probability/ko/entropy-crossentropy.svg)

### 다중 클래스 분류에서의 Cross-Entropy

$$
\mathcal{L} = -\sum_{c=1}^{C} y_c \log \hat{y}_c
$$

**one-hot 인코딩** ($y = [0, 0, 1, 0]$, 정답 = 클래스 2):

$$
\mathcal{L} = -\log \hat{y}_2
$$

```python
# 예시: 정답이 클래스 2
y_true = [0, 0, 1, 0]

# 좋은 예측
y_pred_good = [0.05, 0.05, 0.85, 0.05]
loss_good = -log(0.85) ≈ 0.16

# 나쁜 예측
y_pred_bad = [0.25, 0.25, 0.25, 0.25]
loss_bad = -log(0.25) ≈ 1.39

# 완전 잘못된 예측
y_pred_wrong = [0.9, 0.05, 0.03, 0.02]
loss_wrong = -log(0.03) ≈ 3.51
```

**시각화**:
```
Loss
  │
  │  \
  │   \
3 ├────\
  │     \
1 ├──────────────
  │              ────────
0 └──────────────────────→ ŷ_correct
  0        0.5           1

예측 확률이 높을수록 Loss가 낮음!
```

![Cross-Entropy Loss 곡선](/images/probability/ko/cross-entropy-loss-curve.svg)

### PyTorch에서 Cross-Entropy

```python
import torch
import torch.nn.functional as F

# 방법 1: CrossEntropyLoss (logits 입력)
logits = torch.tensor([[2.0, 1.0, 0.5]])  # softmax 전
target = torch.tensor([0])  # 정답 인덱스

loss = F.cross_entropy(logits, target)
print(f"Loss: {loss.item():.4f}")

# 내부적으로:
# 1. softmax(logits) → [0.659, 0.242, 0.099]
# 2. -log(0.659) = 0.417

# 방법 2: NLLLoss (log_softmax 후)
log_probs = F.log_softmax(logits, dim=1)
loss = F.nll_loss(log_probs, target)

# 방법 3: BCELoss (이진 분류)
pred = torch.tensor([0.7])
target = torch.tensor([1.0])
loss = F.binary_cross_entropy(pred, target)
```

---

## 5. 엔트로피의 딥러닝 활용

### 1) 예측 불확실성 측정

```python
def prediction_entropy(probs):
    """예측의 불확실성 측정"""
    # probs: softmax 출력 [C]
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    return entropy

# 확신하는 예측
confident = torch.tensor([0.95, 0.03, 0.02])
print(f"확신하는 예측 엔트로피: {prediction_entropy(confident):.3f}")  # 낮음

# 불확실한 예측
uncertain = torch.tensor([0.35, 0.35, 0.30])
print(f"불확실한 예측 엔트로피: {prediction_entropy(uncertain):.3f}")  # 높음

# 활용: 불확실한 샘플만 수동 검토
if prediction_entropy(probs) > threshold:
    send_to_human_review(sample)
```

### 2) Label Smoothing

**문제**: one-hot 타겟은 엔트로피가 0 → 모델이 과신하게 됨

**해결**: 타겟에 약간의 불확실성 추가

```python
def label_smoothing(labels, num_classes, smoothing=0.1):
    """
    [0, 0, 1, 0] → [0.025, 0.025, 0.925, 0.025]
    """
    confidence = 1.0 - smoothing
    smooth_value = smoothing / num_classes

    one_hot = F.one_hot(labels, num_classes).float()
    return one_hot * confidence + smooth_value

# 사용
labels = torch.tensor([2])
smooth_labels = label_smoothing(labels, num_classes=4, smoothing=0.1)
# → [0.025, 0.025, 0.925, 0.025]

# 엔트로피 비교
H_onehot = 0  # 확정적
H_smooth = -sum([0.025*log(0.025)*3 + 0.925*log(0.925)]) > 0  # 불확실성 있음
```

**효과**:
- 모델이 100% 확신하지 않도록 유도
- 일반화 성능 향상

### 3) Entropy Regularization

목표: 예측이 너무 확신하지 않도록 (또는 확신하도록)

```python
def entropy_regularized_loss(logits, targets, beta=0.1):
    """
    beta > 0: 예측이 균등해지도록 (탐험 장려)
    beta < 0: 예측이 확신하도록 (결정적 장려)
    """
    ce_loss = F.cross_entropy(logits, targets)

    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()

    return ce_loss - beta * entropy  # 엔트로피가 높으면 loss 감소

# 강화학습에서 많이 사용: 탐험-활용 균형
```

### 4) Knowledge Distillation

Teacher의 "soft label"은 엔트로피가 높음:

```python
# Teacher 출력 (soft)
teacher_output = [0.7, 0.2, 0.1]  # 엔트로피 있음 → 정보 풍부

# Hard label
hard_label = [1, 0, 0]  # 엔트로피 0 → 정보 손실

# Student는 soft label에서 더 많은 정보를 배움!
```

---

## 6. 조건부 엔트로피

### 정의

$$
H(Y|X) = -\sum_{x,y} P(x,y) \log P(y|x)
$$

**해석**: "X를 알 때 Y의 남은 불확실성"

### 핵심 성질

$$
H(Y|X) \leq H(Y)
$$

"정보를 더 알면 불확실성은 줄어든다 (또는 같다)"

### 예시

```
H(날씨) = 높음 (맑음, 흐림, 비, 눈 등 다양)

H(날씨 | 구름 상태) = 더 낮음
  - 구름 없음 → 아마 맑음
  - 구름 많음 → 아마 비

정보(구름 상태)가 불확실성을 줄였다!
```

---

## 7. 상호 정보량 (Mutual Information)

### 정의

$$
I(X; Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)
$$

**해석**: "X와 Y가 공유하는 정보량"

### 시각적 이해

```
┌──────────────────────────────────┐
│           H(X,Y)                 │
│  ┌────────────────────────────┐  │
│  │         H(X)               │  │
│  │    ┌──────────┐            │  │
│  │    │  I(X;Y)  │    H(Y)    │  │
│  │    │  공유     │            │  │
│  │    └──────────┘            │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘

I(X;Y) = H(X) + H(Y) - H(X,Y)
```

### 딥러닝 활용

**InfoNCE Loss (Contrastive Learning)**:
- 목표: 같은 것끼리 I(X;Y) 높이기
- 서로 다른 것끼리 I(X;Y) 낮추기

```python
# SimCLR의 핵심
# 같은 이미지의 다른 augmentation → 상호정보량 최대화
```

![상호 정보량](/images/probability/ko/mutual-information.svg)

---

## 코드로 확인하기

```python
import torch
import numpy as np

# === 엔트로피 계산 ===
def entropy(probs):
    """엔트로피 계산 (자연로그)"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # log(0) 방지
    return -np.sum(probs * np.log(probs))

def entropy_bits(probs):
    """엔트로피 계산 (비트)"""
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# 다양한 분포의 엔트로피
print("=== 다양한 분포의 엔트로피 ===")

# 균등 분포 (4개)
uniform = [0.25, 0.25, 0.25, 0.25]
print(f"균등 분포: {entropy_bits(uniform):.3f} bits (최대: {np.log2(4):.3f})")

# 편향된 분포
biased = [0.9, 0.05, 0.03, 0.02]
print(f"편향된 분포: {entropy_bits(biased):.3f} bits")

# 확정적 분포
deterministic = [1.0, 0.0, 0.0, 0.0]
print(f"확정적 분포: {entropy_bits(deterministic):.3f} bits")

# === Cross-Entropy 계산 ===
print("\n=== Cross-Entropy Loss ===")

def cross_entropy(p, q):
    """Cross-Entropy H(p, q)"""
    p = np.array(p)
    q = np.array(q)
    q = np.clip(q, 1e-15, 1)  # log(0) 방지
    return -np.sum(p * np.log(q))

# 정답: 클래스 2
y_true = [0, 0, 1, 0]

# 좋은 예측
y_good = [0.05, 0.05, 0.85, 0.05]
print(f"좋은 예측 CE Loss: {cross_entropy(y_true, y_good):.4f}")

# 나쁜 예측
y_bad = [0.25, 0.25, 0.25, 0.25]
print(f"나쁜 예측 CE Loss: {cross_entropy(y_true, y_bad):.4f}")

# 완전 잘못된 예측
y_wrong = [0.9, 0.05, 0.03, 0.02]
print(f"틀린 예측 CE Loss: {cross_entropy(y_true, y_wrong):.4f}")

# === PyTorch Cross-Entropy ===
print("\n=== PyTorch Cross-Entropy ===")
import torch.nn.functional as F

logits = torch.tensor([[1.0, 2.0, 5.0, 1.0]])  # 클래스 2가 가장 높음
target = torch.tensor([2])

loss = F.cross_entropy(logits, target)
probs = F.softmax(logits, dim=1)
print(f"Softmax 출력: {probs.tolist()[0]}")
print(f"Cross-Entropy Loss: {loss.item():.4f}")

# === 예측 불확실성 ===
print("\n=== 예측 불확실성 측정 ===")

confident_pred = torch.tensor([0.95, 0.03, 0.02])
uncertain_pred = torch.tensor([0.35, 0.35, 0.30])

H_confident = -torch.sum(confident_pred * torch.log(confident_pred))
H_uncertain = -torch.sum(uncertain_pred * torch.log(uncertain_pred))

print(f"확신하는 예측의 엔트로피: {H_confident:.4f}")
print(f"불확실한 예측의 엔트로피: {H_uncertain:.4f}")

# === Label Smoothing ===
print("\n=== Label Smoothing ===")
smoothing = 0.1
num_classes = 4

one_hot = torch.tensor([0, 0, 1, 0]).float()
smooth_label = one_hot * (1 - smoothing) + smoothing / num_classes

print(f"One-hot: {one_hot.tolist()}")
print(f"Smooth:  {smooth_label.tolist()}")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 적용 |
|------|------|-------------|
| **정보량** | $I(x) = -\log P(x)$ | 드문 사건의 정보가 많음 |
| **엔트로피** | $H(X) = -\sum p \log p$ | 예측 불확실성 측정 |
| **Cross-Entropy** | $H(p,q) = -\sum p \log q$ | 분류 Loss |
| **조건부 엔트로피** | $H(Y\|X)$ | 추가 정보의 가치 |
| **상호 정보량** | $I(X;Y) = H(Y) - H(Y\|X)$ | Contrastive Learning |

---

## 핵심 통찰

```
1. 엔트로피 = 불확실성의 크기
   - 높으면: 불확실, 균등분포에 가까움
   - 낮으면: 확신, 결정적에 가까움

2. Cross-Entropy Loss의 본질
   - 예측을 정답에 가깝게 만들기
   - CE = Entropy + KL Divergence

3. 엔트로피 활용
   - 예측 신뢰도 측정
   - Label Smoothing으로 과신 방지
   - Knowledge Distillation
```

---

## 다음 단계

두 분포가 **얼마나 다른지** 측정하고 싶다면?
→ [KL 발산](/ko/docs/math/probability/kl-divergence)으로!

**Cross-Entropy Loss**의 실제 사용법이 궁금하다면?
→ [Cross-Entropy Loss](/ko/docs/math/training/loss/cross-entropy)로!

---

## 관련 콘텐츠

- [확률분포](/ko/docs/math/probability/distribution) - 분포별 엔트로피
- [기댓값](/ko/docs/math/probability/expectation) - 기댓값으로서의 엔트로피
- [KL 발산](/ko/docs/math/probability/kl-divergence) - 분포 간 차이
- [Cross-Entropy Loss](/ko/docs/math/training/loss/cross-entropy) - 실제 Loss 함수
