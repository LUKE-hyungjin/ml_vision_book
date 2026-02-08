---
title: "베이즈 정리"
weight: 5
math: true
---

# 베이즈 정리 (Bayes' Theorem)

{{% hint info %}}
**선수지식**: [확률의 기초](/ko/docs/math/probability/basics) (조건부 확률)
{{% /hint %}}

## 한 줄 요약

> **"결과를 봤을 때, 원인의 확률을 역으로 계산"**

---

## 왜 베이즈 정리를 배워야 하나요?

### 문제 상황 1: AI가 "고양이"라고 했는데, 정말 고양이일까요?

```python
# AI 모델의 예측
probs = model(image)  # [0.9, 0.08, 0.02]
# "90% 확률로 고양이"라고 했지만...

# 실제로 고양이일 확률은?
# → 베이즈 정리로 계산해야 합니다!
```

**문제**: AI가 "고양이 90%"라고 해도, 데이터셋에 고양이가 희귀하면 진짜 확률은 다릅니다!

### 문제 상황 2: 검사가 양성인데, 정말 아픈 걸까요?

```
의료 AI가 "암 양성" 판정을 내렸습니다.
검사 정확도가 95%라면, 정말 암일 확률은 95%일까요?

정답: 아니요! 훨씬 낮을 수 있습니다!
```

### 문제 상황 3: Weight Decay는 왜 필요한가요?

```python
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
# weight_decay가 뭐길래 과적합을 막을까요?
```

**정답**: 가중치에 대한 **사전 믿음** (prior)을 반영하기 때문입니다!
- "가중치는 0에 가까울 거야" → 베이즈 추론

---

## 베이즈 정리 공식

### 조건부 확률의 역전

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

**해석**: "B가 관측됐을 때, A가 원인일 확률"

### 각 항의 의미

```
                    P(A|B) = P(B|A) × P(A) / P(B)
                       ↓        ↓       ↓      ↓
                    사후      가능도   사전    증거
                  (Posterior) (Likelihood) (Prior) (Evidence)
```

| 용어 | 수식 | 의미 | 예시 |
|------|------|------|------|
| **사후 확률** | $P(A\|B)$ | B를 봤을 때 A일 확률 | 양성일 때 암일 확률 |
| **가능도** | $P(B\|A)$ | A가 참이면 B가 나올 확률 | 암이면 양성 나올 확률 |
| **사전 확률** | $P(A)$ | B를 보기 전 A일 확률 | 아무나 잡았을 때 암일 확률 |
| **증거** | $P(B)$ | B가 나올 전체 확률 | 아무나 검사했을 때 양성 확률 |

![베이즈 정리](/images/probability/ko/bayes-theorem.png)

---

## 예시 1: 의료 검사 (베이즈의 역설)

### 문제

```
• 희귀 질병 유병률: 1% (100명 중 1명)
• 검사 민감도: 99% (환자를 양성으로 판정할 확률)
• 검사 특이도: 95% (정상인을 음성으로 판정할 확률)

→ 양성 판정을 받으면, 정말 아플 확률은?
```

### 직관적인 (잘못된) 답

"검사가 99% 정확하니까, 양성이면 99% 아프겠지?" → **틀림!**

### 베이즈로 정확히 계산

$$
P(\text{질병}|\text{양성}) = \frac{P(\text{양성}|\text{질병}) \cdot P(\text{질병})}{P(\text{양성})}
$$

**1단계**: 전체 양성 확률 (분모) 계산

$$
P(\text{양성}) = P(\text{양성}|\text{질병})P(\text{질병}) + P(\text{양성}|\text{정상})P(\text{정상})
$$
$$
= 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594
$$

**2단계**: 베이즈 정리 적용

$$
P(\text{질병}|\text{양성}) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.167
$$

### 결과 해석

```
검사가 양성이어도, 진짜 질병일 확률은 약 16.7%!

왜 이렇게 낮을까?

10,000명을 검사한다고 생각해보세요:
• 실제 환자: 100명
  - 양성 판정: 99명 (99%)
  - 음성 판정: 1명

• 정상인: 9,900명
  - 양성 판정: 495명 (5% 오진)
  - 음성 판정: 9,405명

→ 양성 판정 594명 중 진짜 환자는 99명
→ 99 / 594 ≈ 16.7%

질병이 희귀하면, 오진(495명)이 실제 환자(99명)보다 많음!
```

![의료 검사 베이즈](/images/probability/ko/medical-test-bayes.jpeg)

---

## 예시 2: AI 분류기의 신뢰도

### 문제

```
• 데이터셋에 고양이가 10%, 강아지가 90%
• 모델이 "고양이"라고 할 때 실제 고양이일 확률: 95%
• 모델이 "강아지"라고 할 때 실제 강아지일 확률: 90%

→ 모델이 "고양이"라고 하면, 정말 고양이일까?
```

### 베이즈 적용

$$
P(\text{고양이}|\text{모델: 고양이}) = \frac{P(\text{모델: 고양이}|\text{고양이}) \cdot P(\text{고양이})}{P(\text{모델: 고양이})}
$$

**분모 계산**:
$$
P(\text{모델: 고양이}) = 0.95 \times 0.10 + 0.10 \times 0.90 = 0.095 + 0.09 = 0.185
$$

**베이즈**:
$$
P(\text{고양이}|\text{모델: 고양이}) = \frac{0.95 \times 0.10}{0.185} \approx 0.514
$$

### 결과

```
모델이 "고양이"라고 해도, 진짜 고양이일 확률은 약 51%!

왜? 데이터셋에 강아지가 90%라서,
모델이 "고양이"라고 한 것 중 상당수가 강아지의 오분류!
```

**교훈**: 클래스 불균형이 심하면 모델 출력을 그대로 믿으면 안 됩니다!

---

## 딥러닝에서 베이즈

### 1) 분류 = 사후 확률 추정

신경망은 $P(\text{클래스}|\text{입력})$을 학습합니다:

```python
# 분류 모델의 본질
P(y=고양이 | x=이미지) = model(image)[0]

# Softmax 출력 = 사후 확률의 근사
probs = torch.softmax(logits, dim=-1)
# probs[k] ≈ P(y=k | x)
```

### 2) Weight Decay = 사전 분포

$$
\theta_{MAP} = \arg\max_\theta \underbrace{P(D|\theta)}_{\text{가능도}} \cdot \underbrace{P(\theta)}_{\text{사전}}
$$

로그를 취하면:
$$
= \arg\max_\theta [\log P(D|\theta) + \log P(\theta)]
$$

**가우시안 사전** $P(\theta) \sim \mathcal{N}(0, \sigma^2)$를 가정하면:

$$
\log P(\theta) = -\frac{1}{2\sigma^2}||\theta||^2 + \text{const}
$$

```python
# 결과적으로:
# loss = NLL + (1/2σ²) × ||W||²

# PyTorch에서:
optimizer = Adam(params, weight_decay=0.01)  # λ = 1/2σ²

# weight_decay = 가중치가 0에 가까울 거라는 사전 믿음!
```

**해석**:
- Weight Decay가 크면 → "가중치가 작을 거야"라는 강한 사전 믿음
- Weight Decay가 작으면 → 데이터에 더 의존

### 3) Bayesian Neural Network

가중치를 **확률분포**로 모델링:

```python
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 가중치의 평균과 분산을 학습
        self.w_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.w_logvar = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        # 가중치 샘플링 (매번 다른 가중치!)
        std = torch.exp(0.5 * self.w_logvar)
        eps = torch.randn_like(std)
        w = self.w_mu + std * eps
        return x @ w.T

    def kl_loss(self):
        # KL(q(w) || p(w)) - 사전분포와의 차이
        return -0.5 * torch.sum(1 + self.w_logvar - self.w_mu.pow(2) - self.w_logvar.exp())
```

**장점**: 불확실성을 정량화할 수 있음!

```python
# 여러 번 예측해서 불확실성 측정
predictions = [model(x) for _ in range(100)]
mean = torch.stack(predictions).mean(0)  # 예측값
std = torch.stack(predictions).std(0)    # 불확실성

# std가 크면 → 모델이 불확실해함
# std가 작으면 → 모델이 확신함
```

### 4) Dropout as Bayesian Approximation

Dropout도 베이지안 근사입니다!

```python
# MC Dropout: 테스트 시에도 Dropout 유지
model.train()  # Dropout 활성화 상태

predictions = []
for _ in range(100):
    pred = model(x)  # 매번 다른 뉴런이 꺼짐
    predictions.append(pred)

mean = torch.stack(predictions).mean(0)
uncertainty = torch.stack(predictions).std(0)
```

---

## 베이즈 업데이트: 믿음이 바뀌는 과정

### 핵심 아이디어

> "새로운 증거를 볼 때마다 믿음을 업데이트"

$$
\text{사후} \propto \text{가능도} \times \text{사전}
$$

### 예시: 동전 공정성 추정

```
목표: 이 동전이 공정한가? (앞면 확률 = 0.5?)

사전 믿음: "아마 공정하겠지" → P(p=0.5) 높음

관측 1: 앞면 나옴
→ 사후 업데이트: 여전히 0.5 근처

관측 2~10: 앞면 9번, 뒷면 1번
→ 사후 업데이트: p > 0.5 쪽으로 이동

관측 11~100: 앞면 80번, 뒷면 20번
→ 사후 확률: p ≈ 0.8에 집중
```

![베이지안 업데이트](/images/probability/ko/bayesian-update.png)

---

## 코드로 확인하기

```python
import numpy as np

# === 의료 검사 예시 ===
print("=== 베이즈 정리: 의료 검사 ===")

# 주어진 확률
p_disease = 0.01       # 유병률 1%
p_positive_given_disease = 0.99  # 민감도 99%
p_negative_given_healthy = 0.95  # 특이도 95%
p_positive_given_healthy = 1 - p_negative_given_healthy  # 위양성률 5%

# P(양성) = P(양성|질병)P(질병) + P(양성|정상)P(정상)
p_positive = (p_positive_given_disease * p_disease +
              p_positive_given_healthy * (1 - p_disease))

# 베이즈 정리
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"검사 양성일 때 진짜 질병일 확률: {p_disease_given_positive:.1%}")
print(f"(검사 정확도 99%인데, 실제로는 {p_disease_given_positive:.1%}!)")

# === 시뮬레이션으로 확인 ===
print("\n=== 시뮬레이션 검증 (10,000명) ===")
n_people = 10000
n_sick = int(n_people * p_disease)
n_healthy = n_people - n_sick

# 검사 결과 시뮬레이션
true_positives = np.random.binomial(n_sick, p_positive_given_disease)
false_positives = np.random.binomial(n_healthy, p_positive_given_healthy)

total_positives = true_positives + false_positives
p_sick_if_positive = true_positives / total_positives

print(f"진양성: {true_positives}, 위양성: {false_positives}")
print(f"양성 {total_positives}명 중 진짜 환자: {true_positives}명")
print(f"시뮬레이션 결과: {p_sick_if_positive:.1%}")

# === 베이지안 업데이트: 동전 ===
print("\n=== 베이지안 업데이트: 동전 공정성 ===")
from scipy import stats

# 사전: Beta(1, 1) = 균등분포 (아무것도 모름)
a_prior, b_prior = 1, 1

# 관측: 앞면 70번, 뒷면 30번
heads, tails = 70, 30

# 사후: Beta(a + heads, b + tails)
a_post = a_prior + heads
b_post = b_prior + tails

posterior = stats.beta(a_post, b_post)

print(f"관측: 앞면 {heads}번, 뒷면 {tails}번")
print(f"사후 평균: {posterior.mean():.3f}")
print(f"95% 신뢰구간: [{posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f}]")

# === Weight Decay와 베이즈 ===
print("\n=== Weight Decay = 가우시안 사전 ===")
import torch
import torch.nn as nn

# 정규화 없이
model1 = nn.Linear(10, 1)
nn.init.normal_(model1.weight, std=1.0)
print(f"정규화 전 가중치 크기: {model1.weight.norm():.3f}")

# Weight Decay는 학습 중에 가중치를 0으로 당김
# 이는 P(W) ~ N(0, σ²) 사전분포를 가정한 것과 같음
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 적용 |
|------|------|-------------|
| **베이즈 정리** | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ | 분류의 이론적 기반 |
| **사전 확률** | $P(\theta)$ | Weight Decay, 정규화 |
| **가능도** | $P(D\|\theta)$ | Loss 함수 |
| **사후 확률** | $P(\theta\|D)$ | 학습된 모델 |
| **MAP 추정** | $\arg\max P(\theta\|D)$ | 정규화된 학습 |

---

## 핵심 통찰

```
1. 베이즈 정리 = 원인과 결과를 뒤집는 공식

2. 사전 확률이 중요함!
   - 희귀한 사건은 증거가 있어도 확률이 낮음
   - 클래스 불균형에서 모델 출력을 그대로 믿으면 안 됨

3. Weight Decay = 베이지안 사전
   - "가중치가 작을 거야"라는 믿음
   - 사전 믿음이 강하면 → 과적합 방지

4. 불확실성을 정량화할 수 있음
   - Bayesian NN, MC Dropout
```

---

## 다음 단계

**확률의 "불확실성"**을 숫자로 측정하고 싶다면?
→ [엔트로피](/ko/docs/math/probability/entropy)로!

분포 간의 **"차이"**를 측정하고 싶다면?
→ [KL 발산](/ko/docs/math/probability/kl-divergence)으로!

---

## 관련 콘텐츠

- [확률의 기초](/ko/docs/math/probability/basics) - 조건부 확률
- [확률분포](/ko/docs/math/probability/distribution) - 가우시안 사전
- [엔트로피](/ko/docs/math/probability/entropy) - 불확실성 측정
- [KL 발산](/ko/docs/math/probability/kl-divergence) - 사전/사후 차이
- [최대 우도 추정](/ko/docs/math/probability/mle) - MLE vs MAP
- [Weight Decay](/ko/docs/components/training/regularization) - 사전 분포의 실제 적용
