---
title: "최대 우도 추정"
weight: 8
math: true
---

# 최대 우도 추정 (Maximum Likelihood Estimation)

{{% hint info %}}
**선수지식**: [확률분포](/ko/docs/math/probability/distribution), [베이즈 정리](/ko/docs/math/probability/bayes) (MAP 비교용)
{{% /hint %}}

## 한 줄 요약

> **"관측된 데이터가 가장 잘 나올 것 같은 파라미터를 찾는다"**

---

## 왜 최대 우도 추정을 배워야 하나요?

### 문제 상황 1: 신경망 학습은 뭘 하는 건가요?

```python
loss = nn.CrossEntropyLoss()
loss.backward()
optimizer.step()  # 이게 대체 뭘 최적화하는 거지?
```

**정답**: 데이터가 나올 **가능도(우도)를 최대화**하는 겁니다!
- **딥러닝 학습 = 최대 우도 추정 (MLE)**

### 문제 상황 2: 왜 Cross-Entropy를 쓰나요?

```python
# 분류
loss = F.cross_entropy(logits, labels)

# 회귀
loss = F.mse_loss(pred, target)

# 왜 이 Loss들을 쓰는 거지? 다른 건 안 되나?
```

**정답**: 둘 다 **MLE에서 유도된** 손실 함수입니다!
- Cross-Entropy = 카테고리 분포의 NLL
- MSE = 가우시안 분포의 NLL

### 문제 상황 3: Weight Decay는 왜 MLE가 아닌가요?

```python
optimizer = Adam(params, weight_decay=0.01)  # 이건 뭔가 다르다고?
```

**정답**: Weight Decay는 **MAP (Maximum A Posteriori)**입니다!
- MLE: 데이터만 보고 파라미터 추정
- MAP: 데이터 + 사전 믿음으로 파라미터 추정

---

## 우도 (Likelihood)란?

### 확률 vs 우도: 핵심 차이

| | 확률 (Probability) | 우도 (Likelihood) |
|---|---|---|
| **고정** | 파라미터 $\theta$ | 데이터 $D$ |
| **변수** | 데이터 $D$ | 파라미터 $\theta$ |
| **질문** | "이 동전으로 앞면이 나올 확률은?" | "앞면이 7번 나왔는데, 동전이 공정할까?" |

### 직관적 이해

```
상황: 동전을 10번 던져서 앞면이 9번 나옴

확률 관점: "공정한 동전(p=0.5)으로 앞면 9번이 나올 확률은?"
         → P(9 heads | p=0.5) = 매우 낮음

우도 관점: "앞면 9번이 나왔는데, 이 동전의 p가 0.5일 가능성은?"
         → L(p=0.5 | 9 heads) = 매우 낮음
         → L(p=0.9 | 9 heads) = 높음!

→ 데이터를 가장 잘 설명하는 p는 0.9에 가깝다!
```

![확률 vs 우도](/images/probability/ko/probability-vs-likelihood.png)

### 우도 함수

$$
L(\theta) = P(D | \theta)
$$

데이터가 독립이면 (i.i.d.):

$$
L(\theta) = \prod_{i=1}^{n} P(x_i | \theta)
$$

---

## 최대 우도 추정 (MLE)

### 정의

$$
\hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta P(D | \theta)
$$

**해석**: "데이터 D가 가장 잘 나올 것 같은 파라미터 θ를 찾아라"

### 왜 Log를 취하는가?

**문제**: 확률의 곱은 아주 작은 숫자가 됨 (underflow)

```
L(θ) = 0.001 × 0.002 × 0.0005 × ... = 0.0000000...
```

**해결**: 로그를 취하면 곱이 합으로 바뀜

$$
\log L(\theta) = \sum_{i=1}^{n} \log P(x_i | \theta)
$$

```
log L(θ) = -6.9 + -6.2 + -7.6 + ... = -xxx (관리 가능한 숫자)
```

### Negative Log-Likelihood (NLL)

최대화 → 최소화로 바꾸기:

$$
\hat{\theta}_{MLE} = \arg\min_\theta \left[ -\sum_{i=1}^{n} \log P(x_i | \theta) \right]
$$

**이게 바로 Loss 함수입니다!**

![MLE 개념](/images/probability/ko/mle.jpeg)

---

## 예시 1: 동전 던지기 (베르누이)

### 문제

동전을 10번 던져서 앞면 7번, 뒷면 3번. $p$의 MLE는?

### 풀이

**1단계: 우도 함수**

$$
L(p) = p^7 (1-p)^3
$$

**2단계: Log-Likelihood**

$$
\ell(p) = 7 \log p + 3 \log(1-p)
$$

**3단계: 미분하여 최대화**

$$
\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0
$$

$$
7(1-p) = 3p \Rightarrow 7 - 7p = 3p \Rightarrow p = 0.7
$$

### 결과

$$
\hat{p}_{MLE} = \frac{7}{10} = 0.7
$$

**직관과 일치**: 관측된 비율 = MLE 추정치!

```python
# 코드로 확인
import numpy as np

data = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # 1=앞면, 0=뒷면
p_mle = np.mean(data)
print(f"p MLE = {p_mle}")  # 0.7
```

![MLE 동전 던지기 예시](/images/probability/ko/mle-coin-example.jpeg)

---

## 예시 2: 정규분포

### 문제

데이터 ${x_1, ..., x_n}$이 $\mathcal{N}(\mu, \sigma^2)$에서 왔을 때, $\mu$와 $\sigma^2$의 MLE는?

### 풀이

**Log-Likelihood**:

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$

**$\mu$에 대해 미분**:

$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0
$$

$$
\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}
$$

**$\sigma^2$에 대해 미분**:

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

### 결과

- $\mu$의 MLE = **표본 평균**
- $\sigma^2$의 MLE = **표본 분산** (n으로 나눔, n-1이 아님)

```python
data = np.random.normal(loc=5, scale=2, size=1000)

mu_mle = data.mean()       # 표본 평균
sigma_mle = data.std()     # ddof=0 (MLE)

print(f"μ MLE = {mu_mle:.3f}")    # ≈ 5
print(f"σ MLE = {sigma_mle:.3f}")  # ≈ 2
```

---

## 딥러닝 = MLE

### 분류: Cross-Entropy = NLL

**모델 가정**: 출력이 카테고리 분포를 따름

$$
P(y | x; \theta) = \text{Categorical}(\text{softmax}(f_\theta(x)))
$$

**NLL (Negative Log-Likelihood)**:

$$
\text{NLL} = -\sum_{i=1}^{n} \log P(y_i | x_i; \theta)
$$

**one-hot 레이블**일 때:

$$
\text{NLL} = -\sum_{i=1}^{n} \log \hat{y}_{i, true} = \text{Cross-Entropy}
$$

```python
# PyTorch에서
logits = model(x)  # [batch, num_classes]
labels = y         # [batch]

# Cross-Entropy = Softmax + NLL
loss = F.cross_entropy(logits, labels)

# 동일한 계산:
# loss = F.nll_loss(F.log_softmax(logits, dim=1), labels)
```

**결론**: Cross-Entropy 최소화 = MLE!

### 회귀: MSE = 가우시안 NLL

**모델 가정**: 출력이 가우시안 분포를 따름

$$
y | x \sim \mathcal{N}(f_\theta(x), \sigma^2)
$$

**NLL**:

$$
\text{NLL} = \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 + \text{const}
$$

$\sigma^2$가 고정이면 (상수 취급):

$$
\text{NLL} \propto \sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 = \text{MSE}
$$

```python
# PyTorch에서
pred = model(x)     # [batch]
target = y          # [batch]

# MSE = 가우시안 가정 하의 NLL
loss = F.mse_loss(pred, target)
```

**결론**: MSE 최소화 = 가우시안 가정 하의 MLE!

### 손실 함수와 분포 가정

| 손실 함수 | 가정하는 분포 | MLE 관점 |
|-----------|---------------|----------|
| **Cross-Entropy** | Categorical | $-\log P(y\|x)$ |
| **MSE** | Gaussian | $(y - \hat{y})^2$ |
| **MAE** | Laplace | $\|y - \hat{y}\|$ |
| **BCE** | Bernoulli | $-y\log\hat{y} - (1-y)\log(1-\hat{y})$ |

![손실 함수와 NLL 관계](/images/probability/ko/loss-distribution-nll.png)

---

## MLE vs MAP

### MLE의 한계

```
데이터가 적을 때:
- 동전 2번 던져서 앞면 2번
- MLE: p = 1.0 (100% 앞면!)
- 이건 과적합...
```

### MAP (Maximum A Posteriori)

**사전 분포**를 추가:

$$
\hat{\theta}_{MAP} = \arg\max_\theta P(\theta | D) = \arg\max_\theta P(D | \theta) P(\theta)
$$

로그 취하면:

$$
= \arg\max_\theta \left[ \underbrace{\log P(D | \theta)}_{\text{Log-Likelihood}} + \underbrace{\log P(\theta)}_{\text{Prior}} \right]
$$

### MAP = MLE + 정규화!

**가우시안 Prior**: $P(\theta) \sim \mathcal{N}(0, \sigma^2)$

$$
\log P(\theta) = -\frac{1}{2\sigma^2}||\theta||^2 + \text{const}
$$

$$
\hat{\theta}_{MAP} = \arg\min_\theta \left[ \text{NLL} + \frac{1}{2\sigma^2}||\theta||^2 \right]
$$

**이게 바로 L2 정규화 (Weight Decay)!**

```python
# MLE: 정규화 없음
optimizer = Adam(params)

# MAP: L2 정규화 (가우시안 Prior)
optimizer = Adam(params, weight_decay=0.01)

# weight_decay = 1/(2σ²)
# σ가 작으면 → weight_decay 크면 → 강한 정규화
```

### 비교 정리

| | MLE | MAP |
|---|---|---|
| **목표** | $\max_\theta P(D\|\theta)$ | $\max_\theta P(\theta\|D)$ |
| **사용 정보** | 데이터만 | 데이터 + 사전 믿음 |
| **정규화** | 없음 | Prior가 정규화 역할 |
| **과적합** | 발생 가능 | 방지 |
| **딥러닝** | 기본 학습 | Weight Decay |

---

## MLE의 좋은 성질

### 1) 일치성 (Consistency)

$$
\hat{\theta}_{MLE} \xrightarrow{p} \theta_{true} \quad \text{as } n \rightarrow \infty
$$

**해석**: 데이터가 충분히 많으면 진짜 파라미터에 수렴

### 2) 점근적 정규성 (Asymptotic Normality)

$$
\sqrt{n}(\hat{\theta}_{MLE} - \theta_{true}) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

**해석**: 추정치의 분포가 정규분포에 가까워짐

### 3) 효율성 (Efficiency)

MLE는 가능한 추정치 중 **분산이 가장 작음** (점근적으로)

---

## 코드로 확인하기

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 베르누이 MLE ===
print("=== 베르누이 MLE ===")
coin_flips = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # 7 성공
p_mle = np.mean(coin_flips)
print(f"관측: 앞면 {sum(coin_flips)}번 / 10번")
print(f"p MLE = {p_mle}")

# === 정규분포 MLE ===
print("\n=== 정규분포 MLE ===")
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, 1000)

mu_mle = data.mean()
sigma_mle = data.std()  # n으로 나눔 (MLE)

print(f"진짜 값: μ={true_mu}, σ={true_sigma}")
print(f"MLE: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")

# === 딥러닝: 분류 = MLE ===
print("\n=== 분류 = Cross-Entropy = NLL ===")

model = nn.Linear(10, 3)  # 3 클래스
x = torch.randn(32, 10)
y = torch.randint(0, 3, (32,))

logits = model(x)
loss_ce = F.cross_entropy(logits, y)

# 직접 NLL 계산과 비교
probs = F.softmax(logits, dim=1)
nll = -torch.log(probs[range(32), y]).mean()

print(f"CrossEntropy: {loss_ce.item():.4f}")
print(f"수동 NLL: {nll.item():.4f}")

# === 딥러닝: 회귀 = MLE ===
print("\n=== 회귀 = MSE = 가우시안 NLL ===")

model = nn.Linear(10, 1)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

pred = model(x)
loss_mse = F.mse_loss(pred, y)

# 가우시안 NLL (σ=1 가정)
sigma = 1.0
nll = 0.5 * ((y - pred) ** 2 / sigma**2).mean()

print(f"MSE: {loss_mse.item():.4f}")
print(f"가우시안 NLL (σ=1): {nll.item():.4f}")

# === MLE vs MAP ===
print("\n=== MLE vs MAP ===")

# MLE: 정규화 없음
optimizer_mle = torch.optim.Adam(model.parameters(), lr=0.01)

# MAP: L2 정규화 = 가우시안 Prior
optimizer_map = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

print("MLE: weight_decay=0")
print("MAP: weight_decay=0.01 (가우시안 Prior)")
```

---

## 핵심 정리

| 개념 | 수식/설명 | 딥러닝 적용 |
|------|-----------|-------------|
| **우도** | $L(\theta) = P(D\|\theta)$ | 데이터가 나올 가능성 |
| **MLE** | $\arg\max_\theta L(\theta)$ | 기본 학습 |
| **NLL** | $-\log L(\theta)$ | Loss 함수 |
| **Cross-Entropy** | Categorical NLL | 분류 Loss |
| **MSE** | Gaussian NLL | 회귀 Loss |
| **MAP** | MLE + Prior | Weight Decay |

---

## 핵심 통찰

```
1. 딥러닝 학습 = MLE
   - Loss 최소화 = Negative Log-Likelihood 최소화
   - = 데이터 우도 최대화

2. Loss 함수는 분포 가정에서 유도됨
   - 분류 → 카테고리 분포 → Cross-Entropy
   - 회귀 → 가우시안 분포 → MSE

3. 정규화 = MAP
   - Weight Decay = 가우시안 Prior
   - L1 Regularization = 라플라스 Prior

4. MLE의 장점
   - 일관성: 데이터 많으면 정답에 수렴
   - 효율성: 분산이 가장 작음
```

---

## 다음 단계

분포에서 **값을 뽑는** 다양한 방법이 궁금하다면?
→ [샘플링](/ko/docs/math/probability/sampling)으로!

---

## 관련 콘텐츠

- [확률분포](/ko/docs/math/probability/distribution) - 우도 함수의 형태
- [베이즈 정리](/ko/docs/math/probability/bayes) - MAP의 이론적 기반
- [엔트로피](/ko/docs/math/probability/entropy) - Cross-Entropy Loss
- [Cross-Entropy Loss](/ko/docs/components/training/loss/cross-entropy) - MLE의 실전 구현
- [정규화](/ko/docs/components/training/regularization) - MAP 관점의 정규화
