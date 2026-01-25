---
title: "최대 우도 추정"
weight: 12
math: true
---

# 최대 우도 추정 (Maximum Likelihood Estimation)

## 개요

> 💡 **MLE**: "데이터가 가장 잘 나올 것 같은" 파라미터 찾기

**딥러닝의 학습 = MLE**입니다!

### 시각적 이해

![최대 우도 추정](/images/probability/ko/mle.svg)

---

## 우도 함수 (Likelihood Function)

### 정의

파라미터 $\theta$가 주어졌을 때 데이터 $D$가 관측될 확률:

$$
L(\theta) = P(D | \theta) = \prod_{i=1}^{n} P(x_i | \theta)
$$

(i.i.d 가정 시)

### 확률 vs 우도

| | 확률 | 우도 |
|---|---|---|
| 고정 | 파라미터 $\theta$ | 데이터 $D$ |
| 변수 | 데이터 $D$ | 파라미터 $\theta$ |
| 의미 | $\theta$일 때 D가 나올 확률 | D가 관측됐을 때 $\theta$의 그럴듯함 |

---

## 최대 우도 추정

### 정의

$$
\hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta P(D | \theta)
$$

### Log-Likelihood

곱셈을 덧셈으로 바꾸기 위해 로그 취함:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(x_i | \theta)
$$

로그는 단조 증가 함수이므로 최대화 결과 동일.

### 음의 로그 우도 (NLL: Negative Log-Likelihood)

최대화를 최소화로:

$$
\hat{\theta}_{MLE} = \arg\min_\theta \left[ -\sum_{i=1}^{n} \log P(x_i | \theta) \right]
$$

---

## 예시: 베르누이 분포

동전을 10번 던져서 앞면이 7번 나옴. $p$의 MLE는?

### 우도 함수

$$
L(p) = p^7 (1-p)^3
$$

### Log-Likelihood

$$
\ell(p) = 7 \log p + 3 \log(1-p)
$$

### 미분하여 최대화

$$
\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0
$$

$$
\hat{p}_{MLE} = \frac{7}{10} = 0.7
$$

직관과 일치: 관측된 비율 = MLE 추정치

---

## 예시: 정규 분포

데이터 $\{x_1, ..., x_n\}$이 $\mathcal{N}(\mu, \sigma^2)$에서 왔을 때:

### Log-Likelihood

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$

### MLE 해

$$
\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}
$$

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

표본 평균과 (편향된) 표본 분산이 MLE.

---

## 딥러닝에서의 MLE

### 분류: Cross-Entropy = NLL

모델이 $P(y|x; \theta)$를 출력할 때:

$$
\text{NLL} = -\sum_{i=1}^{n} \log P(y_i | x_i; \theta)
$$

one-hot 레이블 $y$와 softmax 출력 $\hat{y}$에 대해:

$$
\text{Cross-Entropy} = -\sum_c y_c \log \hat{y}_c = -\log \hat{y}_{true}
$$

**Cross-Entropy 최소화 = MLE!**

### 회귀: MSE와 MLE

출력이 $\mathcal{N}(f_\theta(x), \sigma^2)$를 따른다고 가정하면:

$$
\text{NLL} = \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 + \text{const}
$$

**MSE 최소화 = 가우시안 가정 하에서의 MLE!**

### 요약

| 손실 함수 | MLE 관점 |
|-----------|----------|
| Cross-Entropy | 카테고리컬 분포의 NLL |
| MSE | 가우시안 분포의 NLL |
| MAE | 라플라스 분포의 NLL |

---

## MLE의 성질

### 1. 일치성 (Consistency)

$$
\hat{\theta}_{MLE} \xrightarrow{p} \theta_{true} \quad \text{as } n \rightarrow \infty
$$

데이터가 많으면 진짜 파라미터로 수렴.

### 2. 점근적 정규성 (Asymptotic Normality)

$$
\sqrt{n}(\hat{\theta}_{MLE} - \theta_{true}) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

$I(\theta)$: Fisher Information

### 3. 점근적 효율성 (Asymptotic Efficiency)

점근적으로 Cramér-Rao 하한에 도달 (분산이 가장 작음).

---

## MLE vs MAP

### MAP (Maximum A Posteriori)

사전 분포를 포함:

$$
\hat{\theta}_{MAP} = \arg\max_\theta P(\theta | D) = \arg\max_\theta P(D | \theta) P(\theta)
$$

### Log 형태

$$
\hat{\theta}_{MAP} = \arg\max_\theta \left[ \log P(D | \theta) + \log P(\theta) \right]
$$

### 정규화로의 해석

$$
\hat{\theta}_{MAP} = \arg\min_\theta \left[ \text{NLL} - \log P(\theta) \right]
$$

- $P(\theta) = \mathcal{N}(0, \sigma^2)$ → L2 정규화 (Weight Decay)
- $P(\theta) = \text{Laplace}(0, b)$ → L1 정규화

**MAP = MLE + 정규화!**

---

## 구현

```python
import numpy as np
import torch
import torch.nn as nn

# 예시 1: 베르누이 MLE
data = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 1])  # 7 성공, 3 실패
p_mle = data.mean()
print(f"베르누이 p MLE: {p_mle}")  # 0.7

# 예시 2: 정규 분포 MLE
data = np.random.normal(loc=5, scale=2, size=1000)
mu_mle = data.mean()
sigma_mle = data.std()  # MLE는 n으로 나눔
print(f"정규 분포 μ MLE: {mu_mle:.3f}")
print(f"정규 분포 σ MLE: {sigma_mle:.3f}")

# 예시 3: 신경망 학습 = MLE
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 5),  # 5 클래스
)

# Cross-Entropy Loss = NLL
criterion = nn.CrossEntropyLoss()  # 내부적으로 softmax + NLL

# 가상의 데이터
x = torch.randn(32, 10)
y = torch.randint(0, 5, (32,))

# Forward
logits = model(x)
loss = criterion(logits, y)  # NLL → 이걸 최소화 = MLE

print(f"NLL Loss: {loss.item():.4f}")

# L2 정규화 = MAP with Gaussian prior
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
# weight_decay = L2 regularization = Gaussian prior
```

---

## 핵심 정리

| 개념 | 핵심 |
|------|------|
| 우도 | $L(\theta) = P(D \| \theta)$ |
| MLE | $\arg\max_\theta P(D \| \theta)$ |
| NLL | $-\log P(D \| \theta)$ |
| CE Loss | 카테고리컬 분포의 NLL |
| MSE Loss | 가우시안 분포의 NLL |
| MAP | MLE + Prior = MLE + 정규화 |

---

## 관련 콘텐츠

- [베이즈 정리](/ko/docs/math/probability/bayes) - MAP의 기반
- [Cross-Entropy Loss](/ko/docs/math/training/loss/cross-entropy) - MLE의 구현
- [Weight Decay](/ko/docs/math/training/regularization/weight-decay) - MAP 관점
- [확률분포](/ko/docs/math/probability/distribution) - 우도 함수의 형태
