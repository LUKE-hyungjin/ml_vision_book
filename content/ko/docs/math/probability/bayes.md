---
title: "베이즈 정리"
weight: 1
math: true
---

# 베이즈 정리 (Bayes' Theorem)

## 개요

조건부 확률을 역으로 계산하는 공식으로, 관측 데이터로부터 원인의 확률을 추론합니다.

## 정의

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

- **P(A|B)**: 사후 확률 (Posterior) - B가 주어졌을 때 A의 확률
- **P(B|A)**: 가능도 (Likelihood) - A가 참일 때 B가 관측될 확률
- **P(A)**: 사전 확률 (Prior) - A에 대한 사전 믿음
- **P(B)**: 증거 (Evidence) - B가 관측될 전체 확률

## 직관적 이해

**예시**: 스팸 필터

- A = "스팸 메일"
- B = "'할인' 단어 포함"

$$
P(\text{스팸}|\text{'할인'}) = \frac{P(\text{'할인'}|\text{스팸}) \cdot P(\text{스팸})}{P(\text{'할인'})}
$$

- 스팸 메일 중 70%가 '할인' 포함
- 전체 메일 중 20%가 스팸
- 전체 메일 중 25%가 '할인' 포함

$$
P(\text{스팸}|\text{'할인'}) = \frac{0.7 \times 0.2}{0.25} = 0.56
$$

## 딥러닝에서의 활용

### 1. 분류 문제의 해석

신경망의 Softmax 출력은 사후 확률로 해석:

$$
P(y=k|x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

### 2. Bayesian Neural Networks

가중치를 확률분포로 모델링:

$$
P(W|D) = \frac{P(D|W) \cdot P(W)}{P(D)}
$$

```python
import torch
import torch.nn as nn

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 가중치의 평균과 분산을 학습
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        # 가중치 샘플링 (Reparameterization trick)
        std = torch.exp(0.5 * self.weight_logvar)
        eps = torch.randn_like(std)
        weight = self.weight_mu + std * eps
        return x @ weight.T
```

### 3. 불확실성 추정

여러 번 예측하여 불확실성 측정:

```python
model.train()  # Dropout 활성화
predictions = []
for _ in range(100):
    pred = model(x)
    predictions.append(pred)

mean = torch.stack(predictions).mean(0)  # 예측값
std = torch.stack(predictions).std(0)    # 불확실성
```

## Maximum A Posteriori (MAP)

사후 확률을 최대화하는 파라미터 찾기:

$$
\theta_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta) P(\theta)
$$

로그를 취하면:
$$
\theta_{MAP} = \arg\max_\theta [\log P(D|\theta) + \log P(\theta)]
$$

- 첫 번째 항: 데이터 가능도 → Loss
- 두 번째 항: 사전 분포 → 정규화 (Weight Decay)

## 관련 콘텐츠

- [확률분포](/ko/docs/math/probability/distribution)
- [Cross-Entropy](/ko/docs/math/training/loss/cross-entropy) - 가능도 기반 손실
- [Weight Decay](/ko/docs/math/training/regularization/weight-decay) - 사전 분포의 효과
