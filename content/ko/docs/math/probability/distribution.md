---
title: "확률분포"
weight: 2
math: true
---

# 확률분포 (Probability Distributions)

## 개요

확률분포는 확률변수가 특정 값을 가질 가능성을 나타내는 함수입니다.

## 이산 분포

### 베르누이 분포 (Bernoulli)

동전 던지기처럼 성공/실패 두 가지 결과:

$$
P(X=1) = p, \quad P(X=0) = 1-p
$$

**딥러닝 적용**: Dropout

```python
import torch

p = 0.5  # 드롭 확률
mask = torch.bernoulli(torch.full((10,), 1-p))  # 0 또는 1
x = x * mask / (1 - p)  # Inverted dropout
```

### 카테고리 분포 (Categorical)

K개의 클래스 중 하나 선택:

$$
P(X=k) = p_k, \quad \sum_{k=1}^K p_k = 1
$$

**딥러닝 적용**: Softmax 출력

```python
logits = model(x)  # (batch, num_classes)
probs = torch.softmax(logits, dim=-1)  # 카테고리 분포
```

## 연속 분포

### 가우시안 분포 (Gaussian / Normal)

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**딥러닝 적용**: VAE의 잠재 공간, 노이즈 모델링

```python
mu = encoder_mu(x)
log_var = encoder_logvar(x)

# Reparameterization trick
std = torch.exp(0.5 * log_var)
eps = torch.randn_like(std)  # N(0, 1)에서 샘플링
z = mu + std * eps  # N(mu, sigma^2)에서 샘플링
```

### 다변량 가우시안

$$
p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$

## Softmax와 확률

Softmax는 임의의 실수 벡터를 확률분포로 변환:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

속성:
- 모든 출력 > 0
- 합 = 1
- 가장 큰 값이 가장 높은 확률

```python
def softmax(z):
    exp_z = torch.exp(z - z.max(dim=-1, keepdim=True).values)  # 수치 안정성
    return exp_z / exp_z.sum(dim=-1, keepdim=True)

z = torch.tensor([2.0, 1.0, 0.1])
print(softmax(z))  # [0.659, 0.242, 0.099]
```

## KL Divergence

두 확률분포 간의 거리:

$$
D_{KL}(P||Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

속성:
- D_KL ≥ 0
- D_KL = 0 ⟺ P = Q
- 비대칭: D_KL(P||Q) ≠ D_KL(Q||P)

**딥러닝 적용**: VAE Loss

```python
def kl_divergence(mu, log_var):
    # N(mu, sigma^2)와 N(0, 1) 사이의 KL
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

## 관련 콘텐츠

- [베이즈 정리](/ko/docs/math/probability/bayes)
- [샘플링](/ko/docs/math/probability/sampling)
- [Cross-Entropy](/ko/docs/math/training/loss/cross-entropy)
- [VAE](/ko/docs/architecture/generative/vae)
