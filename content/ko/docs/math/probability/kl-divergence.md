---
title: "KL 발산"
weight: 11
math: true
---

# KL 발산 (Kullback-Leibler Divergence)

## 개요

> 💡 **KL Divergence**: 두 확률분포가 **얼마나 다른지** 측정

VAE, 지식 증류, 정책 최적화 등 딥러닝의 핵심 수학입니다.

### 시각적 이해

![KL Divergence](/images/probability/kl-divergence.svg)

---

## 정의

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_P \left[ \log \frac{P(X)}{Q(X)} \right]
$$

연속인 경우:
$$
D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

### 다른 표현

$$
D_{KL}(P \| Q) = H(P, Q) - H(P)
$$

Cross-Entropy에서 엔트로피를 빼면 KL Divergence.

---

## 직관적 이해

### 정보 이론적 해석

- P를 "진짜" 분포라고 할 때
- Q로 P를 인코딩하면 얼마나 추가 비트가 필요한가

```
P = 실제 데이터 분포
Q = 모델이 학습한 분포

D_KL(P || Q) = P로 샘플링할 때 Q의 놀라움 - P의 놀라움
             = "Q가 P를 잘 설명하지 못하는 정도"
```

### 예시

```
P: [0.5, 0.5]      (공정한 동전)
Q: [0.9, 0.1]      (편향된 동전)

D_KL(P || Q) = 0.5 × log(0.5/0.9) + 0.5 × log(0.5/0.1)
             = 0.5 × (-0.85) + 0.5 × (2.32)
             ≈ 0.74 bits
```

---

## KL 발산의 성질

### 1. 비음수성 (Gibbs' Inequality)

$$
D_{KL}(P \| Q) \geq 0
$$

등호는 $P = Q$일 때만 성립.

### 2. 비대칭성 ⚠️

$$
D_{KL}(P \| Q) \neq D_{KL}(Q \| P)
$$

**중요**: KL 발산은 거리(metric)가 아닙니다!

### Forward vs Reverse KL

| | Forward KL: $D_{KL}(P \| Q)$ | Reverse KL: $D_{KL}(Q \| P)$ |
|---|---|---|
| 최소화 대상 | Q (모델) | Q (모델) |
| P가 높은데 Q가 낮으면 | 큰 페널티 (0으로 나눔) | 작은 페널티 |
| 특성 | Mode-covering | Mode-seeking |
| 결과 | 모든 모드 커버, 흐릿 | 하나의 모드 집중, 선명 |

```
P (실제: 두 봉우리)       Forward KL 결과        Reverse KL 결과

   ╭─╮   ╭─╮               ╭─────╮               ╭─╮
   │ │   │ │               │     │               │ │
───┴─┴───┴─┴───         ───┴─────┴───         ───┴─┴───────
    ↑       ↑               흐릿하지만             하나만 선택
  mode1  mode2            둘 다 커버
```

---

## 딥러닝에서의 활용

### 1. VAE (Variational Autoencoder)

$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

- $q(z|x)$: 인코더 (근사 사후 분포)
- $p(z)$: 사전 분포 (표준 정규)
- KL 항: 잠재 공간을 정규화

### 2. 지식 증류 (Knowledge Distillation)

$$
\mathcal{L}_{KD} = T^2 \cdot D_{KL}(P_{teacher} \| P_{student})
$$

Teacher의 soft label을 Student가 따라하게.

### 3. PPO (강화학습 정책 최적화)

$$
\text{clip}\left( \frac{\pi_{new}}{\pi_{old}}, 1-\epsilon, 1+\epsilon \right)
$$

새 정책이 구 정책에서 너무 벗어나지 않도록.

### 4. 정규화

레이블 스무딩, Focal Loss 등도 분포 간 거리 개념.

---

## 가우시안 KL 발산

두 정규 분포 사이의 KL Divergence (해석적 해):

$$
D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

### VAE에서 자주 쓰는 형태

$q(z|x) = \mathcal{N}(\mu, \sigma^2)$, $p(z) = \mathcal{N}(0, 1)$일 때:

$$
D_{KL}(q \| p) = -\frac{1}{2} \sum_{j=1}^{J} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
$$

---

## 구현

```python
import numpy as np
import torch
import torch.nn.functional as F

def kl_divergence_discrete(p, q, eps=1e-10):
    """이산 분포의 KL Divergence"""
    p = np.array(p) + eps
    q = np.array(q) + eps
    return np.sum(p * np.log(p / q))

def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """두 1D 가우시안 사이의 KL Divergence"""
    return (np.log(sigma2/sigma1) +
            (sigma1**2 + (mu1-mu2)**2) / (2*sigma2**2) - 0.5)

def kl_divergence_vae(mu, logvar):
    """VAE에서 사용하는 KL Divergence (vs 표준 정규)"""
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# 예시 1: 이산 분포
p = [0.5, 0.5]
q = [0.9, 0.1]
print(f"D_KL(P || Q) = {kl_divergence_discrete(p, q):.4f}")
print(f"D_KL(Q || P) = {kl_divergence_discrete(q, p):.4f}")  # 다른 값!

# 예시 2: 가우시안
kl_gauss = kl_divergence_gaussian(mu1=1, sigma1=1, mu2=0, sigma2=1)
print(f"D_KL(N(1,1) || N(0,1)) = {kl_gauss:.4f}")

# 예시 3: VAE KL Loss
mu = torch.randn(32, 64)      # 배치 32, 잠재 차원 64
logvar = torch.randn(32, 64)  # log(sigma^2)
kl_loss = kl_divergence_vae(mu, logvar)
print(f"VAE KL Loss: {kl_loss.item():.4f}")

# 예시 4: PyTorch의 KL Divergence
p_logits = torch.tensor([[1.0, 2.0, 3.0]])
q_logits = torch.tensor([[3.0, 2.0, 1.0]])

p_probs = F.softmax(p_logits, dim=-1)
q_probs = F.softmax(q_logits, dim=-1)

# F.kl_div는 log_probs를 받음
kl_pt = F.kl_div(q_probs.log(), p_probs, reduction='sum')
print(f"PyTorch KL: {kl_pt.item():.4f}")
```

---

## 다른 발산/거리와 비교

| 측도 | 수식 | 대칭 | 특징 |
|------|------|------|------|
| KL Divergence | $\sum p \log(p/q)$ | ❌ | 정보 이론적 의미 |
| Jensen-Shannon | $\frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$ | ✅ | 대칭화된 KL |
| Wasserstein | $\inf_\gamma \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$ | ✅ | 기하학적 거리 |
| Total Variation | $\frac{1}{2}\sum\|p-q\|$ | ✅ | L1 거리 |

---

## 핵심 정리

| 개념 | 핵심 |
|------|------|
| 정의 | $D_{KL}(P\|Q) = \sum p \log(p/q)$ |
| 의미 | P를 Q로 표현할 때 필요한 추가 정보 |
| 비대칭 | $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ |
| 비음수 | $D_{KL}(P\|Q) \geq 0$ |
| Cross-Entropy 관계 | $H(P,Q) = H(P) + D_{KL}(P\|Q)$ |

---

## 관련 콘텐츠

- [엔트로피](/docs/math/probability/entropy) - KL Divergence의 기반
- [Cross-Entropy Loss](/docs/math/training/loss/cross-entropy) - 손실 함수와의 관계
- [확률분포](/docs/math/probability/distribution) - 분포 기초
