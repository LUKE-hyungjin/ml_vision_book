---
title: "VAE"
weight: 1
math: true
---

# VAE (Variational Autoencoder)

## 개요

- **논문**: Auto-Encoding Variational Bayes (2013)
- **저자**: Diederik P. Kingma, Max Welling
- **핵심 기여**: 확률적 잠재 공간에서 생성 가능한 오토인코더

## 핵심 아이디어

> "데이터를 압축하는 동시에 잠재 공간을 정규화하여 생성 가능하게"

일반 Autoencoder는 데이터 압축만 하지만, VAE는 잠재 공간이 연속적이고 의미있는 구조를 갖도록 학습합니다.

---

## Autoencoder vs VAE

### 일반 Autoencoder

```
x → Encoder → z (deterministic) → Decoder → x̂
```

- z가 불연속적
- 학습 데이터 외의 z에서 생성 불가

### VAE

```
x → Encoder → (μ, σ) → z ~ N(μ, σ²) → Decoder → x̂
```

- z가 확률 분포
- 잠재 공간이 연속적 → 생성 가능

---

## 구조

### 전체 아키텍처

```
Input x
    ↓
┌─────────────────────────┐
│       Encoder           │
│  x → hidden → (μ, σ)   │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Reparameterization    │
│   z = μ + σ * ε         │
│   where ε ~ N(0, 1)     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│       Decoder           │
│   z → hidden → x̂       │
└─────────────────────────┘
    ↓
Output x̂
```

### Reparameterization Trick

Gradient가 샘플링을 통과하도록:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

이렇게 하면 $\mu$, $\sigma$에 대해 역전파 가능합니다.

---

## 손실 함수

### ELBO (Evidence Lower Bound)

$$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{KL}(q(z|x) \| p(z))$$

### 두 가지 항

**1. Reconstruction Loss**:
$$\mathcal{L}_{recon} = \|x - \hat{x}\|^2 \quad \text{or} \quad \text{BCE}(x, \hat{x})$$

**2. KL Divergence**:
$$\mathcal{L}_{KL} = -\frac{1}{2}\sum_{j=1}^{J}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

잠재 분포 $q(z|x)$를 표준 정규분포 $p(z) = \mathcal{N}(0, I)$에 가깝게.

### 전체 손실

$$\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}$$

- $\beta = 1$: 표준 VAE
- $\beta > 1$: β-VAE (더 disentangled)

---

## 구현 예시

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
```

---

## 생성

학습 후 새로운 이미지 생성:

```python
# 표준 정규분포에서 샘플링
z = torch.randn(batch_size, latent_dim)

# Decoder로 이미지 생성
generated = model.decode(z)
```

### 잠재 공간 탐색

```python
# 두 이미지 사이 보간
z1, z2 = model.encode(img1), model.encode(img2)
for alpha in np.linspace(0, 1, 10):
    z_interp = alpha * z1 + (1 - alpha) * z2
    img_interp = model.decode(z_interp)
```

---

## VAE 변형

| 변형 | 핵심 아이디어 |
|------|--------------|
| **β-VAE** | KL 가중치 증가 → disentanglement |
| **VQ-VAE** | 이산 잠재 공간 |
| **CVAE** | 조건부 생성 |
| **NVAE** | 계층적 구조, 고해상도 |

---

## 한계점

- 생성된 이미지가 흐릿함 (pixel-wise loss 한계)
- 고해상도 생성 어려움
- GAN 대비 선명도 부족

이러한 한계 때문에 현재는 [Diffusion](/ko/docs/architecture/generative/stable-diffusion) 모델이 주류가 되었습니다.

---

## VAE의 역할

현재 VAE 자체보다 **VAE Encoder**가 중요하게 사용됩니다:

- **Stable Diffusion**: VAE로 latent space 압축
- **VQ-VAE → DALL-E**: 이산 토큰 생성

---

## 관련 콘텐츠

- [확률분포](/ko/docs/math/probability) - VAE의 수학적 기초
- [GAN](/ko/docs/architecture/generative/gan) - 적대적 생성 모델
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - VAE를 encoder로 사용
- [Generation 태스크](/ko/docs/task/generation) - 평가 지표
