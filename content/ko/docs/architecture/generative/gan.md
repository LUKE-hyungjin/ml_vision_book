---
title: "GAN"
weight: 2
math: true
---

# GAN (Generative Adversarial Network)

## 개요

- **논문**: Generative Adversarial Networks (2014)
- **저자**: Ian Goodfellow et al.
- **핵심 기여**: 적대적 학습으로 선명한 이미지 생성

## 핵심 아이디어

> "Generator와 Discriminator가 경쟁하며 학습"

위조지폐범(Generator)과 경찰(Discriminator)의 게임:
- Generator: 진짜처럼 보이는 가짜 생성
- Discriminator: 진짜와 가짜 구분

---

## 구조

### 전체 아키텍처

```
Random Noise z
      ↓
┌─────────────────┐
│    Generator    │
│  z → fake image │
└─────────────────┘
      ↓
┌─────────────────┐      ┌─────────────────┐
│  Fake Image     │      │  Real Image     │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └──────────┬─────────────┘
                    ↓
         ┌─────────────────────┐
         │    Discriminator    │
         │  image → real/fake  │
         └─────────────────────┘
                    ↓
              0 (fake) or 1 (real)
```

---

## 목적 함수

### Minimax Game

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### 각각의 목표

**Discriminator**: $V$를 최대화
- $D(x) \to 1$ (진짜를 진짜로)
- $D(G(z)) \to 0$ (가짜를 가짜로)

**Generator**: $V$를 최소화
- $D(G(z)) \to 1$ (가짜를 진짜로 속이기)

### 실제 학습

Generator의 gradient 문제 해결을 위해:

$$\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

---

## 학습 알고리즘

```python
for epoch in range(epochs):
    for real_images in dataloader:
        # 1. Discriminator 학습
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)

        d_real = D(real_images)
        d_fake = D(fake_images.detach())

        d_loss = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 2. Generator 학습
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        d_fake = D(fake_images)

        g_loss = -torch.mean(torch.log(d_fake))
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

---

## 구현 예시

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, features=64):
        super().__init__()
        self.gen = nn.Sequential(
            # z: latent_dim → 4×4×(features*16)
            nn.ConvTranspose2d(latent_dim, features*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features*16),
            nn.ReLU(True),
            # 4×4 → 8×8
            nn.ConvTranspose2d(features*16, features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.ReLU(True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.ReLU(True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.ReLU(True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(features*2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z.view(-1, z.size(1), 1, 1))


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features=64):
        super().__init__()
        self.disc = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(img_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 4×4
            nn.Conv2d(features*4, features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → 1×1
            nn.Conv2d(features*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.disc(img).view(-1)
```

---

## GAN의 문제점

### 1. Mode Collapse

Generator가 다양성 없이 몇 가지 샘플만 생성

### 2. Training Instability

D와 G의 균형 맞추기 어려움

### 3. Vanishing Gradient

D가 너무 잘 학습되면 G의 gradient가 사라짐

---

## GAN 변형

| 모델 | 핵심 개선 |
|------|----------|
| **DCGAN** | Conv 구조, 안정적 학습 |
| **WGAN** | Wasserstein distance, 안정성 |
| **StyleGAN** | Style-based, 고품질 얼굴 |
| **BigGAN** | Large-scale, ImageNet 생성 |
| **Pix2Pix** | Image-to-image translation |
| **CycleGAN** | Unpaired translation |

---

## StyleGAN

고품질 얼굴 생성의 대표 모델:

```
Mapping Network: z → w (learned latent space)
        ↓
Synthesis Network: w → image (with style injection)
```

**핵심 기술:**
- Style injection: 각 해상도에서 스타일 적용
- Progressive growing: 저해상도 → 고해상도
- Noise injection: 세부 디테일

---

## 현재 상태

GAN은 2014-2021년 이미지 생성의 주류였지만, 현재는 Diffusion 모델에 대부분 대체되었습니다.

**GAN의 장점이 남아있는 분야:**
- 실시간 생성 (빠른 샘플링)
- Image-to-image translation
- 비디오 생성 (일부)

---

## 관련 콘텐츠

- [VAE](/ko/docs/architecture/generative/vae) - 또 다른 생성 모델
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - GAN을 대체한 모델
- [Generation 태스크](/ko/docs/task/generation) - FID 등 평가 지표
