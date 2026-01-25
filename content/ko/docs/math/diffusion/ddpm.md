---
title: "DDPM"
weight: 1
math: true
---

# DDPM (Denoising Diffusion Probabilistic Models)

## 개요

DDPM은 데이터에 점진적으로 노이즈를 추가한 후, 그 역과정을 학습하여 이미지를 생성합니다.

## Forward Process (노이즈 추가)

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

- β_t: 노이즈 스케줄 (보통 0.0001 ~ 0.02)
- t = 1, ..., T (보통 T=1000)

### 한 번에 t 스텝 이동

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
$$

- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

```python
def forward_process(x0, t, noise=None):
    """x0에서 xt로 한 번에 이동"""
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alpha_bar = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t]

    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise
```

## Reverse Process (노이즈 제거)

학습 대상:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

### 평균 예측

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

핵심: 모델은 **노이즈 ε**를 예측

## 학습 목표

Simple loss (MSE):

$$
L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

```python
def train_step(model, x0, optimizer):
    # 랜덤 타임스텝
    t = torch.randint(0, T, (batch_size,))

    # 노이즈 샘플링
    noise = torch.randn_like(x0)

    # Forward process
    xt = forward_process(x0, t, noise)

    # 노이즈 예측
    noise_pred = model(xt, t)

    # Loss
    loss = F.mse_loss(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
```

## 샘플링 (생성)

```python
@torch.no_grad()
def sample(model, shape):
    # 순수 노이즈에서 시작
    x = torch.randn(shape)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t)

        # 노이즈 예측
        noise_pred = model(x, t_batch)

        # x_{t-1} 계산
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (1 / np.sqrt(alpha)) * (
            x - (beta / np.sqrt(1 - alpha_bar)) * noise_pred
        ) + np.sqrt(beta) * noise

    return x
```

## 노이즈 스케줄

```python
def linear_schedule(T, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_schedule(T, s=0.008):
    """Improved DDPM의 cosine 스케줄"""
    steps = torch.linspace(0, T, T + 1)
    f = torch.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0, 0.999)
```

## U-Net 구조

```python
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, time_dim=256):
        super().__init__()
        # 시간 임베딩
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.down1 = DownBlock(in_channels, 64, time_dim)
        self.down2 = DownBlock(64, 128, time_dim)
        self.down3 = DownBlock(128, 256, time_dim)

        # Bottleneck
        self.mid = MidBlock(256, time_dim)

        # Decoder
        self.up1 = UpBlock(256 + 256, 128, time_dim)
        self.up2 = UpBlock(128 + 128, 64, time_dim)
        self.up3 = UpBlock(64 + 64, 64, time_dim)

        self.out = nn.Conv2d(64, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        h1 = self.down1(x, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)

        # Bottleneck
        h = self.mid(h3, t_emb)

        # Decoder with skip connections
        h = self.up1(torch.cat([h, h3], dim=1), t_emb)
        h = self.up2(torch.cat([h, h2], dim=1), t_emb)
        h = self.up3(torch.cat([h, h1], dim=1), t_emb)

        return self.out(h)
```

## 관련 콘텐츠

- [Score Matching](/ko/docs/math/diffusion/score-matching)
- [Sampling](/ko/docs/math/diffusion/sampling)
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion)
