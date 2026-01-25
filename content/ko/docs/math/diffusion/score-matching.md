---
title: "Score Matching"
weight: 2
math: true
---

# Score Matching

## 개요

Score Matching은 확률 분포의 score 함수(로그 확률의 그래디언트)를 학습하는 방법입니다.

## Score 함수

$$
\nabla_x \log p(x)
$$

- 데이터가 높은 확률 영역으로 향하는 방향
- 밀도 함수 자체가 아닌 그래디언트만 학습

## 왜 Score인가?

정규화 상수 없이 학습 가능:

$$
p(x) = \frac{e^{f(x)}}{Z} \implies \nabla_x \log p(x) = \nabla_x f(x)
$$

Z(정규화 상수)가 사라짐!

## Score Matching 목표

원래 목표:
$$
\mathbb{E}_{p_{data}} \left[ \| s_\theta(x) - \nabla_x \log p_{data}(x) \|^2 \right]
$$

문제: $\nabla_x \log p_{data}(x)$를 모름

### Denoising Score Matching

노이즈를 추가한 데이터로 우회:

$$
\mathbb{E}_{p_{data}(x), q(\tilde{x}|x)} \left[ \| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x}|x) \|^2 \right]
$$

가우시안 노이즈 $q(\tilde{x}|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$의 경우:

$$
\nabla_{\tilde{x}} \log q(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

## Score와 노이즈의 관계

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

- Score 예측 ≈ 노이즈 예측
- DDPM의 노이즈 예측은 score matching의 변형

## 구현

```python
import torch
import torch.nn as nn

class ScoreNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),  # +1 for noise level
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x, sigma):
        # 노이즈 레벨을 조건으로
        sigma_embed = sigma.view(-1, 1).expand(-1, 1)
        input = torch.cat([x, sigma_embed], dim=-1)
        return self.net(input)


def denoising_score_matching_loss(model, x, sigmas):
    """
    x: 원본 데이터
    sigmas: 노이즈 레벨
    """
    # 노이즈 추가
    noise = torch.randn_like(x)
    x_noisy = x + sigmas.view(-1, 1) * noise

    # Score 예측
    score_pred = model(x_noisy, sigmas)

    # 타겟: 실제 score
    target = -noise / sigmas.view(-1, 1)

    # Loss (가중치 적용)
    loss = ((score_pred - target) ** 2).sum(dim=-1)
    loss = (loss * sigmas ** 2).mean()  # 노이즈 레벨로 가중

    return loss
```

## Noise Conditional Score Network (NCSN)

여러 노이즈 레벨에서 학습:

```python
# 기하급수적 노이즈 스케줄
sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), L))

def train_step(model, x):
    # 랜덤 노이즈 레벨 선택
    idx = torch.randint(0, L, (len(x),))
    sigma = sigmas[idx]

    loss = denoising_score_matching_loss(model, x, sigma)
    return loss
```

## Langevin Dynamics 샘플링

Score를 사용한 샘플링:

$$
x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} z_t
$$

```python
@torch.no_grad()
def langevin_sampling(score_model, shape, sigma, n_steps=100, step_size=0.01):
    x = torch.randn(shape) * sigma

    for _ in range(n_steps):
        noise = torch.randn_like(x)
        score = score_model(x, sigma)

        x = x + (step_size / 2) * score + np.sqrt(step_size) * noise

    return x
```

## Annealed Langevin Dynamics

높은 노이즈에서 낮은 노이즈로:

```python
@torch.no_grad()
def annealed_langevin(score_model, shape, sigmas, n_steps_each=100):
    x = torch.randn(shape) * sigmas[0]

    for sigma in sigmas:
        step_size = 0.01 * (sigma / sigmas[-1]) ** 2

        for _ in range(n_steps_each):
            noise = torch.randn_like(x)
            score = score_model(x, sigma)
            x = x + (step_size / 2) * score + np.sqrt(step_size) * noise

    return x
```

## Score vs DDPM

| | Score Matching | DDPM |
|---|---------------|------|
| 예측 대상 | Score ∇log p | 노이즈 ε |
| 관계 | $s = -ε/\sqrt{1-\bar{α}}$ | 동일 |
| 샘플링 | Langevin | Ancestral |

## 관련 콘텐츠

- [DDPM](/ko/docs/math/diffusion/ddpm)
- [Sampling](/ko/docs/math/diffusion/sampling)
- [확률 분포](/ko/docs/math/probability/distribution)
