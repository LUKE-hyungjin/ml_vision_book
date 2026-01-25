---
title: "Sampling"
weight: 3
math: true
---

# Diffusion Sampling

## 개요

학습된 Diffusion 모델에서 이미지를 생성하는 다양한 샘플링 방법들입니다.

## DDPM Sampling

기본 ancestral sampling (1000 스텝):

```python
@torch.no_grad()
def ddpm_sample(model, shape, T=1000):
    x = torch.randn(shape)

    for t in reversed(range(T)):
        # 예측된 노이즈
        eps_pred = model(x, t)

        # 평균 계산
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]

        mean = (1 / np.sqrt(alpha)) * (
            x - (beta / np.sqrt(1 - alpha_bar)) * eps_pred
        )

        # 분산
        if t > 0:
            sigma = np.sqrt(beta)
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean

    return x
```

## DDIM (Denoising Diffusion Implicit Models)

결정론적 샘플링, 더 적은 스텝 가능:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right)}_{x_0 예측} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta + \sigma_t \epsilon
$$

```python
@torch.no_grad()
def ddim_sample(model, shape, steps=50, eta=0.0):
    """
    eta=0: 완전 결정론적
    eta=1: DDPM과 동일
    """
    # 시간 스텝 서브샘플링
    times = torch.linspace(0, T-1, steps).long()

    x = torch.randn(shape)

    for i in reversed(range(len(times))):
        t = times[i]
        t_prev = times[i-1] if i > 0 else 0

        # 현재/이전 alpha
        alpha_bar = alphas_cumprod[t]
        alpha_bar_prev = alphas_cumprod[t_prev]

        # 노이즈 예측
        eps_pred = model(x, t)

        # x0 예측
        x0_pred = (x - np.sqrt(1 - alpha_bar) * eps_pred) / np.sqrt(alpha_bar)
        x0_pred = x0_pred.clamp(-1, 1)

        # 분산
        sigma = eta * np.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
        )

        # 방향
        dir_xt = np.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred

        # 업데이트
        noise = torch.randn_like(x) if i > 0 else 0
        x = np.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

    return x
```

## DPM-Solver

ODE 관점의 빠른 solver:

```python
@torch.no_grad()
def dpm_solver_sample(model, shape, steps=20, order=2):
    """DPM-Solver++"""
    # 로그 SNR 기반 시간
    lambda_t = lambda t: np.log(alphas_cumprod[t]) - np.log(1 - alphas_cumprod[t])

    times = get_time_steps(steps)
    x = torch.randn(shape)

    eps_cache = []

    for i, t in enumerate(times[:-1]):
        t_next = times[i + 1]

        eps = model(x, t)
        eps_cache.append(eps)

        if order == 1 or i == 0:
            # 1차 업데이트
            x = dpm_solver_first_order_update(x, eps, t, t_next)
        else:
            # 2차 업데이트 (이전 예측 사용)
            x = dpm_solver_second_order_update(
                x, eps_cache[-2], eps_cache[-1], times[i-1], t, t_next
            )

    return x
```

## Classifier-Free Guidance

조건부 생성 강화:

$$
\tilde{\epsilon}_\theta = (1 + w) \epsilon_\theta(x_t, c) - w \cdot \epsilon_\theta(x_t, \varnothing)
$$

```python
@torch.no_grad()
def cfg_sample(model, shape, condition, guidance_scale=7.5):
    x = torch.randn(shape)

    for t in reversed(range(T)):
        # 조건부 예측
        eps_cond = model(x, t, condition)

        # 무조건 예측
        eps_uncond = model(x, t, null_condition)

        # Guidance 적용
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM 스텝
        x = ddim_step(x, eps, t)

    return x
```

## Negative Prompt

원하지 않는 것 피하기:

```python
def cfg_with_negative(model, x, t, pos_cond, neg_cond, guidance_scale):
    eps_pos = model(x, t, pos_cond)
    eps_neg = model(x, t, neg_cond)

    return eps_neg + guidance_scale * (eps_pos - eps_neg)
```

## 속도 비교

| 방법 | 스텝 수 | 품질 |
|------|---------|------|
| DDPM | 1000 | 기준 |
| DDIM | 50-100 | 유사 |
| DPM-Solver | 20-25 | 유사 |
| LCM | 4-8 | 약간 하락 |

## 샘플링 팁

```python
# 1. 시드 고정 (재현성)
torch.manual_seed(42)

# 2. Guidance 조절
# 높은 값: 충실도↑, 다양성↓
# 낮은 값: 다양성↑, 충실도↓
guidance_scale = 7.5  # Stable Diffusion 기본값

# 3. 스텝 수
# 더 많은 스텝 = 더 좋은 품질, 더 느림
steps = 50  # 적당한 균형
```

## 관련 콘텐츠

- [DDPM](/ko/docs/math/diffusion/ddpm)
- [Score Matching](/ko/docs/math/diffusion/score-matching)
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion)
