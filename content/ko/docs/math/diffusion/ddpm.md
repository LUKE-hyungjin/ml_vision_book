---
title: "DDPM"
weight: 1
math: true
---

# DDPM (Denoising Diffusion Probabilistic Models)

{{% hint info %}}
**선수지식**: [정규분포 (가우시안)](/ko/docs/math/probability) | [MSE Loss](/ko/docs/math/training/loss)
{{% /hint %}}

## 한 줄 요약

> **"노이즈를 예측할 수 있으면, 노이즈를 제거할 수 있다!"**

---

## 왜 DDPM인가?

> **비유**: 사진을 복사기로 1000번 복사하면 점점 흐려집니다. DDPM은 "흐려진 사진에서 원본을 복원하는 방법"을 학습합니다. 복원할 수 있으면, **처음부터 흐린 노이즈에서 새로운 사진을 만들 수 있습니다!**

---

## 핵심 아이디어: 노이즈 예측

{{< figure src="/images/diffusion/ko/ddpm-training.svg" caption="DDPM 학습: 노이즈 예측하기" >}}

| 단계 | 설명 |
|------|------|
| 1. 원본 $x_0$ | 깨끗한 이미지 |
| 2. 노이즈 $\epsilon$ | 랜덤하게 생성 (정답!) |
| 3. 섞인 이미지 $x_t$ | $x_0 + \epsilon$를 적절한 비율로 섞음 |
| 4. U-Net 예측 | $x_t$를 보고 "어떤 노이즈가 섞였을까?" 예측 |
| 5. Loss | 실제 $\epsilon$와 예측 $\hat{\epsilon}$의 차이 최소화 |

---

## Forward Process (노이즈 추가)

### 직관적 이해

> **비유**: 깨끗한 물에 잉크를 조금씩 떨어뜨리면 점점 탁해집니다.

- $t=0$: 깨끗한 이미지
- $t=500$: 반쯤 노이즈
- $t=1000$: 순수 노이즈 (원본 정보 없음)

### 수식

$$
x_t = \underbrace{\sqrt{\bar{\alpha}_t}}_{\text{원본 비율}} \cdot x_0 + \underbrace{\sqrt{1-\bar{\alpha}_t}}_{\text{노이즈 비율}} \cdot \epsilon
$$

**각 기호의 의미:**
- $x_t$: 시간 $t$에서의 (노이즈가 섞인) 이미지
- $x_0$: 원본 깨끗한 이미지
- $\epsilon$: 순수 랜덤 노이즈 (정규분포에서 샘플링)
- $\bar{\alpha}_t$: 원본이 얼마나 남아있는지 (0~1 사이 값)
  - $t=0$: $\bar{\alpha}_0 \approx 1$ (원본 100%)
  - $t=1000$: $\bar{\alpha}_{1000} \approx 0$ (노이즈 100%)

### 이해하기 쉬운 표

| 시간 t | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ | 의미 |
|--------|------------------------|---------------------------|------|
| 0 | ≈ 1.0 | ≈ 0.0 | 원본 100% |
| 250 | ≈ 0.9 | ≈ 0.4 | 원본 많음 |
| 500 | ≈ 0.7 | ≈ 0.7 | 반반 |
| 750 | ≈ 0.4 | ≈ 0.9 | 노이즈 많음 |
| 1000 | ≈ 0.0 | ≈ 1.0 | 순수 노이즈 |

### 코드

```python
def forward_process(x0, t, noise=None):
    """
    x0: 원본 이미지
    t: 타임스텝 (0~1000)
    noise: 추가할 노이즈 (없으면 랜덤 생성)

    반환: (노이즈가 추가된 이미지, 추가된 노이즈)
    """
    if noise is None:
        noise = torch.randn_like(x0)  # 랜덤 노이즈 생성

    # 원본 비율과 노이즈 비율
    sqrt_alpha_bar = sqrt_alphas_cumprod[t]      # 원본 비율
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t]  # 노이즈 비율

    # 섞기!
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    return xt, noise
```

---

## 학습 목표: 노이즈 맞추기

### 직관적 이해

> **비유**: 더러운 그림을 보고 "어떤 얼룩이 묻었을까?"를 맞추는 게임

- 입력: 노이즈가 섞인 이미지 $x_t$, 시간 $t$
- 출력: 예측된 노이즈 $\hat{\epsilon}$
- 정답: 실제로 추가한 노이즈 $\epsilon$

### 수식

$$
L = \mathbb{E}\left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

**해석:**
- $\epsilon$: 실제 추가한 노이즈 (정답)
- $\epsilon_\theta(x_t, t)$: 모델이 예측한 노이즈
- $\| \cdot \|^2$: 두 값의 차이의 제곱 (MSE)
- $\mathbb{E}$: 평균 (여러 샘플에 대해)

**쉽게 말하면**: 예측과 정답의 차이를 최소화!

### 코드

```python
def train_step(model, x0, optimizer):
    """한 번의 학습 스텝"""

    # 1. 랜덤 타임스텝 선택 (0~999 중 하나)
    t = torch.randint(0, 1000, (batch_size,))

    # 2. 랜덤 노이즈 생성 (이게 정답!)
    noise = torch.randn_like(x0)

    # 3. 노이즈 추가 (Forward process)
    xt = forward_process(x0, t, noise)

    # 4. 모델이 노이즈 예측
    noise_pred = model(xt, t)

    # 5. 정답과 비교 (MSE Loss)
    loss = F.mse_loss(noise_pred, noise)

    # 6. 학습!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
```

---

## Reverse Process (노이즈 제거)

### 직관적 이해

> **비유**: 더러운 그림에서 "얼룩"을 알아냈으니, 이제 지우면 됩니다!

노이즈를 예측했으면, 그걸 빼면 원본에 가까워집니다.

### 수식

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \cdot \epsilon_\theta \right) + \sigma_t \cdot z
$$

**각 기호의 의미:**
- $x_t$: 현재 (노이즈 있는) 이미지
- $\epsilon_\theta$: 예측된 노이즈
- $\alpha_t, \beta_t$: 스케줄 파라미터 (미리 정해진 값)
- $z$: 랜덤 노이즈 (약간의 다양성을 위해)
- $x_{t-1}$: 한 스텝 더 깨끗해진 이미지

**직관적 해석:**

$$
x_{t-1} = \underbrace{\text{스케일 조정}}_{\frac{1}{\sqrt{\alpha_t}}} \times \left( \underbrace{\text{현재 이미지}}_{x_t} - \underbrace{\text{노이즈 빼기}}_{\text{예측된 노이즈}} \right) + \underbrace{\text{약간의 랜덤성}}_{\sigma_t z}
$$

### 코드

```python
@torch.no_grad()
def sample(model, shape):
    """노이즈에서 이미지 생성"""

    # 순수 노이즈에서 시작
    x = torch.randn(shape)

    # 1000번 반복해서 노이즈 제거
    for t in reversed(range(1000)):  # 999, 998, ..., 0

        # 노이즈 예측
        noise_pred = model(x, t)

        # 파라미터들
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]

        # 랜덤성 추가 (마지막 스텝 제외)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0  # 마지막엔 랜덤성 없이

        # 한 스텝 denoising
        x = (1 / sqrt(alpha)) * (x - (beta / sqrt(1 - alpha_bar)) * noise_pred)
        x = x + sqrt(beta) * noise

    return x  # 깨끗한 이미지!
```

---

## 노이즈 스케줄: β 값 정하기

### 왜 필요한가?

β (베타) 값은 "각 스텝에서 노이즈를 얼마나 추가할지"를 결정합니다.

### Linear Schedule (기본)

처음에는 조금, 나중에는 많이 추가:

```
β: 0.0001 → 0.0002 → ... → 0.02
```

```python
def linear_schedule(T=1000):
    return torch.linspace(0.0001, 0.02, T)
```

### Cosine Schedule (개선판)

코사인 곡선을 따라 더 부드럽게:

```
β: 부드러운 곡선 ⌒
```

더 좋은 결과를 내는 경우가 많습니다.

---

## U-Net: 노이즈 예측 모델

### 왜 U-Net인가?

> **비유**: 이미지에서 "무엇을 지워야 할지" 알려면, 이미지 전체를 봐야 합니다. U-Net은 전체 그림을 보면서 세부사항도 놓치지 않습니다.

### 구조

```
입력 (노이즈 이미지)
   ↓
[인코더] 점점 작게 (64→128→256)
   ↓
[병목] 가장 작은 표현
   ↓
[디코더] 다시 크게 (256→128→64)
   ↓
출력 (예측된 노이즈)
```

### 핵심: 시간 정보 주입

모델은 "지금 몇 번째 스텝인지" 알아야 합니다.
- $t=100$: 노이즈 조금 → 세밀하게 예측
- $t=900$: 노이즈 많음 → 대략적으로 예측

```python
# 시간 정보를 숫자로 인코딩
time_embedding = sinusoidal_encoding(t)  # t → 256차원 벡터

# 모든 레이어에 시간 정보 주입
h = layer(x) + time_embedding
```

---

## 요약

| 과정 | 하는 일 | 수학 |
|------|---------|------|
| **Forward** | 노이즈 추가 | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ |
| **학습** | 노이즈 예측 | $L = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$ |
| **Reverse** | 노이즈 제거 | $x_{t-1} = f(x_t, \epsilon_\theta, t)$ |

### 핵심 포인트

1. **Forward는 쉬움**: 그냥 노이즈 더하기
2. **Reverse를 학습**: 노이즈가 뭐였는지 맞추기
3. **생성**: 순수 노이즈에서 시작 → 1000번 denoising → 새 이미지!

---

## 관련 콘텐츠

- [Score Matching](/ko/docs/math/diffusion/score-matching) - Score 관점에서 보기
- [Sampling](/ko/docs/math/diffusion/sampling) - 더 빠른 샘플링 방법
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - 실제 구현
