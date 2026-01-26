---
title: "DDPM 수학"
weight: 1
math: true
---

# DDPM 수학

{{% hint info %}}
**선수지식**: [정규분포](/ko/docs/math/probability) | [KL Divergence](/ko/docs/math/probability)
{{% /hint %}}

## 한 줄 요약

> **"노이즈 추가는 수학으로 정의, 노이즈 제거는 신경망으로 학습"**

---

## 핵심 아이디어

DDPM (Denoising Diffusion Probabilistic Models)의 수학적 핵심:

1. **Forward Process**: 데이터에 노이즈를 조금씩 추가 → 순수 노이즈로
2. **Reverse Process**: 노이즈에서 조금씩 제거 → 데이터로

```
Forward (정의됨):  x₀ → x₁ → x₂ → ... → x_T (순수 노이즈)
Reverse (학습):    x_T → x_{T-1} → ... → x₀ (깨끗한 이미지)
```

---

## Forward Process (노이즈 추가)

{{< figure src="/images/math/generative/ddpm/ko/ddpm-forward-math.png" caption="Forward Process: x₀에서 x_T까지 노이즈 추가" >}}

### 한 스텝 전이

각 스텝에서 가우시안 노이즈를 추가합니다:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \mathbf{I})
$$

**기호 설명:**
- $x_{t-1}$: 이전 스텝의 이미지
- $x_t$: 현재 스텝의 (더 노이즈한) 이미지
- $\beta_t$: 노이즈 스케줄 ($\beta_1 < \beta_2 < ... < \beta_T$)
- $\mathcal{N}(\mu, \sigma^2)$: 평균 $\mu$, 분산 $\sigma^2$인 정규분포

### 직관적 이해

> **비유**: 복사기로 복사하면 약간 흐려집니다. 1000번 복사하면 원본을 알아볼 수 없습니다.

- $\sqrt{1-\beta_t}$: 이전 이미지를 약간 줄임 (0.9999 → 0.99)
- $\beta_t$: 노이즈의 분산

---

## 한 번에 t스텝으로 점프

### 핵심 트릭

매번 순차적으로 노이즈를 추가할 필요 없습니다. **한 번에 t스텝 후 결과**를 계산할 수 있습니다:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

**표기법:**
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \alpha_1 \cdot \alpha_2 \cdot ... \cdot \alpha_t$

### 샘플링 공식

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

### 유도 과정

두 가우시안의 곱은 가우시안입니다. $q(x_t \mid x_{t-1})$을 반복 적용하면:

$$
x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1
$$

$$
x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2
$$

$x_1$을 대입하고 정리하면 ($\epsilon_1, \epsilon_2$는 독립 가우시안):

$$
x_2 = \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{1-\alpha_1\alpha_2} \epsilon
$$

일반화하면 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

### 시간에 따른 변화

| 시간 t | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ | 의미 |
|--------|-------------------------|---------------------------|------|
| 0 | ≈ 1.0 | ≈ 0.0 | 원본 100% |
| T/4 | ≈ 0.9 | ≈ 0.4 | 원본 많음 |
| T/2 | ≈ 0.7 | ≈ 0.7 | 반반 |
| 3T/4 | ≈ 0.4 | ≈ 0.9 | 노이즈 많음 |
| T | ≈ 0.0 | ≈ 1.0 | 순수 노이즈 |

---

## Reverse Process (노이즈 제거)

{{< figure src="/images/math/generative/ddpm/ko/ddpm-reverse-math.png" caption="Reverse Process: x_T에서 x₀까지 노이즈 제거 (학습 대상)" >}}

### 목표

Forward의 역방향을 학습합니다:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**핵심 질문**: $\mu_\theta$를 어떻게 파라미터화할까?

### 분산 선택

DDPM 논문에서는 분산을 고정합니다:

$$
\Sigma_\theta(x_t, t) = \sigma_t^2 \mathbf{I}
$$

두 가지 선택:
- $\sigma_t^2 = \beta_t$ (단순)
- $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$ (최적)

---

## Posterior Distribution (사후 분포)

### 핵심 통찰

**만약 $x_0$를 알고 있다면**, $q(x_{t-1} \mid x_t, x_0)$를 정확히 계산할 수 있습니다:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})
$$

### Posterior Mean

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

### Posterior Variance

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

### 유도 (베이즈 정리)

베이즈 정리를 적용:

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) \cdot q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

세 분포 모두 가우시안이므로, 곱하고 지수부를 정리하면 위 결과가 나옵니다.

---

## Loss Function 유도

{{< figure src="/images/math/generative/ddpm/ko/ddpm-loss-derivation.png" caption="Loss 유도 과정: VLB → Simplified Loss" >}}

### Variational Lower Bound (VLB)

음의 로그 가능도의 상한:

$$
-\log p_\theta(x_0) \leq \mathbb{E}_q \left[ -\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] = L_{VLB}
$$

### 분해

$L_{VLB}$를 분해하면:

$$
L_{VLB} = L_0 + L_1 + ... + L_{T-1} + L_T
$$

각 항:
- $L_0 = -\log p_\theta(x_0 \mid x_1)$ (reconstruction)
- $L_{t-1} = D_{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))$ for $t > 1$
- $L_T = D_{KL}(q(x_T \mid x_0) \| p(x_T))$ (상수, 학습 안 함)

### L 계산

두 가우시안 사이의 KL divergence:

$$
L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \lVert \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \rVert^2 \right] + C
$$

---

## 노이즈 예측으로 재파라미터화

### 핵심 아이디어

$\mu_\theta$를 직접 예측하는 대신, **노이즈 $\epsilon$을 예측**합니다.

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$에서 $x_0$를 풀면:

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1-\bar{\alpha}_t} \epsilon)
$$

이를 posterior mean $\tilde{\mu}_t$에 대입하면:

$$
\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)
$$

### 평균의 파라미터화

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

여기서 $\epsilon_\theta$는 **노이즈를 예측하는 신경망**입니다.

---

## Simplified Loss

### 유도

$L_{t-1}$에 노이즈 파라미터화를 대입하면:

$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon, t} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2 \right]
$$

### Simplified Loss

앞의 계수를 무시하고 단순화:

$$
L_{simple} = \mathbb{E}_{x_0, \epsilon, t} \left[ \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2 \right]
$$

**해석**: 추가한 노이즈 $\epsilon$와 예측한 노이즈 $\epsilon_\theta$의 MSE

### 왜 단순화가 잘 작동하나?

- 원래 가중치는 $t$가 작을 때 매우 큼 (노이즈 적을 때)
- 단순화하면 모든 $t$를 균등하게 학습
- 실험적으로 더 좋은 샘플 품질

---

## 노이즈 스케줄

### Linear Schedule

$$
\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)
$$

DDPM 논문: $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$

### Cosine Schedule (Improved DDPM)

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
$$

더 부드러운 노이즈 추가 → 더 좋은 품질

---

## 샘플링 수식

학습된 $\epsilon_\theta$로 샘플링:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

여기서 $z \sim \mathcal{N}(0, \mathbf{I})$ (마지막 스텝 제외)

---

## 요약

| 구성요소 | 수식 |
|----------|------|
| **Forward (한 스텝)** | $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$ |
| **Forward (직접)** | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ |
| **Posterior** | $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t \mathbf{I})$ |
| **Reverse** | $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 \mathbf{I})$ |
| **Loss** | $L_{simple} = \mathbb{E}[\lVert\epsilon - \epsilon_\theta(x_t, t)\rVert^2]$ |

---

## 관련 콘텐츠

- [Score Matching](/ko/docs/math/generative/score-matching) - Score 관점에서의 해석
- [Sampling](/ko/docs/math/generative/sampling) - DDIM 등 빠른 샘플링
- [Flow Matching](/ko/docs/math/generative/flow-matching) - 더 간단한 수학적 프레임워크
- [DDPM 아키텍처](/ko/docs/architecture/generative/ddpm) - U-Net 구현체
