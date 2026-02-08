---
title: "Flow Matching"
weight: 4
math: true
---

# Flow Matching

{{% hint info %}}
**선수지식**: [DDPM](/ko/docs/components/generative/ddpm) | [Score Matching](/ko/docs/components/generative/score-matching) | [미분방정식 기초](/ko/docs/math/calculus)
{{% /hint %}}

## 한 줄 요약

> **"노이즈에서 데이터로 가는 가장 직선적인 경로를 학습한다"**

---

## 왜 Flow Matching인가?

### DDPM의 문제점

DDPM은 1000스텝이 필요합니다. 왜 이렇게 많이 필요할까요?

```
DDPM의 경로: 구불구불 ~~~~~~ (확률적, 비효율적)
Flow Matching: 직선 ────── (결정론적, 효율적)
```

> **비유**: 서울에서 부산 가는데 DDPM은 전국 방방곡곡 들르고, Flow Matching은 KTX처럼 직선으로 갑니다!

---

## 핵심 아이디어

### 1. 흐름(Flow)이란?

시간에 따라 점들이 이동하는 경로입니다.

$$
\frac{dx}{dt} = v_\theta(x, t)
$$

**기호 설명:**
- $x$: 현재 위치 (이미지)
- $t$: 시간 (0=노이즈, 1=데이터)
- $v_\theta$: 속도장 (velocity field) - "어느 방향으로 얼마나 빨리 이동하나?"
- $\frac{dx}{dt}$: 위치의 변화율

### 2. 최적 경로: 직선!

노이즈 $x_0$에서 데이터 $x_1$로 가는 **가장 간단한 경로**는?

$$
x_t = (1-t) \cdot x_0 + t \cdot x_1
$$

**해석**: 시간 $t$에서의 위치는 노이즈와 데이터의 **선형 보간**!

```
t=0: 100% 노이즈, 0% 데이터
t=0.5: 50% 노이즈, 50% 데이터
t=1: 0% 노이즈, 100% 데이터
```

### 3. 속도장 (Velocity Field)

직선 경로의 속도는?

$$
v(x_t, t) = x_1 - x_0
$$

**해석**: 속도 = 도착점 - 출발점 (일정한 속도로 직진!)

---

## DDPM vs Flow Matching

| 특성 | DDPM | Flow Matching |
|------|------|---------------|
| 경로 | 확률적 (구불구불) | 결정론적 (직선) |
| 수식 | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | $x_t = (1-t)x_0 + tx_1$ |
| 학습 목표 | 노이즈 $\epsilon$ 예측 | 속도 $v$ 예측 |
| 샘플링 | SDE (확률 미분방정식) | ODE (상미분방정식) |
| 스텝 수 | 많이 필요 | 적어도 됨 |

### 시각적 비교

```
DDPM:
노이즈 ·~·~·~·~·~·~·~·~·~· 데이터
      (확률적 경로, 많은 스텝)

Flow Matching:
노이즈 ·─────────────────· 데이터
      (직선 경로, 적은 스텝)
```

---

## 학습 방법

### 목적 함수

$$
L = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

**기호 설명:**
- $v_\theta(x_t, t)$: 모델이 예측한 속도
- $x_1 - x_0$: 실제 속도 (정답)
- $\| \cdot \|^2$: 차이의 제곱 (MSE)

**해석**: 모델이 예측한 속도가 실제 속도와 같아지도록 학습!

### 학습 과정 (간단 버전)

```python
def flow_matching_loss(model, x_1):
    """
    x_1: 실제 데이터 (학습 이미지)
    """
    # 1. 노이즈 생성
    x_0 = torch.randn_like(x_1)

    # 2. 랜덤 시간 선택
    t = torch.rand(x_1.shape[0])

    # 3. 중간 지점 계산 (선형 보간)
    x_t = (1 - t) * x_0 + t * x_1

    # 4. 실제 속도 (정답)
    v_target = x_1 - x_0

    # 5. 모델 예측
    v_pred = model(x_t, t)

    # 6. 손실 계산
    loss = ((v_pred - v_target) ** 2).mean()

    return loss
```

---

## 샘플링 (생성)

### ODE 풀기

$$
x_1 = x_0 + \int_0^1 v_\theta(x_t, t) dt
$$

**해석**: 속도를 시간에 따라 적분하면 최종 위치(이미지)가 나옴!

### Euler 방법 (간단한 근사)

```python
def generate(model, num_steps=20):
    # 노이즈에서 시작
    x = torch.randn(1, 3, 512, 512)

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = i / num_steps

        # 속도 예측
        v = model(x, t)

        # 한 스텝 이동
        x = x + v * dt

    return x  # 생성된 이미지
```

**20스텝**으로도 고품질 이미지 생성 가능! (DDPM은 1000스텝)

---

## Conditional Flow Matching

텍스트 조건을 추가하려면?

$$
v_\theta(x_t, t, c)
$$

- $c$: 조건 (텍스트 임베딩)

Cross-attention으로 조건을 주입합니다 (Stable Diffusion과 동일).

---

## 장점 요약

| 장점 | 설명 |
|------|------|
| **빠른 샘플링** | 20스텝으로 고품질 |
| **간단한 수식** | 선형 보간, 직관적 |
| **안정적 학습** | ODE 기반, 분산 낮음 |
| **유연한 확장** | 다양한 조건 추가 쉬움 |

---

## 사용하는 모델

- **Stable Diffusion 3**: Flow Matching 기반
- **Flux**: Flow Matching + DiT
- **SORA**: 비디오 생성에 활용

---

## 요약

| 질문 | 답변 |
|------|------|
| Flow Matching이 뭔가요? | 노이즈→데이터 직선 경로 학습 |
| DDPM과 뭐가 다른가요? | 직선(빠름) vs 구불구불(느림) |
| 왜 더 빠른가요? | 최단 경로라서 적은 스텝으로 충분 |
| 어디에 쓰이나요? | SD3, Flux, SORA 등 최신 모델 |

---

## 관련 콘텐츠

- [DDPM](/ko/docs/components/generative/ddpm) - 기존 Diffusion 방식
- [Score Matching](/ko/docs/components/generative/score-matching) - Score 기반 관점
- [Sampling](/ko/docs/components/generative/sampling) - 다양한 샘플링 방법
- [Flux](/ko/docs/architecture/generative/flux) - Flow Matching 구현체
- [DiT](/ko/docs/architecture/generative/dit) - Transformer 기반 Diffusion
