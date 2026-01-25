---
title: "확률 변수"
weight: 2
math: true
---

# 확률 변수 (Random Variable)

## 개요

확률 변수는 랜덤 실험의 결과를 숫자로 매핑하는 함수입니다. 딥러닝의 모든 입력과 출력은 확률 변수로 모델링됩니다.

---

## 정의

$$
X: \Omega \rightarrow \mathbb{R}
$$

표본 공간 $\Omega$의 각 원소를 실수 값으로 대응시키는 함수.

### 예시: 동전 던지기

```
표본 공간: Ω = {앞면, 뒷면}

확률 변수 X: "앞면이 나온 횟수"
X(앞면) = 1
X(뒷면) = 0
```

---

## 이산 확률 변수 (Discrete Random Variable)

셀 수 있는 값을 가지는 확률 변수.

### 확률 질량 함수 (PMF: Probability Mass Function)

$$
P(X = x) = p(x)
$$

**성질**:
- $p(x) \geq 0$
- $\sum_x p(x) = 1$

### 주요 이산 분포

| 분포 | PMF | 사용 예 |
|------|-----|---------|
| 베르누이 | $p^x(1-p)^{1-x}$ | 이진 분류 |
| 이항 | $\binom{n}{k}p^k(1-p)^{n-k}$ | n번 시행 중 성공 횟수 |
| 카테고리컬 | $\prod_i p_i^{[x=i]}$ | 다중 클래스 분류 |
| 포아송 | $\frac{\lambda^k e^{-\lambda}}{k!}$ | 드문 이벤트 발생 횟수 |

### 시각화

```
베르누이 분포 (p=0.3)

P(X)
  │
0.7├─────────────┐
  │             │
0.3├──────┐     │
  │      │     │
  └──────┴─────┴────→ X
         0     1
```

---

## 연속 확률 변수 (Continuous Random Variable)

연속적인 값을 가지는 확률 변수.

### 확률 밀도 함수 (PDF: Probability Density Function)

$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

**성질**:
- $f(x) \geq 0$
- $\int_{-\infty}^{\infty} f(x) \, dx = 1$

**주의**: $P(X = x) = 0$ (특정 점에서의 확률은 0)

### 주요 연속 분포

| 분포 | PDF | 사용 예 |
|------|-----|---------|
| 균등 | $\frac{1}{b-a}$ | 초기화 |
| 정규(가우시안) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | 노이즈, 잠재 공간 |
| 지수 | $\lambda e^{-\lambda x}$ | 대기 시간 |
| 베타 | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | 확률의 확률 |

---

## 누적 분포 함수 (CDF: Cumulative Distribution Function)

$$
F(x) = P(X \leq x)
$$

### 이산 vs 연속

```
이산 CDF (계단 함수)          연속 CDF (부드러운 곡선)

F(x)                         F(x)
  │    ┌───────              │         ╭────
1 ├────┘                   1 ├────────╯
  │  ┌─┘                     │     ╭──╯
  ├──┘                       │  ╭─╯
  │                          │╭╯
0 └─────────→ x            0 └────────────→ x
```

### PDF와 CDF의 관계

$$
f(x) = \frac{dF(x)}{dx}
$$

$$
F(x) = \int_{-\infty}^{x} f(t) \, dt
$$

---

## 딥러닝에서의 확률 변수

### 입력 데이터

$$
X \sim P_{data}(x)
$$

이미지, 텍스트 등 학습 데이터는 미지의 분포에서 샘플링된 것으로 가정.

### 모델 출력 (분류)

$$
Y | X \sim \text{Categorical}(\text{softmax}(f_\theta(X)))
$$

### 잠재 변수 (VAE)

$$
Z \sim \mathcal{N}(0, I)
$$

표준 정규 분포에서 샘플링 후 디코딩.

### 노이즈 (Diffusion)

$$
\epsilon \sim \mathcal{N}(0, I)
$$

---

## 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 이산 확률 변수: 베르누이
bernoulli = stats.bernoulli(p=0.3)
samples_discrete = bernoulli.rvs(size=1000)
print(f"베르누이 평균: {samples_discrete.mean():.3f}")  # ≈ 0.3

# 연속 확률 변수: 정규 분포
normal = stats.norm(loc=0, scale=1)  # μ=0, σ=1
samples_continuous = normal.rvs(size=1000)
print(f"정규 분포 평균: {samples_continuous.mean():.3f}")  # ≈ 0

# PDF 계산
x = np.linspace(-4, 4, 100)
pdf_values = normal.pdf(x)

# CDF 계산
cdf_values = normal.cdf(x)

# P(-1 < X < 1) 계산
prob = normal.cdf(1) - normal.cdf(-1)
print(f"P(-1 < X < 1) = {prob:.3f}")  # ≈ 0.683

# 분위수 (Quantile): P(X < q) = 0.95인 q 찾기
q_95 = normal.ppf(0.95)
print(f"95% 분위수: {q_95:.3f}")  # ≈ 1.645
```

---

## 다변량 확률 변수 (Multivariate Random Variable)

여러 확률 변수의 벡터:

$$
\mathbf{X} = (X_1, X_2, ..., X_n)^T
$$

### 다변량 정규 분포

$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

- $\boldsymbol{\mu}$: 평균 벡터
- $\boldsymbol{\Sigma}$: 공분산 행렬

```python
# 2D 다변량 정규 분포
mean = [0, 0]
cov = [[1, 0.5],   # X1 분산=1, X1-X2 공분산=0.5
       [0.5, 1]]   # X2 분산=1

samples = np.random.multivariate_normal(mean, cov, 1000)
print(f"Shape: {samples.shape}")  # (1000, 2)
```

---

## 핵심 정리

| 구분 | 이산 | 연속 |
|------|------|------|
| 확률 함수 | PMF: $P(X=x)$ | PDF: $f(x)$ |
| 확률 계산 | $\sum_x p(x)$ | $\int f(x)dx$ |
| 특정 값 확률 | $P(X=x) > 0$ 가능 | $P(X=x) = 0$ |
| 예시 | 클래스 레이블 | 픽셀 값, 잠재 벡터 |

---

## 관련 콘텐츠

- [확률의 기초](/ko/docs/math/probability/basics) - 선수 지식
- [기댓값과 분산](/ko/docs/math/probability/expectation) - 확률 변수의 특성
- [확률분포](/ko/docs/math/probability/distribution) - 주요 분포 상세
- [샘플링](/ko/docs/math/probability/sampling) - 분포에서 값 추출
