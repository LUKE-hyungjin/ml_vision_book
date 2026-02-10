---
title: "적분 기초"
weight: 2
math: true
---

# 적분 기초 (Integral Basics)

{{< figure src="/images/math/calculus/ko/integral-area-probability.jpeg" caption="적분의 핵심: 잘게 쪼개서 더하기 → 넓이 = 확률, 기댓값, Monte Carlo" >}}

{{% hint info %}}
**선수지식**: [미분 기초](/ko/docs/math/calculus/basics)
{{% /hint %}}

> **한 줄 요약**: 적분은 **"잘게 쪼개서 전부 더한다"**는 것입니다. 확률분포의 넓이가 1인 이유, 기댓값 공식, 정규화 상수 — 모두 적분입니다.

## 왜 적분을 배워야 하나요?

### 문제 상황 1: "확률밀도함수(PDF)의 넓이가 왜 1이어야 하나요?"

```python
import torch
from torch.distributions import Normal

dist = Normal(0, 1)  # 표준정규분포
# P(-∞ < X < ∞) = 1  ← 이게 왜 1인지?
# → PDF를 전체 구간에서 적분하면 1
```

→ $\int_{-\infty}^{\infty} f(x) dx = 1$ — 이것이 **적분**입니다.

### 문제 상황 2: "기댓값 공식에 왜 적분이 나오나요?"

```python
# E[X] = ∫ x·f(x) dx  ← 이 적분이 뭘 의미하지?
mean = dist.mean  # 0.0
```

→ "모든 가능한 값 × 그 확률"을 **전부 더한 것** = 적분

### 문제 상황 3: "VAE의 ELBO에 적분이 나오는데..."

```python
# ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
# E_q[...] = ∫ q(z|x) · log p(x|z) dz  ← 이걸 어떻게 계산?
```

→ 직접 적분이 어려워서 **샘플링으로 근사**합니다 (Monte Carlo)

---

## 적분이란 무엇인가?

### 핵심 아이디어: 잘게 쪼개서 전부 더하기

**미분**이 "순간 변화율"이라면, **적분**은 "변화율의 총합"입니다.

| 미분 | 적분 |
|------|------|
| 속도 → 순간 속도 | 속도 → 이동 거리 |
| 쪼개기 | 모으기 |
| $f(x) \rightarrow f'(x)$ | $f'(x) \rightarrow f(x)$ |

### 비유: 속도계와 주행 거리

- **미분**: 매 순간의 속도계 수치 (순간 변화율)
- **적분**: 속도계 수치를 시간에 걸쳐 전부 더하면 → **총 이동 거리**

```python
import torch

# 속도 함수: v(t) = 2t (시간에 비례하는 가속)
# 0~5초 동안의 총 이동 거리 = ∫₀⁵ 2t dt

# 수치 적분: 잘게 쪼개서 더하기
dt = 0.001
t = torch.arange(0, 5, dt)
v = 2 * t  # 속도

distance = (v * dt).sum()  # Σ v(t)·Δt
print(f"수치 적분: {distance:.3f}")  # ≈ 25.0

# 해석적 적분: ∫₀⁵ 2t dt = [t²]₀⁵ = 25 - 0 = 25
print(f"해석적 적분: 25.0")
```

---

## 부정적분 (Indefinite Integral)

### 정의: 미분의 역과정

$F'(x) = f(x)$이면, $F(x)$는 $f(x)$의 **부정적분(역도함수)**:

$$
\int f(x)\,dx = F(x) + C
$$

**각 기호의 의미:**
- $\int$ : 적분 기호 — "전부 더한다"
- $f(x)$ : **피적분함수** — 더할 대상
- $dx$ : "$x$에 대해 적분" — 어떤 변수로 더하는지
- $C$ : **적분 상수** — 미분하면 사라지는 상수

### 기본 적분 공식

| 함수 $f(x)$ | 적분 $\int f(x)\,dx$ | 딥러닝 연결 |
|-------------|---------------------|------------|
| $x^n$ | $\frac{x^{n+1}}{n+1} + C$ | 기본 |
| $e^x$ | $e^x + C$ | 정규분포, Softmax |
| $\frac{1}{x}$ | $\ln\|x\| + C$ | 엔트로피, KL-divergence |
| $\cos x$ | $\sin x + C$ | Positional encoding |

```python
import sympy as sp

x = sp.Symbol('x')

# 기본 적분 확인
print(sp.integrate(x**2, x))       # x³/3
print(sp.integrate(sp.exp(x), x))  # exp(x)
print(sp.integrate(1/x, x))        # log(x)
```

---

## 정적분 (Definite Integral)

### 정의: 구간에서의 넓이

$$
\int_a^b f(x)\,dx = F(b) - F(a)
$$

**각 기호의 의미:**
- $a, b$ : 적분 구간의 시작과 끝
- $F(b) - F(a)$ : 역도함수의 차이 = **넓이**

### 미적분의 기본 정리 (Fundamental Theorem of Calculus)

**미분과 적분은 서로 역연산**:

$$
\frac{d}{dx}\int_a^x f(t)\,dt = f(x)
$$

→ 적분한 것을 미분하면 원래 함수로 돌아옵니다.

```python
import torch

# 정적분: ∫₀¹ x² dx = [x³/3]₀¹ = 1/3
dx = 0.0001
x = torch.arange(0, 1, dx)
area = (x**2 * dx).sum()
print(f"수치 적분: {area:.6f}")   # ≈ 0.333333
print(f"해석적 답: {1/3:.6f}")   # 0.333333
```

---

## 수치 적분: 컴퓨터로 적분하기

해석적 풀이가 어려울 때 **수치적으로** 계산합니다.

### 사각형 법 (Rectangle Rule)

```python
def rectangle_integrate(f, a, b, n=10000):
    """가장 단순한 수치 적분"""
    dx = (b - a) / n
    x = torch.linspace(a, b, n)
    return (f(x) * dx).sum()

# ∫₀¹ x² dx
result = rectangle_integrate(lambda x: x**2, 0, 1)
print(f"사각형법: {result:.6f}")  # ≈ 0.333333
```

### 사다리꼴 법 (Trapezoidal Rule)

```python
def trapezoidal_integrate(f, a, b, n=10000):
    """더 정확한 수치 적분"""
    x = torch.linspace(a, b, n)
    y = f(x)
    dx = (b - a) / (n - 1)
    return dx * (y[0]/2 + y[1:-1].sum() + y[-1]/2)

result = trapezoidal_integrate(lambda x: x**2, 0, 1)
print(f"사다리꼴법: {result:.6f}")  # ≈ 0.333333
```

### 딥러닝에서는 왜 직접 적분 안 하나?

| 방법 | 장점 | 단점 | 사용 장면 |
|------|------|------|----------|
| 해석적 적분 | 정확 | 풀이 불가능할 때 많음 | 간단한 분포 |
| 수치 적분 | 범용 | 고차원에서 느림 | 1~3차원 |
| **Monte Carlo** | 고차원 OK | 분산 큼 | **딥러닝 (VAE, RL 등)** |

→ 딥러닝에서는 수백 차원의 적분이 필요 → **Monte Carlo 샘플링**으로 근사

---

## 딥러닝에서의 적분

### 1. 확률분포의 정규화

확률밀도함수(PDF)는 적분하면 반드시 1:

$$
\int_{-\infty}^{\infty} f(x)\,dx = 1
$$

```python
from torch.distributions import Normal

dist = Normal(0, 1)

# 수치적으로 확인
dx = 0.001
x = torch.arange(-10, 10, dx)
pdf = torch.exp(dist.log_prob(x))
total = (pdf * dx).sum()
print(f"PDF 전체 적분: {total:.6f}")  # ≈ 1.0
```

### 2. 기댓값 = 적분

$$
\mathbb{E}[g(X)] = \int g(x) f(x)\,dx
$$

```python
# E[X²] for X ~ N(0, 1) → 분산 = 1
dx = 0.001
x = torch.arange(-10, 10, dx)
pdf = torch.exp(dist.log_prob(x))

# E[X] = ∫ x·f(x) dx
E_X = (x * pdf * dx).sum()
print(f"E[X] = {E_X:.4f}")  # ≈ 0.0

# E[X²] = ∫ x²·f(x) dx
E_X2 = (x**2 * pdf * dx).sum()
print(f"E[X²] = {E_X2:.4f}")  # ≈ 1.0 (분산)
```

### 3. Monte Carlo 적분: 딥러닝의 핵심 트릭

적분을 **샘플 평균**으로 근사:

$$
\int g(x) f(x)\,dx \approx \frac{1}{N}\sum_{i=1}^N g(x_i), \quad x_i \sim f(x)
$$

```python
# Monte Carlo로 E[X²] 계산
N = 100000
samples = dist.sample((N,))

# 샘플 평균 = 적분 근사
E_X2_mc = (samples**2).mean()
print(f"Monte Carlo E[X²] = {E_X2_mc:.4f}")  # ≈ 1.0

# VAE에서의 ELBO도 이렇게 계산!
# E_q[log p(x|z)] ≈ (1/N) Σ log p(x|z_i), z_i ~ q(z|x)
```

### 4. 정규화 상수

많은 확률 모델에서 적분이 필요합니다:

$$
p(x) = \frac{1}{Z} \tilde{p}(x), \quad Z = \int \tilde{p}(x)\,dx
$$

```python
# 정규분포의 정규화 상수
# f(x) = exp(-x²/2)의 적분 = √(2π)
import math

dx = 0.001
x = torch.arange(-10, 10, dx)
unnormalized = torch.exp(-x**2 / 2)
Z = (unnormalized * dx).sum()
print(f"정규화 상수 Z = {Z:.4f}")          # ≈ 2.5066
print(f"√(2π) = {math.sqrt(2*math.pi):.4f}")  # 2.5066
```

---

## 핵심 정리

| 개념 | 수식 | 의미 | 딥러닝 적용 |
|------|------|------|------------|
| 부정적분 | $\int f(x)\,dx = F(x)+C$ | 미분의 역과정 | 이론적 기초 |
| 정적분 | $\int_a^b f(x)\,dx$ | 구간의 넓이 | 확률 계산 |
| PDF 정규화 | $\int f(x)\,dx = 1$ | 확률의 총합 | 모든 확률 모델 |
| 기댓값 | $\int x f(x)\,dx$ | 평균 | Loss 함수, ELBO |
| Monte Carlo | $\frac{1}{N}\sum g(x_i)$ | 샘플링으로 적분 근사 | VAE, RL, Diffusion |

## 핵심 통찰

1. **적분 = 전부 더하기**: 확률, 기댓값, 넓이 — 모두 "잘게 쪼개서 합산"
2. **PDF의 넓이 = 1**: 확률분포의 가장 기본적인 조건이 적분
3. **고차원에서는 Monte Carlo**: 딥러닝의 적분은 거의 모두 샘플링으로 근사
4. **정규화 상수가 어려운 이유**: $Z = \int \tilde{p}(x)\,dx$를 닫힌 형태로 구할 수 없는 경우가 많음

---

## 다음 단계

적분의 기초를 이해했습니다. 이제 **함수를 다항식으로 근사하는 방법**을 알아봅니다.

→ [테일러 급수](/ko/docs/math/calculus/taylor-series): 복잡한 함수를 간단하게 근사하기

## 관련 콘텐츠

- [미분 기초](/ko/docs/math/calculus/basics) — 적분의 역연산
- [확률분포](/ko/docs/math/probability/distribution) — PDF와 적분의 관계
- [기댓값](/ko/docs/math/probability/expectation) — 기댓값 = 적분
- [KL-divergence](/ko/docs/math/probability/kl-divergence) — 분포 간 거리 (적분으로 정의)
