---
title: "기댓값과 분산"
weight: 3
math: true
---

# 기댓값과 분산 (Expectation & Variance)

## 개요

> 💡 **기댓값**: 평균적으로 어떤 값이 나올까?
> 💡 **분산**: 값들이 얼마나 퍼져 있을까?

딥러닝의 Batch Normalization, 초기화, 손실 함수 모두 이 개념을 사용합니다.

### 시각적 이해

![기댓값과 분산](/images/probability/ko/expectation-variance.svg)

---

## 기댓값 (Expectation)

### 정의

**이산**:
$$
\mathbb{E}[X] = \sum_x x \cdot P(X=x)
$$

**연속**:
$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
$$

### 직관적 이해

기댓값 = 확률로 가중평균한 값

```
주사위 기댓값:
E[X] = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
     = 3.5
```

### 기댓값의 성질

1. **선형성**: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
2. **상수**: $\mathbb{E}[c] = c$
3. **독립일 때**: $\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y]$

---

## 분산 (Variance)

### 정의

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

### 직관적 이해

분산 = 평균으로부터 얼마나 퍼져 있는가

```
분산이 작음 (집중)        분산이 큼 (퍼짐)

     ▲                        ▲
     │                        │
   █████                    █ █ █ █ █
   █████                    █ █ █ █ █
─────┼─────             ─────┼─────────
     μ                       μ
```

### 표준 편차 (Standard Deviation)

$$
\sigma = \sqrt{\text{Var}(X)}
$$

원래 데이터와 같은 단위를 가짐.

### 분산의 성질

1. $\text{Var}(c) = 0$
2. $\text{Var}(aX) = a^2 \text{Var}(X)$
3. **독립일 때**: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

---

## 공분산과 상관계수

### 공분산 (Covariance)

두 변수가 함께 변하는 정도:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

- $\text{Cov}(X, Y) > 0$: 양의 상관 (X↑ → Y↑)
- $\text{Cov}(X, Y) < 0$: 음의 상관 (X↑ → Y↓)
- $\text{Cov}(X, Y) = 0$: 무상관 (독립이면 무상관, 역은 성립 안함)

### 상관계수 (Correlation)

정규화된 공분산 (-1 ~ 1 범위):

$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

---

## 딥러닝에서의 활용

### 1. 손실 함수

MSE Loss는 기댓값:
$$
\mathcal{L} = \mathbb{E}[(y - \hat{y})^2]
$$

### 2. Batch Normalization

배치 내 평균과 분산으로 정규화:
$$
\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}(x) + \epsilon}}
$$

### 3. 가중치 초기화 (Xavier/He)

출력의 분산을 유지하도록 초기화:

**Xavier**: $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$

**He**: $\text{Var}(W) = \frac{2}{n_{in}}$

### 4. Dropout 기댓값 보정

학습 시 드롭아웃하면 기댓값이 줄어듦 → 스케일링으로 보정:

$$
\mathbb{E}[\text{dropout}(x)] = p \cdot x + (1-p) \cdot 0 = p \cdot x
$$

테스트 시 $x$를 $p$로 스케일링하거나, 학습 시 $\frac{1}{p}$로 스케일링.

---

## 구현

```python
import numpy as np

# 샘플 데이터
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# 기댓값 (표본 평균)
mean_X = np.mean(X)
mean_Y = np.mean(Y)
print(f"E[X] = {mean_X}, E[Y] = {mean_Y}")

# 분산
var_X = np.var(X)  # 모분산 (N으로 나눔)
var_X_sample = np.var(X, ddof=1)  # 표본분산 (N-1로 나눔)
print(f"Var(X) = {var_X}")

# 표준 편차
std_X = np.std(X)
print(f"Std(X) = {std_X}")

# 공분산
cov_XY = np.cov(X, Y, ddof=0)[0, 1]
print(f"Cov(X,Y) = {cov_XY}")

# 상관계수
corr_XY = np.corrcoef(X, Y)[0, 1]
print(f"Corr(X,Y) = {corr_XY}")

# Batch Normalization 구현
def batch_norm(x, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm

batch = np.random.randn(32, 64)  # 배치 크기 32, 특성 64
normalized = batch_norm(batch)
print(f"정규화 후 평균: {normalized.mean(axis=0).mean():.6f}")  # ≈ 0
print(f"정규화 후 분산: {normalized.var(axis=0).mean():.6f}")   # ≈ 1
```

---

## 조건부 기댓값

### 정의

$$
\mathbb{E}[Y|X=x] = \sum_y y \cdot P(Y=y|X=x)
$$

### 전기댓값 법칙 (Law of Total Expectation)

$$
\mathbb{E}[Y] = \mathbb{E}[\mathbb{E}[Y|X]]
$$

### 딥러닝에서의 의미

모델의 출력 $\hat{y} = f(x)$는 $\mathbb{E}[Y|X=x]$의 추정:

$$
f^*(x) = \mathbb{E}[Y|X=x] = \arg\min_f \mathbb{E}[(Y - f(X))^2]
$$

MSE를 최소화하면 조건부 기댓값을 학습!

---

## 주요 분포의 기댓값과 분산

| 분포 | $\mathbb{E}[X]$ | $\text{Var}(X)$ |
|------|-----------------|-----------------|
| 베르누이$(p)$ | $p$ | $p(1-p)$ |
| 이항$(n, p)$ | $np$ | $np(1-p)$ |
| 포아송$(\lambda)$ | $\lambda$ | $\lambda$ |
| 균등$(a, b)$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| 정규$(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ |
| 지수$(\lambda)$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

---

## 핵심 정리

| 개념 | 수식 | 의미 |
|------|------|------|
| 기댓값 | $\mathbb{E}[X]$ | 평균적인 값 |
| 분산 | $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | 퍼진 정도 |
| 공분산 | $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ | 함께 변하는 정도 |
| 상관계수 | $\rho = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ | 정규화된 공분산 |

---

## 관련 콘텐츠

- [확률 변수](/ko/docs/math/probability/random-variable) - 선수 지식
- [확률분포](/ko/docs/math/probability/distribution) - 분포별 기댓값/분산
- [Batch Normalization](/ko/docs/math/normalization/batch-norm) - 분산 정규화
- [Xavier/He 초기화](/ko/docs/math/training/regularization/weight-decay) - 분산 유지 초기화
