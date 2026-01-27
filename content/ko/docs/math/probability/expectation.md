---
title: "기댓값과 분산"
weight: 3
math: true
---

# 기댓값과 분산 (Expectation & Variance)

{{% hint info %}}
**선수지식**: [확률 변수](/ko/docs/math/probability/random-variable) (이산/연속 구분)
{{% /hint %}}

## 한 줄 요약

> **기댓값 = "평균적으로 얼마?"** | **분산 = "얼마나 퍼져있어?"**

---

## 왜 기댓값과 분산을 배워야 하나요?

### 문제 상황 1: Batch Normalization은 뭘 하는 건가요?

```python
# PyTorch의 BatchNorm
nn.BatchNorm2d(64)  # 이게 뭘 하는 거지?
```

**정답**: 데이터의 **평균을 0으로, 분산을 1로** 만듭니다!

$$
\hat{x} = \frac{x - \text{평균}}{\sqrt{\text{분산}}}
$$

### 문제 상황 2: 왜 가중치 초기화가 중요한가요?

```python
# 이 두 초기화의 차이는?
nn.init.xavier_uniform_(layer.weight)
nn.init.kaiming_normal_(layer.weight)
```

**정답**: 출력의 **분산**을 유지하기 위해서입니다!
- 분산이 너무 커지면 → 폭발 (gradient exploding)
- 분산이 너무 작아지면 → 소멸 (gradient vanishing)

---

## 1. 기댓값 (Expectation) = 평균

### 직관적 정의

> **기댓값** = "오래 반복하면 평균적으로 이 값이 나온다"

### 예시: 주사위의 기댓값

```
주사위를 1000번 던지면 평균이 얼마일까?

1이 나올 확률 = 1/6  → 1 × (1/6)
2가 나올 확률 = 1/6  → 2 × (1/6)
3이 나올 확률 = 1/6  → 3 × (1/6)
...
6이 나올 확률 = 1/6  → 6 × (1/6)

기댓값 = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
      = (1+2+3+4+5+6)/6
      = 21/6
      = 3.5
```

주사위에서 3.5는 절대 안 나오지만, **평균적으로** 3.5입니다!

### 수학적 정의

**이산** (셀 수 있는 경우):
$$
\mathbb{E}[X] = \sum_x x \cdot P(X=x)
$$

**연속** (무한히 많은 경우):
$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
$$

### 기댓값의 중요한 성질

| 성질 | 수식 | 의미 |
|------|------|------|
| **선형성** | $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$ | 상수 곱/덧셈 가능 |
| **합의 기댓값** | $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ | 항상 성립! |
| **곱의 기댓값** | $\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y]$ | **독립일 때만!** |

---

## 2. 분산 (Variance) = 퍼짐 정도

### 직관적 정의

> **분산** = "평균에서 얼마나 떨어져 있는가"의 평균

### 시각적 이해

```
분산이 작음 (집중됨)           분산이 큼 (퍼져있음)

        ▲                           ▲
        │                           │
      █████                      █     █
      █████                    █   █   █
    █████████                █   █   █   █
──────┼──────────        ──────┼──────────────
      μ                         μ

같은 평균이라도 퍼짐이 다름!
```

### 수학적 정의

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
$$

**해석**: "(값 - 평균)²"의 평균

### 계산에 편리한 공식

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**해석**: "제곱의 평균 - 평균의 제곱"

### 표준편차 (Standard Deviation)

$$
\sigma = \sqrt{\text{Var}(X)}
$$

**왜 필요한가?**: 분산은 단위가 제곱이라서 해석이 어려움.
- 키의 분산: 25 cm² (???)
- 키의 표준편차: 5 cm (이해하기 쉬움!)

### 분산의 중요한 성질

| 성질 | 수식 | 의미 |
|------|------|------|
| **상수** | $\text{Var}(c) = 0$ | 상수는 안 퍼짐 |
| **스케일링** | $\text{Var}(aX) = a^2 \text{Var}(X)$ | 2배 → 분산 4배 |
| **합의 분산** | $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ | **독립일 때만!** |

---

## 3. 딥러닝에서 기댓값과 분산

### 1) Batch Normalization

**문제**: 층을 지나면서 분포가 이상해진다 (Internal Covariate Shift)

**해결**: 각 층에서 평균=0, 분산=1로 정규화!

```python
def batch_norm(x, eps=1e-5):
    mean = x.mean(dim=0)      # 평균 계산
    var = x.var(dim=0)        # 분산 계산
    x_norm = (x - mean) / torch.sqrt(var + eps)  # 정규화
    return x_norm

# 정규화 후
# E[x_norm] ≈ 0
# Var(x_norm) ≈ 1
```

### 2) 가중치 초기화

**문제**: 가중치를 어떻게 초기화해야 학습이 잘 될까?

**해결**: 출력의 **분산**이 유지되도록!

```python
# Xavier 초기화 (tanh, sigmoid 용)
# Var(W) = 2 / (n_in + n_out)
nn.init.xavier_uniform_(layer.weight)

# He 초기화 (ReLU 용)
# Var(W) = 2 / n_in
nn.init.kaiming_normal_(layer.weight)
```

**왜?**: 입력의 분산 = 출력의 분산이면 gradient가 안정적!

### 3) MSE Loss

**MSE = 기댓값!**

$$
\text{MSE} = \mathbb{E}[(y - \hat{y})^2]
$$

실제로는 배치 평균으로 근사:

```python
loss = ((y - y_pred) ** 2).mean()  # E[(y - ŷ)²]
```

### 4) Dropout

**문제**: Dropout하면 기댓값이 변한다!

```
학습 시: 50% 확률로 뉴런을 끔
→ E[output] = 0.5 × x + 0.5 × 0 = 0.5x  (절반으로 줄어듦!)
```

**해결**: 스케일링으로 보정

```python
# 방법 1: 테스트 시 0.5 곱하기
# 방법 2: 학습 시 2 곱하기 (inverted dropout)
x = x * mask / (1 - dropout_prob)  # PyTorch 방식
```

---

## 4. 공분산과 상관계수

### 공분산 (Covariance)

> "두 변수가 **함께** 변하는 정도"

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

```
Cov > 0: X가 크면 Y도 큰 경향 (양의 상관)
Cov < 0: X가 크면 Y가 작은 경향 (음의 상관)
Cov = 0: 관계 없음 (무상관)
```

### 상관계수 (Correlation)

> 공분산을 -1 ~ 1 범위로 정규화

$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

```
ρ = 1:  완벽한 양의 상관 (직선 관계)
ρ = 0:  무상관
ρ = -1: 완벽한 음의 상관
```

### 딥러닝에서 왜 중요한가?

**PCA, Whitening**: 특성 간 상관관계를 제거
**Loss 함수**: Contrastive Learning에서 상관관계 활용

---

## 5. 조건부 기댓값과 회귀

### 핵심 통찰

> **회귀 모델 = 조건부 기댓값 학습!**

$$
f^*(x) = \mathbb{E}[Y | X = x]
$$

**해석**: "입력 x가 주어졌을 때, 출력 y의 평균값을 예측"

### 왜 MSE인가?

$$
\arg\min_f \mathbb{E}[(Y - f(X))^2] = \mathbb{E}[Y|X]
$$

**MSE를 최소화하는 함수 = 조건부 기댓값!**

```python
# 회귀 모델 학습 = 조건부 기댓값 학습
model.fit(X, y)  # f(x) ≈ E[Y|X=x]
```

---

## 코드로 확인하기

```python
import numpy as np
import torch
import torch.nn as nn

# === 기댓값과 분산 계산 ===
print("=== 기본 계산 ===")
data = np.array([1, 2, 3, 4, 5, 6])

mean = np.mean(data)      # 기댓값
var = np.var(data)        # 분산
std = np.std(data)        # 표준편차

print(f"기댓값 E[X] = {mean}")       # 3.5
print(f"분산 Var(X) = {var:.2f}")   # 2.92
print(f"표준편차 σ = {std:.2f}")    # 1.71

# === Batch Normalization ===
print("\n=== Batch Normalization ===")
batch = torch.randn(32, 64)  # 배치 크기 32, 특성 64

bn = nn.BatchNorm1d(64)
normalized = bn(batch)

print(f"정규화 전 - 평균: {batch.mean():.4f}, 분산: {batch.var():.4f}")
print(f"정규화 후 - 평균: {normalized.mean():.4f}, 분산: {normalized.var():.4f}")

# === 가중치 초기화 비교 ===
print("\n=== 가중치 초기화 ===")
linear = nn.Linear(1000, 1000, bias=False)

# Xavier 초기화
nn.init.xavier_uniform_(linear.weight)
print(f"Xavier - Var(W): {linear.weight.var():.6f}")

# He 초기화
nn.init.kaiming_normal_(linear.weight)
print(f"He     - Var(W): {linear.weight.var():.6f}")

# === Dropout 기댓값 보정 ===
print("\n=== Dropout ===")
x = torch.ones(1000) * 10  # 모두 10인 벡터
dropout = nn.Dropout(p=0.5)

# 학습 모드
dropout.train()
x_dropped = dropout(x)
print(f"Dropout 후 평균: {x_dropped.mean():.2f}")  # 여전히 ≈ 10 (보정됨)

# === 공분산/상관계수 ===
print("\n=== 공분산/상관계수 ===")
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

cov = np.cov(X, Y, ddof=0)[0, 1]
corr = np.corrcoef(X, Y)[0, 1]

print(f"Cov(X, Y) = {cov:.2f}")
print(f"Corr(X, Y) = {corr:.2f}")
```

---

## 주요 분포의 기댓값과 분산

| 분포 | 기댓값 $\mathbb{E}[X]$ | 분산 $\text{Var}(X)$ |
|------|----------------------|---------------------|
| 베르누이(p) | $p$ | $p(1-p)$ |
| 균등(a, b) | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| 정규(μ, σ²) | $\mu$ | $\sigma^2$ |
| 지수(λ) | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 활용 |
|------|------|-----------|
| **기댓값** | $\mathbb{E}[X]$ | Loss 계산, Dropout 보정 |
| **분산** | $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | BatchNorm, 초기화 |
| **표준편차** | $\sqrt{\text{Var}(X)}$ | 정규화, 시각화 |
| **공분산** | $\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ | PCA, 특성 분석 |

---

## 다음 단계

주요 확률 **분포**들을 자세히 알고 싶다면?
→ [확률분포](/ko/docs/math/probability/distribution)로!

분포에서 **값을 뽑는** 방법이 궁금하다면?
→ [샘플링](/ko/docs/math/probability/sampling)으로!

---

## 관련 콘텐츠

- [확률 변수](/ko/docs/math/probability/random-variable) - 선수 지식
- [확률분포](/ko/docs/math/probability/distribution) - 분포별 기댓값/분산
- [Batch Normalization](/ko/docs/math/normalization/batch-norm) - 분산 정규화
- [가중치 초기화](/ko/docs/math/training/regularization) - 분산 유지 초기화
