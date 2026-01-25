---
title: "확률의 기초"
weight: 1
math: true
---

# 확률의 기초 (Probability Basics)

## 개요

확률은 불확실성을 수치로 표현하는 방법입니다. 딥러닝에서 모든 예측은 확률로 표현됩니다.

---

## 확률의 정의

### 고전적 정의

$$
P(A) = \frac{\text{사건 A가 일어나는 경우의 수}}{\text{전체 경우의 수}}
$$

**예시**: 주사위에서 짝수가 나올 확률
$$
P(\text{짝수}) = \frac{3}{6} = 0.5
$$

### 확률의 공리 (Kolmogorov)

1. **비음수성**: $P(A) \geq 0$
2. **정규성**: $P(\Omega) = 1$ (전체 사건의 확률은 1)
3. **가산 가법성**: 서로 배반인 사건들에 대해 $P(A \cup B) = P(A) + P(B)$

---

## 조건부 확률 (Conditional Probability) ⭐

> **A가 일어났을 때, B가 일어날 확률은?**

$$
P(B|A) = \frac{P(A \cap B)}{P(A)}
$$

### 시각적 이해

![조건부 확률](/images/probability/ko/conditional-probability.svg)

### 쉬운 비유

> 🎯 **레스토랑 비유**
> - 전체 손님 중 30%가 디저트를 주문 → P(디저트)
> - 스테이크 주문한 손님 중 50%가 디저트도 주문 → P(디저트|스테이크)

### 딥러닝에서의 예시

- **분류**: $P(\text{고양이}|\text{이미지})$ = 이미지가 주어졌을 때 고양이일 확률
- **언어 모델**: $P(\text{다음 토큰}|\text{이전 토큰들})$

---

## 결합 확률과 주변 확률

### 시각적 이해

![결합 확률과 주변 확률](/images/probability/ko/joint-marginal.svg)

### 결합 확률 (Joint Probability)

두 사건이 동시에 일어날 확률:

$$
P(A, B) = P(A \cap B) = P(B|A) \cdot P(A) = P(A|B) \cdot P(B)
$$

### 주변 확률 (Marginal Probability)

결합 확률에서 하나의 변수를 "합산"하여 얻는 확률:

$$
P(A) = \sum_B P(A, B)
$$

연속인 경우:
$$
P(A) = \int P(A, B) \, dB
$$

### 예시: 이미지 분류

| | 고양이 (C) | 강아지 (D) | 합계 |
|---|---|---|---|
| 실내 (I) | 0.3 | 0.2 | 0.5 |
| 실외 (O) | 0.1 | 0.4 | 0.5 |
| 합계 | 0.4 | 0.6 | 1.0 |

- 결합: $P(C, I) = 0.3$
- 주변: $P(C) = P(C,I) + P(C,O) = 0.3 + 0.1 = 0.4$

---

## 독립 (Independence)

두 사건이 서로 영향을 주지 않을 때:

$$
P(A, B) = P(A) \cdot P(B)
$$

또는 동치로:
$$
P(A|B) = P(A)
$$

### 조건부 독립 (Conditional Independence)

C가 주어졌을 때 A와 B가 독립:

$$
P(A, B | C) = P(A|C) \cdot P(B|C)
$$

**딥러닝 예시**: Naive Bayes 분류기
$$
P(x_1, x_2, ..., x_n | y) = \prod_i P(x_i | y)
$$

---

## 전확률 법칙 (Law of Total Probability)

$$
P(B) = \sum_i P(B|A_i) \cdot P(A_i)
$$

여기서 $\\{A_i\\}$는 전체 표본 공간의 분할 (서로 배반이고 합집합이 전체).

### 시각적 이해

![전확률 법칙](/images/probability/ko/total-probability.svg)

### 예시

```
P(모델 정확) = P(모델 정확|쉬운 문제) × P(쉬운 문제)
             + P(모델 정확|어려운 문제) × P(어려운 문제)
```

---

## 구현

```python
import numpy as np

# 조건부 확률 계산
def conditional_prob(joint_prob, marginal_prob):
    """P(B|A) = P(A,B) / P(A)"""
    return joint_prob / marginal_prob

# 예시: 결합 확률 테이블
joint = np.array([
    [0.3, 0.2],  # 실내: 고양이, 강아지
    [0.1, 0.4]   # 실외: 고양이, 강아지
])

# 주변 확률
p_indoor = joint[0].sum()  # P(실내) = 0.5
p_cat = joint[:, 0].sum()  # P(고양이) = 0.4

# 조건부 확률
p_cat_given_indoor = joint[0, 0] / p_indoor  # P(고양이|실내) = 0.6

print(f"P(실내) = {p_indoor}")
print(f"P(고양이) = {p_cat}")
print(f"P(고양이|실내) = {p_cat_given_indoor}")

# 독립성 검정
# 독립이면: P(고양이, 실내) = P(고양이) × P(실내)
independent = np.isclose(joint[0, 0], p_cat * p_indoor)
print(f"독립 여부: {independent}")  # False (0.3 ≠ 0.4 × 0.5 = 0.2)
```

---

## 핵심 정리

| 개념 | 수식 | 의미 |
|------|------|------|
| 조건부 확률 | $P(B\|A) = P(A,B)/P(A)$ | A가 일어났을 때 B의 확률 |
| 결합 확률 | $P(A,B) = P(A\|B) \cdot P(B)$ | 동시에 일어날 확률 |
| 주변 확률 | $P(A) = \sum_B P(A,B)$ | 하나의 변수에 대한 확률 |
| 독립 | $P(A,B) = P(A) \cdot P(B)$ | 서로 영향 없음 |

---

## 관련 콘텐츠

- [베이즈 정리](/ko/docs/math/probability/bayes) - 조건부 확률의 역산
- [확률 변수](/ko/docs/math/probability/random-variable) - 확률의 함수적 표현
- [확률분포](/ko/docs/math/probability/distribution) - 주요 분포들
