---
title: "결합/조건부 분포"
weight: 11
math: true
---

# 결합 분포와 조건부 분포 (Joint & Conditional Distribution)

{{% hint info %}}
**선수지식**: [확률 변수](/ko/docs/math/probability/random-variable) | [확률분포](/ko/docs/math/probability/distribution)
{{% /hint %}}

## 한 줄 요약

> **결합 분포 = "여러 변수를 동시에"** | **조건부 분포 = "하나를 알 때 나머지"**

---

## 왜 결합/조건부 분포를 배워야 하나요?

### 문제 상황 1: 분류 모델의 출력은 뭘 의미하나요?

```python
output = model(image)
probs = softmax(output)
# → [0.85, 0.10, 0.05]  (고양이, 강아지, 새)
```

**정답**: 이건 **조건부 확률** $P(\text{class} | \text{image})$입니다!
- "이 이미지가 주어졌을 때, 각 클래스일 확률"

### 문제 상황 2: Diffusion 모델의 조건부 생성은 어떻게 작동하나요?

```python
# 텍스트 조건부 이미지 생성
image = pipeline("a cat sitting on a chair")
```

**정답**: $P(\text{image} | \text{text})$를 학습합니다!
- 결합 분포 $P(\text{image}, \text{text})$에서
- 텍스트가 주어졌을 때의 **조건부 분포**로 이미지 생성

### 문제 상황 3: 베이즈 추론은 어떻게 결합 분포를 사용하나요?

```python
# 사후 확률 = 사전 × 가능도 / 증거
P(θ|data) = P(data|θ) × P(θ) / P(data)
```

**정답**: 이 모든 것이 **결합 분포** $P(\theta, \text{data})$의 분해입니다!

---

## 1. 결합 분포 (Joint Distribution)

### 직관적 정의

> **결합 분포** = "두 개 이상의 변수가 동시에 어떤 값을 가지는지"

### 이산 확률 변수의 결합 분포

$$
P(X=x, Y=y) = P_{XY}(x, y)
$$

### 예시: 날씨와 아이스크림 판매

![결합 분포 히트맵](/images/probability/ko/joint-distribution-heatmap.png)

```
→ P(맑음, 많이) = 0.35  (맑은 날 아이스크림 많이 파는 확률)
→ P(비, 적게)   = 0.20  (비 오는 날 적게 파는 확률)
→ 모든 칸의 합  = 1.00
```

### 연속 확률 변수의 결합 분포

$$
f_{XY}(x, y)
$$

**조건**: $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{XY}(x,y) \, dx \, dy = 1$

---

## 2. 주변 분포 (Marginal Distribution)

### 직관적 정의

> **주변 분포** = "결합 분포에서 한 변수를 무시하고 나머지만 본 것"

### 수학적 정의

**이산**:
$$
P(X=x) = \sum_y P(X=x, Y=y)
$$

**연속**:
$$
f_X(x) = \int_{-\infty}^{\infty} f_{XY}(x, y) \, dy
$$

### 예시: 날씨의 주변 분포

```
날씨(X)의 주변 분포:
P(맑음) = 0.30 + 0.10 + 0.02 = 0.42  (맑음 행의 합)
P(흐림) = 0.10 + 0.15 + 0.08 = 0.33
P(비)   = 0.05 + 0.10 + 0.10 = 0.25

아이스크림(Y)의 주변 분포:
P(많이) = 0.30 + 0.10 + 0.05 = 0.45  (많이 열의 합)
P(보통) = 0.10 + 0.15 + 0.10 = 0.35
P(적게) = 0.02 + 0.08 + 0.10 = 0.20
```

### 핵심 관계

![결합→주변→조건부 분포 관계](/images/probability/ko/joint-marginal-conditional.png)

---

## 3. 조건부 분포 (Conditional Distribution)

### 직관적 정의

> **조건부 분포** = "하나의 값을 알 때, 나머지의 분포"

### 수학적 정의

$$
P(Y=y | X=x) = \frac{P(X=x, Y=y)}{P(X=x)}
$$

**해석**: $\frac{\text{결합 분포}}{\text{주변 분포}}$

### 예시: 맑은 날의 아이스크림 판매

```
P(Y | X=맑음) = P(X=맑음, Y) / P(X=맑음)

P(많이 | 맑음) = 0.30 / 0.42 = 0.71
P(보통 | 맑음) = 0.10 / 0.42 = 0.24
P(적게 | 맑음) = 0.02 / 0.42 = 0.05

→ 맑은 날은 아이스크림이 많이 팔릴 확률이 71%!

P(Y | X=비):
P(많이 | 비)   = 0.05 / 0.25 = 0.20
P(보통 | 비)   = 0.10 / 0.25 = 0.40
P(적게 | 비)   = 0.10 / 0.25 = 0.40

→ 비 오는 날은 적게 팔릴 확률이 높음!
```

### 연속 확률 변수의 조건부 분포

$$
f_{Y|X}(y|x) = \frac{f_{XY}(x, y)}{f_X(x)}
$$

---

## 4. 핵심 관계: 곱셈 법칙

### 결합 = 조건부 × 주변

$$
P(X, Y) = P(Y|X) \cdot P(X) = P(X|Y) \cdot P(Y)
$$

**이 관계가 중요한 이유**:
- 결합 분포를 직접 구하기 어려울 때
- 조건부 분포와 주변 분포로 **분해**할 수 있음

### 베이즈 정리로의 연결

$$
P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}
$$

이것은 곱셈 법칙의 **직접적인 결과**입니다!

```
P(X, Y) = P(Y|X) · P(X) = P(X|Y) · P(Y)

양변을 P(X)로 나누면:
P(Y|X) = P(X|Y) · P(Y) / P(X)

→ 베이즈 정리!
```

---

## 5. 독립과 결합 분포

### 독립의 정의

X와 Y가 **독립**이면:

$$
P(X, Y) = P(X) \cdot P(Y)
$$

**해석**: 결합 분포 = 주변 분포의 곱

### 독립이면 조건부 = 주변

$$
P(Y|X) = P(Y)
$$

"X를 알아도 Y에 대한 정보가 없다"

### 예시: 독립 vs 종속

```
독립: 주사위 두 개
P(주사위1=3, 주사위2=5) = P(주사위1=3) × P(주사위2=5)
= (1/6) × (1/6) = 1/36

종속: 오늘 날씨와 아이스크림
P(맑음, 많이) = 0.30 ≠ P(맑음) × P(많이) = 0.42 × 0.45 = 0.189
→ 독립이 아님!
```

---

## 6. 딥러닝에서의 활용

### 1) 분류 모델 = 조건부 확률 학습

```python
# 분류 모델은 P(Y|X)를 학습한다!
# Y: 클래스, X: 이미지

model = ResNet50(num_classes=1000)
logits = model(image)               # 점수
probs = F.softmax(logits, dim=-1)   # P(Y|X=image)

# probs[i] = P(Y=i | X=image)
```

### 2) 생성 모델 = 결합 분포 학습

```python
# VAE: P(X, Z) = P(X|Z) · P(Z)
#   - P(Z): 사전 분포 (가우시안)
#   - P(X|Z): 디코더 (조건부 분포)

# Diffusion: P(X₀, X₁, ..., X_T)
#   - 결합 분포를 마르코프 체인으로 분해
#   P(X₀:T) = P(X_T) × ∏ P(X_{t-1}|X_t)
```

### 3) 조건부 생성

```python
# Text-to-Image: P(image | text)를 학습
image = stable_diffusion("a sunset over the ocean")

# Class-conditional: P(image | class)를 학습
image = conditional_gan(class_label=3)

# 핵심: 모두 조건부 분포!
```

### 4) 전체 확률의 법칙 (Marginalization)

$$
P(X) = \sum_z P(X|Z=z) P(Z=z) = \int P(X|Z=z) P(Z=z) \, dz
$$

```python
# VAE에서 이미지의 확률:
# P(X) = ∫ P(X|Z) P(Z) dZ
# → 직접 계산 불가능! (적분이 어려움)
# → ELBO로 근사

# 이것이 VAE의 핵심 동기!
```

---

## 코드로 확인하기

```python
import numpy as np

# === 결합 분포 테이블 ===
print("=== 결합 분포 ===")

# 날씨(X) × 아이스크림(Y) 결합 분포
joint = np.array([
    [0.30, 0.10, 0.02],  # 맑음: 많이, 보통, 적게
    [0.10, 0.15, 0.08],  # 흐림
    [0.05, 0.10, 0.10],  # 비
])
weather = ['맑음', '흐림', '비']
sales = ['많이', '보통', '적게']

print("결합 분포 테이블:")
print(f"         {'  '.join(sales)}")
for i, w in enumerate(weather):
    print(f"  {w}   {joint[i]}")
print(f"합계: {joint.sum():.2f}")

# === 주변 분포 ===
print("\n=== 주변 분포 ===")
marginal_weather = joint.sum(axis=1)  # 행 합
marginal_sales = joint.sum(axis=0)    # 열 합

print(f"P(날씨): {dict(zip(weather, marginal_weather.round(2)))}")
print(f"P(판매): {dict(zip(sales, marginal_sales.round(2)))}")

# === 조건부 분포 ===
print("\n=== 조건부 분포 ===")

# P(Y | X=맑음)
cond_sunny = joint[0] / marginal_weather[0]
print(f"P(판매 | 맑음): {dict(zip(sales, cond_sunny.round(3)))}")

# P(Y | X=비)
cond_rainy = joint[2] / marginal_weather[2]
print(f"P(판매 | 비):   {dict(zip(sales, cond_rainy.round(3)))}")

# 조건부 분포의 합 = 1 확인
print(f"조건부 분포 합: {cond_sunny.sum():.2f}")

# === 독립성 검정 ===
print("\n=== 독립성 검정 ===")

# 독립이면: P(X,Y) = P(X) × P(Y)
independent_joint = np.outer(marginal_weather, marginal_sales)
print("독립 가정 시 결합 분포:")
print(independent_joint.round(3))
print("\n실제 결합 분포:")
print(joint)
print(f"\n독립인가? {np.allclose(joint, independent_joint, atol=0.01)}")

# === 곱셈 법칙 확인 ===
print("\n=== 곱셈 법칙 ===")
# P(X=맑음, Y=많이) = P(Y=많이|X=맑음) × P(X=맑음)
lhs = joint[0, 0]
rhs = cond_sunny[0] * marginal_weather[0]
print(f"P(맑음, 많이) = {lhs:.3f}")
print(f"P(많이|맑음) × P(맑음) = {rhs:.3f}")
print(f"일치: {np.isclose(lhs, rhs)}")

# === 분류 모델의 조건부 확률 ===
print("\n=== 분류 모델 = 조건부 확률 ===")
import torch
import torch.nn.functional as F

# 모델 출력 (logits)
logits = torch.tensor([[3.0, 1.5, 0.5]])

# P(class | input) = softmax(logits)
probs = F.softmax(logits, dim=-1)
print(f"P(class | image) = {probs.tolist()[0]}")
print(f"합: {probs.sum().item():.2f}")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 활용 |
|------|------|-------------|
| **결합 분포** | $P(X, Y)$ | 생성 모델이 학습하는 대상 |
| **주변 분포** | $P(X) = \sum_y P(X,Y)$ | VAE의 ELBO 유도 |
| **조건부 분포** | $P(Y\|X) = P(X,Y)/P(X)$ | 분류, 조건부 생성 |
| **곱셈 법칙** | $P(X,Y) = P(Y\|X)P(X)$ | 베이즈, 체인 룰 |
| **독립** | $P(X,Y) = P(X)P(Y)$ | 가정 단순화 |

---

## 핵심 통찰

```
1. 결합 분포 = 여러 변수의 "전체 그림"
   - 모든 정보를 담고 있음
   - 주변, 조건부 분포를 유도할 수 있음

2. 조건부 분포 = 딥러닝의 핵심!
   - 분류 = P(class | image)
   - 생성 = P(image | condition)
   - 추론 = P(parameter | data)

3. 결합 → 분해 → 계산
   - 복잡한 결합 분포를 조건부의 곱으로 분해
   - P(A,B,C) = P(C|A,B) · P(B|A) · P(A)
   - 이것이 Chain Rule of Probability!
```

---

## 다음 단계

여러 변수의 **가우시안 분포**가 궁금하다면?
→ [다변량 가우시안](/ko/docs/math/probability/multivariate-gaussian)으로!

"X를 알면 Y의 불확실성이 줄어드는 정도"를 측정하고 싶다면?
→ [상호 정보량](/ko/docs/math/probability/mutual-information)으로!

---

## 관련 콘텐츠

- [확률 변수](/ko/docs/math/probability/random-variable) - 선수 지식
- [확률분포](/ko/docs/math/probability/distribution) - 선수 지식
- [베이즈 정리](/ko/docs/math/probability/bayes) - 조건부 확률의 핵심 응용
- [다변량 가우시안](/ko/docs/math/probability/multivariate-gaussian) - 연속 결합 분포의 대표
- [VAE](/ko/docs/architecture/generative/vae) - 결합/조건부 분포 활용
