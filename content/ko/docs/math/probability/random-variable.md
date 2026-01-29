---
title: "확률 변수"
weight: 2
math: true
---

# 확률 변수 (Random Variable)

{{% hint info %}}
**선수지식**: [확률의 기초](/ko/docs/math/probability/basics) (조건부 확률까지)
{{% /hint %}}

## 한 줄 요약

> **"불확실한 결과를 숫자로 바꾸는 방법"**

---

## 왜 확률 변수를 배워야 하나요?

### 문제 상황: 컴퓨터는 "고양이"를 모릅니다

```
사람: "동전 앞면이 나오면 승리!"
컴퓨터: "앞면"이 뭔가요? 숫자로 말해주세요.

사람: "이 사진은 고양이야"
컴퓨터: "고양이"가 뭔가요? 숫자로 말해주세요.
```

**해결**: 결과를 **숫자로 변환**하면 컴퓨터가 이해할 수 있습니다!

```
앞면 → 1,  뒷면 → 0
고양이 → 0,  강아지 → 1,  새 → 2
```

이렇게 **"결과를 숫자로 바꾸는 것"**이 바로 확률 변수입니다.

---

## 1. 확률 변수란?

### 직관적 정의

> **확률 변수** = 랜덤한 실험의 결과를 **숫자**로 표현한 것

### 예시: 동전 던지기

```
실험: 동전을 던진다
결과: 앞면 또는 뒷면

확률 변수 X를 정의:
- X = 1  (앞면이 나왔을 때)
- X = 0  (뒷면이 나왔을 때)
```

이제 "앞면이 나올 확률"을 **수식으로** 쓸 수 있습니다:
$$
P(X = 1) = 0.5
$$

### 수학적 정의 (참고용)

$$
X: \Omega \rightarrow \mathbb{R}
$$

결과 집합(Ω)을 실수(ℝ)로 대응시키는 **함수**입니다.

---

## 2. 이산 vs 연속: 가장 중요한 구분!

### 핵심 질문: "결과를 셀 수 있나요?"

| | 이산 (Discrete) | 연속 (Continuous) |
|---|:---:|:---:|
| **셀 수 있나?** | O (1, 2, 3, ...) | X (무한히 많음) |
| **예시** | 주사위, 동전, 클래스 | 키, 몸무게, 온도 |
| **딥러닝** | **분류** | **회귀** |

### 딥러닝에서의 구분

```python
# 이산: 분류 (Classification)
# → 결과가 "고양이", "강아지", "새" 중 하나
output = model(image)  # [0.9, 0.05, 0.05]
class_id = output.argmax()  # 0 (고양이)

# 연속: 회귀 (Regression)
# → 결과가 어떤 숫자든 가능
age = model(face_image)  # 25.7살
```

**핵심**: 분류 = 이산, 회귀 = 연속!

![이산 vs 연속](/images/probability/ko/discrete-vs-continuous.svg)

---

## 3. 이산 확률 변수

### 확률 질량 함수 (PMF: Probability Mass Function)

> "각 값이 나올 확률"을 표로 정리한 것

$$
P(X = x) = p(x)
$$

### 예시: 주사위

```
X = 주사위 눈

P(X=1) = 1/6
P(X=2) = 1/6
P(X=3) = 1/6
...
P(X=6) = 1/6

→ 모두 합하면: 1/6 × 6 = 1 (100%)
```

### 시각화

```
PMF: 막대그래프로 표현

P(X)
  │
1/6├──┬──┬──┬──┬──┬──
  │  █  █  █  █  █  █
  └──┴──┴──┴──┴──┴──→ X
     1  2  3  4  5  6
```

### 딥러닝에서의 PMF: Softmax 출력!

```python
# 모델의 Softmax 출력 = PMF
logits = model(image)  # [2.0, 1.0, 0.5]
probs = softmax(logits)  # [0.7, 0.2, 0.1]

# 이게 바로 PMF!
# P(X=고양이) = 0.7
# P(X=강아지) = 0.2
# P(X=새) = 0.1
# 합: 0.7 + 0.2 + 0.1 = 1.0 ✓
```

---

## 4. 연속 확률 변수

### 문제: 연속에서는 "정확히 그 값"의 확률이 0!

```
질문: 키가 정확히 175.000000...cm일 확률은?
답: 0! (175.0001cm도 아니고 174.9999cm도 아닌 딱 그 값?)
```

무한히 많은 값 중 하나니까, 확률이 사실상 0입니다.

### 해결: 확률 밀도 함수 (PDF: Probability Density Function)

> "특정 값"이 아닌 **"범위"**의 확률을 계산

$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

### 시각화

```
PDF: 부드러운 곡선
        ↓ 면적 = 확률
      ╭─────╮
     ╱ ░░░░░ ╲
────╱─░░░░░░░─╲────→ X
    a         b

P(a ≤ X ≤ b) = 색칠된 면적
```

### 딥러닝에서의 PDF: 가우시안 노이즈

```python
# Diffusion 모델의 노이즈
noise = torch.randn(shape)  # N(0, 1)에서 샘플링

# VAE의 잠재 공간
z = mu + sigma * torch.randn_like(sigma)  # N(mu, sigma^2)
```

---

## 5. PMF vs PDF 비교

![PMF vs PDF vs CDF](/images/probability/ko/pmf-pdf-cdf.svg)

| | PMF (이산) | PDF (연속) |
|---|:---:|:---:|
| **의미** | 특정 값의 확률 | 밀도 (면적이 확률) |
| **P(X=x)** | 0보다 클 수 있음 | 항상 0 |
| **확률 계산** | $\sum$ (더하기) | $\int$ (적분) |
| **합/적분** | = 1 | = 1 |
| **그래프** | 막대 | 곡선 |

### 핵심 차이

```
이산: P(X=2) = 1/6  ← 값 자체가 확률!

연속: P(X=1.5) = 0  ← 항상 0
     P(1 < X < 2) > 0  ← 범위로 물어야 함
```

---

## 6. 누적 분포 함수 (CDF)

### 정의

> "X가 x 이하일 확률"

$$
F(x) = P(X \leq x)
$$

### 왜 필요한가요?

**문제**: "상위 5%에 들려면 몇 점이 필요한가요?"

```python
from scipy import stats

# 정규분포 N(50, 10^2) - 평균 50점, 표준편차 10점
exam = stats.norm(loc=50, scale=10)

# 상위 5% = 하위 95%
score = exam.ppf(0.95)  # 95% 분위수
print(f"상위 5% 커트라인: {score:.1f}점")  # 약 66.4점
```

### CDF 그래프

```
CDF: 항상 0에서 1로 증가

F(x)
  │        ╭────────
1 ├───────╯
  │    ╭──╯
  │ ╭─╯
  │╯
0 └─────────────────→ x
```

---

## 7. 딥러닝에서 확률 변수가 쓰이는 곳

### 1) 분류 출력 (이산)

```python
# Softmax 출력 = 카테고리 분포의 파라미터
probs = softmax(logits)  # [0.8, 0.15, 0.05]

# Y ~ Categorical(probs)
# Y는 이산 확률 변수
```

### 2) VAE 잠재 공간 (연속)

```python
# Encoder가 평균과 분산을 출력
mu = encoder_mu(x)      # 평균
sigma = encoder_std(x)   # 표준편차

# z ~ N(mu, sigma^2)
# z는 연속 확률 변수
z = mu + sigma * torch.randn_like(sigma)
```

### 3) Diffusion 노이즈 (연속)

```python
# 노이즈는 표준정규분포에서 샘플링
epsilon = torch.randn_like(x)  # ε ~ N(0, I)
```

### 4) Dropout (이산)

```python
# 각 뉴런을 끌지 말지 = 베르누이 분포
mask = torch.bernoulli(torch.full_like(x, 0.5))  # 50% 확률로 1
x = x * mask
```

---

## 코드로 확인하기

```python
import numpy as np
from scipy import stats

# === 이산 확률 변수 ===
print("=== 이산: 베르누이 분포 ===")
p = 0.3  # 성공 확률
bernoulli = stats.bernoulli(p)

# PMF 확인
print(f"P(X=0) = {bernoulli.pmf(0):.1f}")  # 0.7
print(f"P(X=1) = {bernoulli.pmf(1):.1f}")  # 0.3

# 샘플링
samples = bernoulli.rvs(size=1000)
print(f"1000번 샘플 평균: {samples.mean():.3f}")  # ≈ 0.3

# === 연속 확률 변수 ===
print("\n=== 연속: 정규분포 ===")
mu, sigma = 0, 1  # 표준정규분포
normal = stats.norm(loc=mu, scale=sigma)

# 특정 값의 확률 = 0
print(f"P(X=0) = 0 (연속이라서)")

# 범위의 확률
prob = normal.cdf(1) - normal.cdf(-1)
print(f"P(-1 < X < 1) = {prob:.3f}")  # ≈ 0.683 (68-95-99.7 규칙)

# 샘플링
samples = normal.rvs(size=1000)
print(f"1000번 샘플 평균: {samples.mean():.3f}")  # ≈ 0
print(f"1000번 샘플 표준편차: {samples.std():.3f}")  # ≈ 1

# === 딥러닝 스타일 ===
print("\n=== PyTorch 스타일 ===")
import torch

# 분류: Categorical (Softmax 출력)
logits = torch.tensor([2.0, 1.0, 0.5])
probs = torch.softmax(logits, dim=0)
print(f"Softmax 출력 (PMF): {probs.tolist()}")

# 생성: Gaussian 샘플링
z = torch.randn(3)  # N(0, 1)
print(f"가우시안 샘플: {z.tolist()}")
```

---

## 핵심 정리

| 개념 | 설명 | 딥러닝 예시 |
|------|------|-----------|
| **확률 변수** | 결과를 숫자로 표현 | 모든 입출력 |
| **이산** | 셀 수 있는 값 | 분류, Dropout |
| **연속** | 무한히 많은 값 | 회귀, VAE, Diffusion |
| **PMF** | 이산의 확률 | Softmax 출력 |
| **PDF** | 연속의 밀도 | 가우시안 노이즈 |
| **CDF** | 누적 확률 | 분위수 계산 |

---

## 다음 단계

확률 변수의 **평균**과 **퍼짐 정도**를 계산하고 싶다면?
→ [기댓값과 분산](/ko/docs/math/probability/expectation)으로!

주요 확률 **분포**들을 자세히 알고 싶다면?
→ [확률분포](/ko/docs/math/probability/distribution)로!

---

## 관련 콘텐츠

- [확률의 기초](/ko/docs/math/probability/basics) - 선수 지식
- [기댓값과 분산](/ko/docs/math/probability/expectation) - 확률 변수의 특성
- [확률분포](/ko/docs/math/probability/distribution) - 주요 분포 상세
- [샘플링](/ko/docs/math/probability/sampling) - 분포에서 값 추출
