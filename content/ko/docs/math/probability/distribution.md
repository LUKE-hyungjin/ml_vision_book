---
title: "확률분포"
weight: 4
math: true
---

# 확률분포 (Probability Distributions)

{{% hint info %}}
**선수지식**: [확률 변수](/ko/docs/math/probability/random-variable) (이산/연속 구분), [기댓값과 분산](/ko/docs/math/probability/expectation)
{{% /hint %}}

## 한 줄 요약

> **"값이 어떤 패턴으로 나오는가?"를 수학적으로 표현한 것**

---

## 왜 확률분포를 배워야 하나요?

### 문제 상황 1: 이미지 생성 모델은 어떻게 "랜덤"하게 그릴까?

```python
# Diffusion 모델로 이미지 생성
noise = torch.randn(1, 3, 512, 512)  # 이 randn이 뭘까?
image = diffusion_model.generate(noise)
```

**정답**: `randn`은 **가우시안 분포**에서 값을 뽑습니다!
- 가우시안 분포가 뭔지 모르면 → Diffusion 이해 불가능

### 문제 상황 2: Softmax 출력은 왜 "확률"인가요?

```python
logits = model(image)  # [2.0, 1.0, 0.5]
probs = softmax(logits)  # [0.7, 0.2, 0.1] ← 이게 왜 확률?
```

**정답**: Softmax는 숫자를 **카테고리 분포**로 변환합니다!
- 합이 1이 되고, 모두 양수 → 확률의 조건 만족

### 문제 상황 3: Dropout은 어떻게 "50%"를 구현하나요?

```python
nn.Dropout(p=0.5)  # 50% 확률로 뉴런을 끔
```

**정답**: 각 뉴런마다 **베르누이 분포**로 0 또는 1을 결정합니다!

---

## 주요 분포 시각화

![주요 확률 분포](/images/probability/ko/distributions.jpeg)

---

## 1. 이산 분포 = 셀 수 있는 결과

### 베르누이 분포 (Bernoulli) - 동전 던지기

> "성공(1) 또는 실패(0), 둘 중 하나"

$$
P(X=1) = p, \quad P(X=0) = 1-p
$$

**시각화**:
```
p = 0.3일 때

P(X)
  │
0.7├────█
0.3├────     █
  └────┴─────┴───→ X
       0     1
```

**딥러닝 적용: Dropout**

```python
# Dropout의 내부 동작
p_drop = 0.5  # 끌 확률
keep_prob = 1 - p_drop

# 각 뉴런마다 베르누이 분포로 결정
mask = torch.bernoulli(torch.full((100,), keep_prob))
# mask = [1, 0, 1, 1, 0, 1, ...]  (50%가 1, 50%가 0)

output = x * mask / keep_prob  # 살아남은 것만 스케일 조정
```

---

### 카테고리 분포 (Categorical) - 주사위 던지기

> "K개 중 하나를 선택"

$$
P(X=k) = p_k, \quad \sum_{k=1}^K p_k = 1
$$

**시각화**:
```
3개 클래스 (고양이, 강아지, 새)

P(X)
  │
0.7├───█
0.2├───     █
0.1├───          █
  └───┴────┴────┴───→ X
      고양이  강아지  새
```

**딥러닝 적용: 분류 모델의 출력**

```python
# 모델 출력 → Softmax → 카테고리 분포
logits = model(image)  # [2.0, 1.0, 0.5]
probs = softmax(logits)  # [0.7, 0.2, 0.1]

# 이제 probs는 카테고리 분포의 파라미터!
# P(고양이) = 0.7
# P(강아지) = 0.2
# P(새) = 0.1

# 확률적 예측 (샘플링)
pred = torch.multinomial(probs, num_samples=1)  # [0] (고양이)

# 결정적 예측 (가장 높은 확률)
pred = probs.argmax()  # 0 (고양이)
```

---

### 이항 분포 (Binomial) - 동전 n번 던지기

> "n번 시도해서 성공한 횟수"

$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

**예시**: 동전을 10번 던졌을 때 앞면이 k번 나올 확률

```
n=10, p=0.5일 때

P(X)
  │         ╭──╮
  │       ╭─╯  ╰─╮
  │     ╭─╯      ╰─╮
  │   ╭─╯          ╰─╮
  └───┴──┴──┴──┴──┴──┴──┴──→ k
      0  1  2  3  4  5  6  7  8  9  10
                  ↑
              가장 높음 (5번)
```

**딥러닝 적용**: Dropout에서 살아남는 뉴런 수의 분포

---

## 2. 연속 분포 = 무한히 많은 결과

### 균등 분포 (Uniform) - 가장 단순한 분포

> "모든 값이 똑같은 확률"

$$
f(x) = \frac{1}{b-a}, \quad a \leq x \leq b
$$

**시각화**:
```
[0, 1] 균등 분포

f(x)
  │████████████
1 ├████████████
  │████████████
0 └────────────┴───→ x
  0            1
```

**딥러닝 적용: 가중치 초기화**

```python
# Xavier uniform 초기화
nn.init.uniform_(weight, -limit, limit)

# 내부 동작
weight = torch.empty(100, 100)
weight.uniform_(-0.1, 0.1)  # [-0.1, 0.1] 균등 분포
```

---

### 가우시안 분포 (Gaussian/Normal) - 가장 중요한 분포!

> "평균 주변에 값이 몰려 있는 종 모양"

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**왜 가장 중요한가?**
1. **자연계의 많은 현상**이 가우시안을 따름 (키, 측정 오차 등)
2. **중심극한정리**: 무엇을 많이 더하면 가우시안이 됨
3. **딥러닝 어디서나** 등장

**시각화**:
```
표준정규분포 N(0, 1)

f(x)
  │       ╭───╮
  │      ╱     ╲
  │     ╱       ╲
  │   ╱           ╲
  │  ╱             ╲
──┴─╱───────────────╲──→ x
   -3  -2  -1  0  1  2  3
           ↑
          평균

68-95-99.7 규칙:
- 68%가 μ±1σ 안에
- 95%가 μ±2σ 안에
- 99.7%가 μ±3σ 안에
```

**딥러닝 적용 1: 가중치 초기화**

```python
# Kaiming normal 초기화
nn.init.kaiming_normal_(weight)

# 내부 동작: N(0, std^2)에서 샘플링
weight = torch.empty(100, 100)
weight.normal_(mean=0, std=0.02)
```

**딥러닝 적용 2: VAE 잠재 공간**

```python
# Encoder가 평균과 분산을 예측
mu = encoder_mu(x)       # μ
log_var = encoder_var(x)  # log(σ²)

# Reparameterization Trick: z = μ + σ × ε
std = torch.exp(0.5 * log_var)  # σ = √(σ²)
eps = torch.randn_like(std)      # ε ~ N(0, 1)
z = mu + std * eps               # z ~ N(μ, σ²)
```

**딥러닝 적용 3: Diffusion 노이즈**

```python
# 이미지에 가우시안 노이즈 추가
noise = torch.randn_like(image)  # N(0, 1)
noisy_image = image + noise_level * noise
```

---

### 다변량 가우시안 (Multivariate Gaussian)

> "여러 변수가 함께 가우시안을 따를 때"

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

**파라미터**:
- $\boldsymbol{\mu}$: 평균 벡터 (중심점)
- $\boldsymbol{\Sigma}$: 공분산 행렬 (퍼짐과 기울기)

**시각화** (2D):
```
등고선으로 표현

 y │    ╭───╮
   │  ╭─╯   ╰─╮
   │ ╭╯   ★   ╰╮   ★ = 평균
   │ ╰╮       ╭╯
   │  ╰─╮   ╭─╯
   │    ╰───╯
   └──────────────→ x

공분산이 크면 → 더 넓게 퍼짐
공분산이 대각선이면 → 타원이 기울어짐
```

**딥러닝 적용: VAE 잠재 공간**

```python
# VAE는 보통 대각 공분산 (각 차원이 독립)을 가정
# z ~ N(μ, diag(σ²))

mu = encoder_mu(x)        # [batch, latent_dim]
log_var = encoder_var(x)  # [batch, latent_dim]

# 각 차원 독립적으로 샘플링
z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
```

---

## 3. Softmax = 숫자 → 확률분포 변환기

### 왜 Softmax인가?

**문제**: 모델 출력이 `[2.0, 1.0, 0.5]`입니다. 이건 확률이 아닙니다!
- 합이 1이 아님
- 음수가 될 수도 있음

**해결**: Softmax로 변환!

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

```
입력 (logits)           Softmax            출력 (확률)
   [2.0]                   │               [0.659]
   [1.0]      ───→        exp + 정규화  ───→   [0.242]
   [0.5]                   │               [0.099]
                                            합 = 1.0 ✓
```

### Softmax의 특성

| 특성 | 설명 | 수식 |
|------|------|------|
| **양수** | 모든 출력 > 0 | $e^x > 0$ |
| **합 = 1** | 확률 조건 만족 | $\sum_i p_i = 1$ |
| **순서 보존** | 큰 logit → 큰 확률 | $z_i > z_j \Rightarrow p_i > p_j$ |
| **차이 강조** | 큰 값이 더 커짐 | 지수함수 특성 |

### 수치 안정성

**문제**: $e^{1000}$은 overflow!

**해결**: 최댓값을 빼도 결과는 같음

$$
\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

```python
def softmax_stable(z):
    z_max = z.max(dim=-1, keepdim=True).values
    exp_z = torch.exp(z - z_max)  # overflow 방지
    return exp_z / exp_z.sum(dim=-1, keepdim=True)

# PyTorch는 이미 안정적으로 구현되어 있음
probs = torch.softmax(logits, dim=-1)
```

### Temperature로 "확신도" 조절

$$
\text{softmax}(z/T)_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

![Temperature Scaling](/images/probability/ko/sampling-temperature.png)

```python
def softmax_with_temperature(logits, temperature=1.0):
    return torch.softmax(logits / temperature, dim=-1)

# T < 1: 더 확신 있게 (가장 높은 값이 더 높아짐)
# T > 1: 더 불확실하게 (값들이 비슷해짐)
# T → 0: argmax와 같아짐
# T → ∞: 균등 분포가 됨
```

**딥러닝 적용**: 텍스트 생성에서 창의성 조절

```python
# T 낮음: 예측 가능한 응답
# T 높음: 창의적인 응답
probs = softmax_with_temperature(logits, temperature=0.7)
```

---

## 4. 분포 간 관계

![분포 간 관계](/images/probability/ko/distribution-relationship.jpeg)

**핵심 관계**:
1. **베르누이 → 이항**: n번 시행의 성공 횟수
2. **이항 → 정규**: n이 커지면 정규분포에 근사 (중심극한정리)
3. **베르누이 → 카테고리**: 2개 → K개로 확장

---

## 코드로 확인하기

```python
import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# === 베르누이 분포 ===
print("=== 베르누이 분포 (Dropout) ===")
p_keep = 0.7
mask = torch.bernoulli(torch.full((1000,), p_keep))
print(f"살아남은 비율: {mask.mean():.3f} (기대값: {p_keep})")

# === 카테고리 분포 (Softmax) ===
print("\n=== 카테고리 분포 (분류) ===")
logits = torch.tensor([2.0, 1.0, 0.5])
probs = torch.softmax(logits, dim=0)
print(f"Softmax 출력: {probs.tolist()}")
print(f"합: {probs.sum():.1f}")

# 샘플링
samples = torch.multinomial(probs, num_samples=1000, replacement=True)
print(f"1000번 샘플링 - 클래스 0: {(samples==0).sum()}, 1: {(samples==1).sum()}, 2: {(samples==2).sum()}")

# === 가우시안 분포 ===
print("\n=== 가우시안 분포 ===")
samples = torch.randn(10000)  # N(0, 1)
print(f"평균: {samples.mean():.4f} (기대값: 0)")
print(f"표준편차: {samples.std():.4f} (기대값: 1)")

# 68-95-99.7 규칙 확인
within_1std = ((samples > -1) & (samples < 1)).float().mean()
within_2std = ((samples > -2) & (samples < 2)).float().mean()
print(f"±1σ 안에 있는 비율: {within_1std:.1%} (기대값: 68%)")
print(f"±2σ 안에 있는 비율: {within_2std:.1%} (기대값: 95%)")

# === Temperature ===
print("\n=== Temperature Scaling ===")
logits = torch.tensor([3.0, 1.0, 0.5])
for T in [0.5, 1.0, 2.0]:
    probs = torch.softmax(logits / T, dim=0)
    print(f"T={T}: {[f'{p:.3f}' for p in probs.tolist()]}")

# === VAE Reparameterization ===
print("\n=== VAE Reparameterization Trick ===")
mu = torch.tensor([1.0, 2.0])
log_var = torch.tensor([0.0, 0.5])  # σ² = 1, σ² ≈ 1.65

std = torch.exp(0.5 * log_var)
eps = torch.randn_like(std)
z = mu + std * eps

print(f"mu: {mu.tolist()}")
print(f"std: {std.tolist()}")
print(f"sampled z: {z.tolist()}")
```

---

## 주요 분포 정리표

| 분포 | 타입 | 파라미터 | 기댓값 | 분산 | 딥러닝 적용 |
|------|------|----------|--------|------|-------------|
| **베르누이** | 이산 | $p$ | $p$ | $p(1-p)$ | Dropout |
| **카테고리** | 이산 | $p_1,...,p_K$ | - | - | Softmax 출력 |
| **이항** | 이산 | $n, p$ | $np$ | $np(1-p)$ | - |
| **균등** | 연속 | $a, b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Xavier 초기화 |
| **가우시안** | 연속 | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | VAE, Diffusion, He 초기화 |

---

## 핵심 정리

| 개념 | 핵심 | 딥러닝 예시 |
|------|------|-------------|
| **베르누이** | 0 또는 1 | Dropout 마스크 |
| **카테고리** | K개 중 하나 | 분류 출력 |
| **가우시안** | 종 모양, 중심에 몰림 | VAE, Diffusion, 초기화 |
| **Softmax** | 숫자 → 확률 변환 | 분류 마지막 층 |
| **Temperature** | 확신도 조절 | 텍스트 생성 |

---

## 다음 단계

확률을 **역으로 계산**하고 싶다면?
→ [베이즈 정리](/ko/docs/math/probability/bayes)로!

분포에서 **값을 뽑는** 다양한 방법이 궁금하다면?
→ [샘플링](/ko/docs/math/probability/sampling)으로!

---

## 관련 콘텐츠

- [확률 변수](/ko/docs/math/probability/random-variable) - 이산/연속 구분
- [기댓값과 분산](/ko/docs/math/probability/expectation) - 분포의 특성
- [베이즈 정리](/ko/docs/math/probability/bayes) - 조건부 확률 역산
- [샘플링](/ko/docs/math/probability/sampling) - 분포에서 값 추출
- [VAE](/ko/docs/architecture/generative/vae) - 가우시안 활용
- [Cross-Entropy Loss](/ko/docs/components/training/loss/cross-entropy) - 분포 간 거리
