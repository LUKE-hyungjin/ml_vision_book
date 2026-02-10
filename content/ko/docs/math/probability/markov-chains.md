---
title: "마르코프 체인"
weight: 14
math: true
---

# 마르코프 체인 (Markov Chains)

{{% hint info %}}
**선수지식**: [확률의 기초](/ko/docs/math/probability/basics) | [확률 변수](/ko/docs/math/probability/random-variable)
{{% /hint %}}

## 한 줄 요약

> **"미래는 현재에만 의존하고, 과거에는 의존하지 않는다" — 상태 간 확률적 전이 과정**

---

## 왜 마르코프 체인을 배워야 하나요?

### 문제 상황 1: Diffusion 모델은 어떻게 이미지를 생성하나요?

```python
# DDPM: 순차적으로 노이즈를 제거
for t in reversed(range(T)):
    x = denoise_step(x, t)  # x_t → x_{t-1}
```

**정답**: Diffusion은 **마르코프 체인**입니다!
- $x_t$에서 $x_{t-1}$을 만들 때, $x_t$만 보면 됨
- $x_{t+1}, x_{t+2}, \ldots$는 필요 없음

### 문제 상황 2: MCMC 샘플링이 뭔가요?

```python
# MCMC: Markov Chain Monte Carlo
samples = []
x = initial_state
for _ in range(10000):
    x = transition(x)  # 마르코프 전이
    samples.append(x)
```

**정답**: 마르코프 체인을 이용해 **복잡한 분포에서 샘플링**하는 방법!

### 문제 상황 3: PageRank는 어떻게 작동하나요?

```
웹페이지 A → B → C → A → B → ...
```

**정답**: 웹 서핑을 **마르코프 체인으로 모델링**하여 페이지 중요도를 계산!

---

## 1. 마르코프 성질 (Markov Property)

### 핵심 정의

$$
P(X_{t+1} | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} | X_t)
$$

**해석**: "다음 상태는 **현재 상태**에만 의존한다"

### 직관적 이해

```
마르코프 성질이 있는 예:
  보드 게임 → 현재 위치만 알면 다음 이동 가능
  날씨 모델 → 오늘 날씨만 알면 내일 날씨 예측 가능

마르코프 성질이 없는 예:
  주식 가격 → 과거 추세도 중요
  자연어 → "나는 오늘 ___" (앞의 문맥이 중요)
```

### 왜 유용한가?

```
전체 히스토리 저장:   O(T) 메모리, 계산 복잡
마르코프 가정:        O(1) 메모리, 계산 단순!

P(X_0, X_1, ..., X_T) = P(X_0) × P(X_1|X_0) × P(X_2|X_1) × ... × P(X_T|X_{T-1})
                         ─────    ──────────    ──────────         ──────────────
                         초기     전이 확률만 필요!
```

---

## 2. 전이 행렬 (Transition Matrix)

### 정의

상태가 $\{1, 2, \ldots, n\}$일 때, **전이 확률**:

$$
T_{ij} = P(X_{t+1} = j \mid X_t = i)
$$

전이 행렬:

$$
T = \begin{pmatrix}
T_{11} & T_{12} & \cdots & T_{1n} \\
T_{21} & T_{22} & \cdots & T_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
T_{n1} & T_{n2} & \cdots & T_{nn}
\end{pmatrix}
$$

**조건**: 각 행의 합 = 1 (어딘가로는 가야 함)

### 예시: 날씨 모델

![마르코프 체인 상태 전이](/images/probability/ko/markov-chain-weather.png)

```
상태: {맑음, 흐림, 비}

        → 맑음  흐림   비
맑음 →  [ 0.6   0.3   0.1 ]
흐림 →  [ 0.4   0.4   0.2 ]
비   →  [ 0.2   0.3   0.5 ]

해석:
- 오늘 맑으면 → 내일 맑을 확률 60%, 흐릴 확률 30%, 비 올 확률 10%
- 오늘 비면 → 내일 맑을 확률 20%, 흐릴 확률 30%, 비 올 확률 50%
```

### k 단계 후의 확률

$$
T^{(k)} = T^k
$$

**해석**: 전이 행렬을 k번 곱하면 k 단계 후의 전이 확률!

```python
import numpy as np

T = np.array([
    [0.6, 0.3, 0.1],
    [0.4, 0.4, 0.2],
    [0.2, 0.3, 0.5]
])

# 3일 후의 전이 확률
T_3 = np.linalg.matrix_power(T, 3)
print(T_3)

# 오늘 맑음일 때, 3일 후 비 올 확률
print(f"3일 후 비 확률: {T_3[0, 2]:.4f}")
```

---

## 3. 정상 분포 (Stationary Distribution)

### 정의

오랫동안 전이를 반복하면 수렴하는 분포:

$$
\boldsymbol{\pi} T = \boldsymbol{\pi}
$$

**해석**: 전이 행렬을 적용해도 **변하지 않는 분포**

### 직관적 이해

![정상 분포로의 수렴](/images/probability/ko/stationary-distribution-convergence.png)

시작 상태에 관계없이 같은 정상 분포로 수렴합니다!

### 계산 방법

$\boldsymbol{\pi} T = \boldsymbol{\pi}$를 풀면:

$$
\boldsymbol{\pi} = \text{left eigenvector of } T \text{ with eigenvalue } 1
$$

```python
# 정상 분포 계산
eigenvalues, eigenvectors = np.linalg.eig(T.T)

# 고유값 1에 해당하는 고유벡터
idx = np.argmin(np.abs(eigenvalues - 1))
pi = eigenvectors[:, idx].real
pi = pi / pi.sum()  # 정규화

print(f"정상 분포: 맑음={pi[0]:.3f}, 흐림={pi[1]:.3f}, 비={pi[2]:.3f}")
```

### 왜 중요한가?

- **MCMC**: 목표 분포 = 마르코프 체인의 정상 분포
- **PageRank**: 정상 분포 = 페이지 중요도
- **Diffusion**: 정상 분포 = $\mathcal{N}(0, I)$ (순수 노이즈)

---

## 4. 상세 균형 조건 (Detailed Balance)

### 정의

$$
\pi_i \cdot T_{ij} = \pi_j \cdot T_{ji}
$$

**해석**: "i에서 j로 가는 흐름 = j에서 i로 오는 흐름"

### 왜 중요한가?

상세 균형을 만족하면:
1. $\boldsymbol{\pi}$가 정상 분포임이 보장됨
2. MCMC에서 이 조건을 이용해 전이 확률을 설계

---

## 5. 딥러닝에서의 활용

### 1) Diffusion 모델 = 마르코프 체인

**Forward Process** (노이즈 추가):

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \mathbf{I})
$$

**마르코프 성질**: $x_t$는 $x_{t-1}$에만 의존!

$$
q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})
$$

**Reverse Process** (노이즈 제거):

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})
$$

```python
# Diffusion = 마르코프 체인의 역과정
def ddpm_sample(model, shape, T=1000):
    x = torch.randn(shape)  # x_T ~ N(0, I) = 정상 분포

    for t in reversed(range(T)):
        # 마르코프 전이: x_t → x_{t-1}
        noise_pred = model(x, t)
        x = reverse_step(x, noise_pred, t)  # P(x_{t-1} | x_t)

    return x  # x_0 = 생성된 이미지
```

**Diffusion과 마르코프 체인의 연결:**

![Diffusion = 마르코프 체인](/images/probability/ko/diffusion-markov-chain.png)

### 2) MCMC (Markov Chain Monte Carlo)

**목표**: 복잡한 분포 $p(x)$에서 샘플링

**아이디어**: $p(x)$를 정상 분포로 하는 마르코프 체인을 만들면, 충분히 오래 돌린 후의 상태 = $p(x)$에서의 샘플!

```python
def metropolis_hastings(target_log_prob, initial, n_samples, step_size=1.0):
    """Metropolis-Hastings 알고리즘"""
    samples = [initial]
    current = initial

    for _ in range(n_samples):
        # 제안: 현재 위치 근처에서 랜덤
        proposal = current + step_size * np.random.randn(*current.shape)

        # 수락/거절
        log_alpha = target_log_prob(proposal) - target_log_prob(current)

        if np.log(np.random.rand()) < log_alpha:
            current = proposal  # 수락

        samples.append(current)

    return np.array(samples)

# 목표 분포: 2차원 가우시안 혼합
def target(x):
    return np.logaddexp(
        -0.5 * np.sum((x - np.array([2, 2]))**2),
        -0.5 * np.sum((x - np.array([-2, -2]))**2)
    )

samples = metropolis_hastings(target, np.zeros(2), 10000)
```

### 3) Hidden Markov Model (HMM)

관측할 수 없는 마르코프 체인:

```
숨겨진 상태:  S_1 → S_2 → S_3 → S_4   (마르코프 체인)
                ↓      ↓      ↓      ↓
관측값:       O_1    O_2    O_3    O_4

예: 날씨(숨겨진) → 사람 행동(관측)
    맑음 → 산책, 흐림 → 집에 있음, 비 → 우산
```

### 4) 강화학습의 MDP

마르코프 결정 과정 (Markov Decision Process):

$$
P(s_{t+1} | s_t, a_t)
$$

"다음 상태는 현재 상태와 행동에만 의존" → 마르코프 성질!

---

## 코드로 확인하기

```python
import numpy as np

# === 전이 행렬과 시뮬레이션 ===
print("=== 날씨 마르코프 체인 ===")

states = ['맑음', '흐림', '비']
T = np.array([
    [0.6, 0.3, 0.1],  # 맑음 →
    [0.4, 0.4, 0.2],  # 흐림 →
    [0.2, 0.3, 0.5],  # 비 →
])

# 시뮬레이션
def simulate_chain(T, start, n_steps):
    """마르코프 체인 시뮬레이션"""
    states_seq = [start]
    current = start
    for _ in range(n_steps):
        current = np.random.choice(len(T), p=T[current])
        states_seq.append(current)
    return states_seq

# 100일 시뮬레이션 (맑음에서 시작)
seq = simulate_chain(T, start=0, n_steps=100)
print(f"처음 10일: {[states[s] for s in seq[:10]]}")

# 빈도 계산
from collections import Counter
counts = Counter(seq)
for i, s in enumerate(states):
    print(f"{s}: {counts[i]/len(seq):.3f}")

# === 정상 분포 ===
print("\n=== 정상 분포 ===")

# 방법 1: 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(T.T)
idx = np.argmin(np.abs(eigenvalues - 1))
pi = eigenvectors[:, idx].real
pi = pi / pi.sum()
print(f"고유값 분해: {dict(zip(states, pi.round(4)))}")

# 방법 2: 반복 곱셈
dist = np.array([1.0, 0.0, 0.0])  # 맑음에서 시작
for i in range(100):
    dist = dist @ T
print(f"100번 전이: {dict(zip(states, dist.round(4)))}")

# 방법 3: 시뮬레이션 (위에서 이미 함)
sim_dist = np.array([counts[i]/len(seq) for i in range(3)])
print(f"시뮬레이션:  {dict(zip(states, sim_dist.round(4)))}")
print("→ 세 방법 모두 같은 정상 분포로 수렴!")

# === k단계 전이 확률 ===
print("\n=== k단계 전이 확률 ===")

for k in [1, 3, 10, 50]:
    T_k = np.linalg.matrix_power(T, k)
    print(f"\n{k}단계 후:")
    for i, s in enumerate(states):
        print(f"  {s} → {dict(zip(states, T_k[i].round(3)))}")

print("\n→ k가 커지면 모든 행이 정상 분포로 수렴!")

# === Diffusion과 마르코프 체인 ===
print("\n=== Diffusion = 마르코프 체인 ===")

def diffusion_forward(x, T_steps=100):
    """간단한 1D Diffusion Forward Process"""
    trajectory = [x]
    betas = np.linspace(0.01, 0.2, T_steps)

    for t in range(T_steps):
        # 마르코프 전이: x_t → x_{t+1}
        noise = np.random.randn(*x.shape)
        x = np.sqrt(1 - betas[t]) * x + np.sqrt(betas[t]) * noise
        trajectory.append(x)

    return trajectory

# 데이터에서 시작
x_0 = np.random.randn(1000) * 0.5 + 3  # 평균 3, 분산 작음
trajectory = diffusion_forward(x_0)

print(f"t=0:   평균={trajectory[0].mean():.3f}, 분산={trajectory[0].var():.3f}")
print(f"t=50:  평균={trajectory[50].mean():.3f}, 분산={trajectory[50].var():.3f}")
print(f"t=100: 평균={trajectory[100].mean():.3f}, 분산={trajectory[100].var():.3f}")
print("→ 정상 분포 N(0, 1)로 수렴!")

# === Metropolis-Hastings ===
print("\n=== MCMC (Metropolis-Hastings) ===")

def target_pdf(x):
    """목표 분포: 두 가우시안 혼합"""
    return 0.5 * np.exp(-0.5 * (x - 2)**2) + 0.5 * np.exp(-0.5 * (x + 2)**2)

def mh_sample(n_samples, step_size=1.0):
    samples = []
    x = 0.0
    for _ in range(n_samples):
        x_new = x + step_size * np.random.randn()
        alpha = target_pdf(x_new) / (target_pdf(x) + 1e-10)
        if np.random.rand() < alpha:
            x = x_new
        samples.append(x)
    return np.array(samples)

samples = mh_sample(10000)
print(f"MCMC 샘플 평균: {samples[1000:].mean():.3f} (기대: 0)")
print(f"MCMC 샘플 분산: {samples[1000:].var():.3f}")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 활용 |
|------|------|-------------|
| **마르코프 성질** | $P(X_{t+1}\|X_t, \ldots) = P(X_{t+1}\|X_t)$ | Diffusion, RNN |
| **전이 행렬** | $T_{ij} = P(j\|i)$ | 상태 전이 모델링 |
| **정상 분포** | $\boldsymbol{\pi}T = \boldsymbol{\pi}$ | Diffusion의 N(0,I) |
| **MCMC** | 마르코프 체인으로 샘플링 | 베이즈 추론 |

---

## 핵심 통찰

```
1. 마르코프 성질 = "과거를 잊는다"
   - 현재만 알면 미래 예측 가능
   - 계산을 극적으로 단순화

2. 전이 행렬 = "규칙표"
   - 각 상태에서 다음 상태로 갈 확률
   - 행렬 곱으로 여러 단계 계산

3. 정상 분포 = "장기적 행동"
   - 충분히 오래 돌리면 수렴
   - 시작 상태에 관계없이 같은 결과

4. 딥러닝의 핵심!
   - Diffusion = 마르코프 체인 (forward & reverse)
   - MCMC = 복잡한 분포에서 샘플링
   - MDP = 강화학습의 기초
```

---

## 관련 콘텐츠

- [확률의 기초](/ko/docs/math/probability/basics) - 선수 지식
- [확률 변수](/ko/docs/math/probability/random-variable) - 선수 지식
- [확률분포](/ko/docs/math/probability/distribution) - 전이 확률의 분포
- [샘플링](/ko/docs/math/probability/sampling) - MCMC 활용
- [DDPM (수학)](/ko/docs/components/generative/ddpm) - Diffusion의 마르코프 체인
- [DDPM (아키텍처)](/ko/docs/architecture/generative/ddpm) - Diffusion 모델
- [고유값 분해](/ko/docs/math/linear-algebra/eigenvalue) - 정상 분포 계산
