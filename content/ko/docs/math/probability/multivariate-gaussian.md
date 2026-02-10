---
title: "다변량 가우시안"
weight: 12
math: true
---

# 다변량 가우시안 분포 (Multivariate Gaussian)

{{% hint info %}}
**선수지식**: [확률분포](/ko/docs/math/probability/distribution) | [공분산과 상관계수](/ko/docs/math/probability/covariance-correlation) | [행렬](/ko/docs/math/linear-algebra/matrix)
{{% /hint %}}

## 한 줄 요약

> **1차원 가우시안의 고차원 확장 — 평균 벡터와 공분산 행렬로 "타원형" 분포를 표현**

---

## 왜 다변량 가우시안을 배워야 하나요?

### 문제 상황 1: VAE의 잠재 공간은 어떤 모양인가요?

```python
# VAE 인코더
mu, log_var = encoder(image)  # 64차원 벡터 2개

# 잠재 벡터 샘플링
z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
```

**정답**: VAE의 잠재 공간은 **다변량 가우시안**입니다!
- $\mu$: 평균 벡터 (분포의 중심)
- $\sigma^2$: 분산 벡터 (각 차원의 퍼짐)
- KL Loss가 이를 $\mathcal{N}(\mathbf{0}, \mathbf{I})$에 가깝게 만듦

### 문제 상황 2: Diffusion 모델의 노이즈는 왜 가우시안인가요?

```python
# DDPM: 가우시안 노이즈 추가
noise = torch.randn_like(image)  # 이건 뭘까?
```

**정답**: `torch.randn`은 **표준 다변량 가우시안** $\mathcal{N}(\mathbf{0}, \mathbf{I})$에서 샘플링!
- 각 픽셀이 독립적으로 N(0,1)에서 뽑힘

### 문제 상황 3: 가중치 초기화의 "정규분포"란?

```python
nn.init.normal_(layer.weight, mean=0, std=0.02)
# 이것도 다변량 가우시안?
```

**정답**: 네! 모든 가중치가 **독립적인** 가우시안에서 초기화됩니다.
- 공분산 행렬 = $0.02^2 \cdot \mathbf{I}$ (대각 행렬)

---

## 1. 1차원에서 다차원으로

### 1차원 가우시안 (복습)

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- $\mu$: 평균 (중심)
- $\sigma^2$: 분산 (퍼짐)

### 다변량 가우시안 (d차원)

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
$$

**각 기호의 의미:**
- $\mathbf{x} \in \mathbb{R}^d$ : d차원 벡터
- $\boldsymbol{\mu} \in \mathbb{R}^d$ : 평균 벡터 (분포의 중심)
- $\Sigma \in \mathbb{R}^{d \times d}$ : 공분산 행렬 (모양과 방향)
- $|\Sigma|$ : 공분산 행렬의 행렬식 (부피 스케일)
- $\Sigma^{-1}$ : 공분산 행렬의 역행렬

![1차원에서 다변량 가우시안으로](/images/probability/ko/univariate-to-multivariate.png)

### 1차원 → 다차원 대응

| 1차원 | 다차원 | 역할 |
|-------|--------|------|
| $\mu$ (스칼라) | $\boldsymbol{\mu}$ (벡터) | 중심 위치 |
| $\sigma^2$ (스칼라) | $\Sigma$ (행렬) | 퍼짐의 정도와 방향 |
| $(x-\mu)^2 / \sigma^2$ | $(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$ | 마할라노비스 거리 |

---

## 2. 공분산 행렬이 모양을 결정한다

### 2차원 예시

$$
\boldsymbol{\mu} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

**Case 1: 단위 행렬** $\Sigma = \mathbf{I}$

$$
\Sigma = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

```
  Y
  │    ···
  │  ·······
  │ ·········
  │  ·······
  │    ···
  └──────── X
  → 원형 (모든 방향 동일)
```

**Case 2: 대각 행렬** (다른 분산)

$$
\Sigma = \begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}
$$

```
  Y
  │    ···
  │  ·······
  │···········
  │  ·······
  │    ···
  └──────────── X
  → 축 방향 타원 (X 방향이 더 넓음)
```

**Case 3: 비대각 성분 있음** (상관관계)

$$
\Sigma = \begin{pmatrix} 2 & 1.5 \\ 1.5 & 2 \end{pmatrix}
$$

```
  Y
  │      ····
  │    ····
  │  ····
  │····
  └──────── X
  → 기울어진 타원 (양의 상관관계)
```

### 핵심 통찰

| 공분산 행렬 | 모양 | 의미 |
|------------|------|------|
| $\sigma^2 \mathbf{I}$ | 원 | 모든 방향 동일, 독립 |
| 대각 행렬 | 축 방향 타원 | 방향별 다른 분산, 독립 |
| 일반 행렬 | 기울어진 타원 | 변수 간 상관관계 있음 |

---

## 3. 마할라노비스 거리

### 정의

$$
d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})}
$$

### 직관적 이해

![마할라노비스 거리 vs 유클리드 거리](/images/probability/ko/mahalanobis-vs-euclidean.png)

```
유클리드 거리: "직선 거리"
→ 퍼짐을 무시

마할라노비스 거리: "분포를 고려한 거리"
→ 분산이 큰 방향에서는 멀어도 괜찮
→ 분산이 작은 방향에서는 조금만 멀어도 이상함

예시:
  키: 평균 170cm, 표준편차 10cm
  몸무게: 평균 65kg, 표준편차 5kg

  A: 키 180cm, 몸무게 65kg → 유클리드 큼, 마할라노비스 작음 (키 편차 1σ)
  B: 키 170cm, 몸무게 80kg → 유클리드 작음, 마할라노비스 큼 (몸무게 편차 3σ)
```

### 딥러닝 활용

**이상 탐지 (Anomaly Detection)**:

```python
def mahalanobis_distance(x, mu, cov):
    """마할라노비스 거리 계산"""
    diff = x - mu
    cov_inv = np.linalg.inv(cov)
    return np.sqrt(diff @ cov_inv @ diff)

# 정상 데이터의 평균, 공분산 계산
mu = normal_features.mean(axis=0)
cov = np.cov(normal_features.T)

# 새 데이터의 이상도 측정
distance = mahalanobis_distance(new_feature, mu, cov)
if distance > threshold:
    print("이상 탐지!")
```

---

## 4. 조건부 분포와 주변 분포

### 다변량 가우시안의 강력한 성질

$\mathbf{x} = \begin{pmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{pmatrix}$이 다변량 가우시안을 따르면:

$$
\begin{pmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}\right)
$$

### 주변 분포: 가우시안!

$$
\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_{11})
$$

"일부 변수만 보면 → 여전히 가우시안!"

### 조건부 분포: 역시 가우시안!

$$
\mathbf{x}_1 | \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \Sigma_{1|2})
$$

$$
\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)
$$

$$
\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
$$

**핵심**: $\mathbf{x}_2$를 알면 → $\mathbf{x}_1$의 **평균이 조정**되고, **분산이 줄어듦**!

![조건부 가우시안](/images/probability/ko/conditional-gaussian.png)

정보를 알면 불확실성이 줄어듭니다!

---

## 5. 딥러닝에서의 활용

### 1) VAE의 잠재 공간

```python
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)           # 평균 벡터
        log_var = self.fc_logvar(x)   # log(분산) 벡터

        # 재매개변수화 트릭 (Reparameterization Trick)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)   # N(0, I)에서 샘플
        z = mu + std * eps            # N(mu, diag(sigma²))에서 샘플

        return z, mu, log_var
```

**주의**: VAE는 보통 **대각 공분산** 가정!
- $\Sigma = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_d^2)$
- 차원 간 독립 가정 → 계산 단순화

### 2) KL Divergence (가우시안 간)

$$
D_{KL}(\mathcal{N}(\boldsymbol{\mu}, \Sigma) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) = \frac{1}{2}\left(-\log|\Sigma| - d + \text{tr}(\Sigma) + \boldsymbol{\mu}^T\boldsymbol{\mu}\right)
$$

대각 공분산일 때 ($\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$):

$$
D_{KL} = \frac{1}{2}\sum_{i=1}^{d} \left(-\log\sigma_i^2 - 1 + \sigma_i^2 + \mu_i^2\right)
$$

```python
def kl_divergence_gaussian(mu, log_var):
    """D_KL(N(mu, sigma²) || N(0, I))"""
    # log_var = log(sigma²)
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

### 3) Diffusion 모델의 노이즈

```python
# Forward process: 가우시안 노이즈를 점진적으로 추가
def q_sample(x_0, t, noise=None):
    """x_0에 t 단계만큼 노이즈 추가"""
    if noise is None:
        noise = torch.randn_like(x_0)  # N(0, I)

    sqrt_alpha_bar = sqrt_alphas_cumprod[t]
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t]

    # x_t ~ N(sqrt(ᾱ_t) · x_0, (1-ᾱ_t) · I)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
    return x_t
```

### 4) Gaussian Mixture Model (GMM)

여러 가우시안의 혼합:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \Sigma_k)
$$

```python
from sklearn.mixture import GaussianMixture

# 데이터를 K개의 가우시안 혼합으로 모델링
gmm = GaussianMixture(n_components=3)
gmm.fit(features)

# 각 클러스터의 평균, 공분산
print(gmm.means_)        # 평균 벡터들
print(gmm.covariances_)  # 공분산 행렬들
```

---

## 코드로 확인하기

```python
import numpy as np
import torch

# === 2차원 가우시안 시각화 준비 ===
print("=== 2차원 다변량 가우시안 ===")

# Case 1: 표준 가우시안 (원형)
mu1 = np.array([0, 0])
cov1 = np.array([[1, 0], [0, 1]])

# Case 2: 다른 분산 (축 방향 타원)
mu2 = np.array([0, 0])
cov2 = np.array([[4, 0], [0, 1]])

# Case 3: 상관관계 (기울어진 타원)
mu3 = np.array([0, 0])
cov3 = np.array([[2, 1.5], [1.5, 2]])

# 샘플링
samples1 = np.random.multivariate_normal(mu1, cov1, 1000)
samples2 = np.random.multivariate_normal(mu2, cov2, 1000)
samples3 = np.random.multivariate_normal(mu3, cov3, 1000)

for name, samples, cov in [("원형", samples1, cov1),
                             ("축 타원", samples2, cov2),
                             ("기울어진", samples3, cov3)]:
    print(f"\n{name}: 공분산 행렬 =")
    print(cov)
    print(f"  샘플 공분산 ≈")
    print(np.cov(samples.T).round(2))

# === 마할라노비스 거리 ===
print("\n=== 마할라노비스 거리 ===")

mu = np.array([170, 65])  # 키(cm), 몸무게(kg)
cov = np.array([[100, 30], [30, 25]])  # 공분산 행렬

def mahalanobis(x, mu, cov):
    diff = x - mu
    return np.sqrt(diff @ np.linalg.inv(cov) @ diff)

# 유클리드 vs 마할라노비스
A = np.array([180, 65])   # 키만 다름
B = np.array([170, 80])   # 몸무게만 다름

euclid_A = np.linalg.norm(A - mu)
euclid_B = np.linalg.norm(B - mu)
mahal_A = mahalanobis(A, mu, cov)
mahal_B = mahalanobis(B, mu, cov)

print(f"A(키 180, 몸무게 65):")
print(f"  유클리드: {euclid_A:.2f}, 마할라노비스: {mahal_A:.2f}")
print(f"B(키 170, 몸무게 80):")
print(f"  유클리드: {euclid_B:.2f}, 마할라노비스: {mahal_B:.2f}")
print("→ 마할라노비스는 분산을 고려한 거리!")

# === VAE의 재매개변수화 트릭 ===
print("\n=== VAE 재매개변수화 트릭 ===")

mu_vae = torch.tensor([1.0, -0.5, 0.3])
log_var = torch.tensor([0.5, -0.2, 0.1])

# N(mu, sigma²)에서 샘플
std = torch.exp(0.5 * log_var)
eps = torch.randn_like(std)
z = mu_vae + std * eps

print(f"mu: {mu_vae.tolist()}")
print(f"std: {std.tolist()}")
print(f"z:   {z.tolist()}")

# KL Divergence
kl = -0.5 * torch.sum(1 + log_var - mu_vae.pow(2) - log_var.exp())
print(f"KL(q||p): {kl.item():.4f}")

# === 조건부 분포 ===
print("\n=== 조건부 분포 ===")

mu_full = np.array([0, 0])
cov_full = np.array([[1.0, 0.8],
                      [0.8, 1.0]])

# x2 = 1.5일 때 x1의 조건부 분포
x2_observed = 1.5
mu_cond = mu_full[0] + cov_full[0,1] / cov_full[1,1] * (x2_observed - mu_full[1])
var_cond = cov_full[0,0] - cov_full[0,1]**2 / cov_full[1,1]

print(f"x2 = {x2_observed}일 때:")
print(f"  x1의 조건부 평균: {mu_cond:.3f}")
print(f"  x1의 조건부 분산: {var_cond:.3f}")
print(f"  x1의 주변 분산:   {cov_full[0,0]:.3f}")
print("→ 조건부 분산 < 주변 분산 (정보가 불확실성을 줄임)")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 활용 |
|------|------|-------------|
| **다변량 가우시안** | $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ | VAE 잠재 공간, Diffusion 노이즈 |
| **공분산 행렬** | $\Sigma$ | 분포의 모양과 방향 |
| **마할라노비스 거리** | $\sqrt{(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ | 이상 탐지 |
| **조건부 가우시안** | $\mathbf{x}_1\|\mathbf{x}_2 \sim \mathcal{N}(\cdot, \cdot)$ | Gaussian Process |

---

## 핵심 통찰

```
1. 다변량 가우시안 = 1차원의 자연스러운 확장
   - 평균 → 평균 벡터
   - 분산 → 공분산 행렬

2. 공분산 행렬이 모든 것을 결정
   - 모양 (원, 타원)
   - 방향 (기울기)
   - 크기 (퍼짐)

3. 가우시안의 강력한 성질
   - 주변 분포 → 가우시안
   - 조건부 분포 → 가우시안
   - 합 → 가우시안
   - 이것이 가우시안이 인기 있는 이유!

4. 딥러닝 핵심 적용
   - VAE: 잠재 공간 = 다변량 가우시안
   - Diffusion: 노이즈 = 표준 다변량 가우시안
   - 이상 탐지: 마할라노비스 거리
```

---

## 다음 단계

**상호 정보량**으로 변수 간의 의존성을 측정하고 싶다면?
→ [상호 정보량](/ko/docs/math/probability/mutual-information)으로!

가우시안에서 **값을 뽑는 방법**이 궁금하다면?
→ [샘플링](/ko/docs/math/probability/sampling)으로!

---

## 관련 콘텐츠

- [확률분포](/ko/docs/math/probability/distribution) - 1차원 가우시안
- [공분산과 상관계수](/ko/docs/math/probability/covariance-correlation) - 공분산 행렬의 기초
- [행렬](/ko/docs/math/linear-algebra/matrix) - 행렬 연산
- [고유값 분해](/ko/docs/math/linear-algebra/eigenvalue) - 공분산 행렬 분석
- [SVD](/ko/docs/math/linear-algebra/svd) - 차원 축소
- [VAE](/ko/docs/architecture/generative/vae) - 잠재 공간
- [DDPM](/ko/docs/architecture/generative/ddpm) - 가우시안 노이즈
