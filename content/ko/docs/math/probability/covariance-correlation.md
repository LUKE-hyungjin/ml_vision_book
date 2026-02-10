---
title: "공분산과 상관계수"
weight: 10
math: true
---

# 공분산과 상관계수 (Covariance & Correlation)

{{% hint info %}}
**선수지식**: [기댓값과 분산](/ko/docs/math/probability/expectation) | [확률 변수](/ko/docs/math/probability/random-variable)
{{% /hint %}}

## 한 줄 요약

> **공분산 = "두 변수가 함께 변하는 정도"** | **상관계수 = "공분산을 -1~1로 정규화"**

---

## 왜 공분산과 상관계수를 배워야 하나요?

### 문제 상황 1: PCA가 뭘 하는 건가요?

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
features = pca.fit_transform(features)  # 이게 뭘 하는 거지?
```

**정답**: 특성 간 **상관관계를 제거**하고, 가장 많이 퍼진 방향만 남깁니다!
- 공분산 행렬을 분석해서 → 주요 방향을 찾음

### 문제 상황 2: Whitening은 왜 하나요?

```python
# 데이터 전처리
mean = X.mean(axis=0)
X_centered = X - mean
cov = np.cov(X_centered.T)  # 공분산 행렬?
```

**정답**: 특성들의 **공분산을 0으로** 만들어 학습을 쉽게 합니다!

### 문제 상황 3: 두 특성이 "관련 있다"를 어떻게 수치화하나요?

```python
# 이미지에서 추출한 두 특성
brightness = [...]  # 밝기
contrast = [...]    # 대비

# 이 둘이 관련 있는지 어떻게 알지?
```

**정답**: **상관계수**로 측정!
- 1에 가까우면 → 함께 증가
- -1에 가까우면 → 하나가 증가하면 다른 하나가 감소
- 0이면 → 관계 없음

---

## 1. 공분산 (Covariance)

### 직관적 정의

> **공분산** = "X가 평균보다 클 때, Y도 평균보다 큰 경향이 있는가?"

### 시각적 이해

![공분산의 부호와 산점도](/images/probability/ko/covariance-scatter.png)

### 수학적 정의

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

**각 기호의 의미:**
- $X, Y$ : 두 확률 변수
- $\mathbb{E}[X]$ : X의 평균
- $(X - \mathbb{E}[X])$ : "X가 평균에서 얼마나 벗어났나"

### 계산에 편리한 공식

$$
\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

**해석**: "곱의 평균 - 평균의 곱"

### 예시: 키와 몸무게

```
학생    키(X)    몸무게(Y)
A       160      50
B       170      65
C       175      70
D       180      75
E       165      55

E[X] = 170, E[Y] = 63

Cov = E[(X-170)(Y-63)]
    = [(-10)(-13) + (0)(2) + (5)(7) + (10)(12) + (-5)(-8)] / 5
    = [130 + 0 + 35 + 120 + 40] / 5
    = 65

Cov > 0 → 키가 크면 몸무게도 큰 경향!
```

### 공분산의 중요한 성질

| 성질 | 수식 | 의미 |
|------|------|------|
| **대칭** | $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ | 순서 무관 |
| **자기 공분산** | $\text{Cov}(X, X) = \text{Var}(X)$ | 자기 자신과의 공분산 = 분산 |
| **독립이면** | $\text{Cov}(X, Y) = 0$ | 독립 → 무상관 |
| **선형성** | $\text{Cov}(aX+b, Y) = a \cdot \text{Cov}(X, Y)$ | 상수 곱만 영향 |
| **합의 분산** | $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ | 공분산이 분산에 영향 |

{{% hint warning %}}
**주의**: 무상관 ≠ 독립!
- 독립이면 반드시 무상관 (O)
- 무상관이면 반드시 독립 (X)

예: $X \sim \text{Uniform}(-1, 1)$, $Y = X^2$이면
- $\text{Cov}(X, Y) = 0$ (무상관)
- 하지만 Y는 X에 완전히 종속됨!
{{% /hint %}}

---

## 2. 상관계수 (Correlation Coefficient)

### 왜 필요한가?

공분산의 문제: **단위에 의존**합니다.

```
키(cm)와 몸무게(kg): Cov = 65
키(m)와 몸무게(kg): Cov = 0.65

같은 관계인데 숫자가 다름! → 비교 불가
```

### 정의

$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

**각 기호의 의미:**
- $\rho_{XY}$ : X와 Y의 상관계수 (-1 ~ 1)
- $\sigma_X$ : X의 표준편차
- $\sigma_Y$ : Y의 표준편차

### 해석

```
ρ =  1  : 완벽한 양의 상관 (직선: Y = aX + b, a > 0)
ρ =  0  : 무상관 (선형 관계 없음)
ρ = -1  : 완벽한 음의 상관 (직선: Y = aX + b, a < 0)

|ρ| > 0.7  : 강한 상관
|ρ| 0.3~0.7: 중간 상관
|ρ| < 0.3  : 약한 상관
```

### 상관계수의 핵심 성질

| 성질 | 수식 | 의미 |
|------|------|------|
| **범위** | $-1 \leq \rho \leq 1$ | 항상 이 범위 |
| **무차원** | 단위 없음 | cm든 m이든 같은 값 |
| **선형 관계** | $\rho = \pm 1$ ⟺ 완벽한 직선 | 비선형은 못 잡음 |

---

## 3. 공분산 행렬 (Covariance Matrix)

### 왜 필요한가?

변수가 2개면 공분산 하나로 충분하지만, **d개 변수**가 있으면?

$$
\Sigma = \begin{pmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_d) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_d) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_d, X_1) & \text{Cov}(X_d, X_2) & \cdots & \text{Var}(X_d)
\end{pmatrix}
$$

**핵심 포인트:**
- 대각선: 각 변수의 **분산**
- 비대각선: 변수 쌍의 **공분산**
- 대칭 행렬: $\Sigma_{ij} = \Sigma_{ji}$

### 예시: 2차원

$$
\Sigma = \begin{pmatrix}
\sigma_X^2 & \text{Cov}(X,Y) \\
\text{Cov}(X,Y) & \sigma_Y^2
\end{pmatrix}
$$

![공분산 행렬과 분포 모양](/images/probability/ko/covariance-matrix-shapes.png)

### 공분산 행렬의 중요한 성질

| 성질 | 의미 |
|------|------|
| **대칭** | $\Sigma = \Sigma^T$ |
| **양반정치** | 모든 고유값 $\geq 0$ |
| **대각선 = 분산** | $\Sigma_{ii} = \text{Var}(X_i)$ |

---

## 4. 딥러닝에서의 활용

### 1) PCA (Principal Component Analysis)

**목표**: 공분산 행렬의 **고유벡터** 방향으로 데이터를 회전

```python
import numpy as np

# 데이터 (N x d)
X = np.random.randn(1000, 10)

# 1. 중심화
X_centered = X - X.mean(axis=0)

# 2. 공분산 행렬 계산
cov_matrix = np.cov(X_centered.T)  # (d x d)

# 3. 고유값 분해
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 4. 상위 k개 성분 선택
k = 3
top_k = eigenvectors[:, -k:]  # 큰 고유값 순

# 5. 투영
X_pca = X_centered @ top_k  # (N x k)
```

**공분산과의 연결**:
- 고유값 = 해당 방향의 **분산**
- 고유벡터 = 데이터가 **가장 많이 퍼진 방향**
- PCA 후 → 변환된 변수들 간 **공분산 = 0**

### 2) Whitening (ZCA Whitening)

**목표**: 공분산 행렬을 **단위 행렬**로 변환

$$
X_{white} = \Sigma^{-1/2} (X - \mu)
$$

```python
# Whitening 후
# Cov(X_white) = I (단위 행렬)
# → 모든 변수가 분산 1, 상호 공분산 0
```

**효과**: 학습이 빨라짐 (모든 방향으로 균등하게 최적화)

### 3) Batch Normalization의 한계

BatchNorm은 각 채널을 **독립적으로** 정규화:

```python
# BatchNorm: 각 채널의 평균=0, 분산=1
nn.BatchNorm2d(64)  # 64개 채널 각각 정규화

# 하지만 채널 간 공분산은 건드리지 않음!
# → Decorrelated Batch Normalization이 이를 해결
```

### 4) Contrastive Learning에서 상관관계

```python
# Barlow Twins의 핵심 아이디어
# 같은 이미지의 두 augmentation → 상관계수 = 1 (대각선)
# 다른 특성끼리 → 상관계수 = 0 (비대각선)

def barlow_twins_loss(z1, z2, lambda_param=0.005):
    # 정규화
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)

    # 상관 행렬 계산
    N = z1.shape[0]
    c = z1_norm.T @ z2_norm / N  # (d x d) 상관 행렬

    # 대각선 = 1, 비대각선 = 0 이 되도록
    on_diag = ((1 - c.diagonal()) ** 2).sum()
    off_diag = (c ** 2).fill_diagonal_(0).sum()

    return on_diag + lambda_param * off_diag
```

---

## 코드로 확인하기

```python
import numpy as np
import torch

# === 공분산 계산 ===
print("=== 공분산 ===")
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# 수동 계산
mean_X, mean_Y = X.mean(), Y.mean()
cov_manual = np.mean((X - mean_X) * (Y - mean_Y))
print(f"수동 계산 Cov(X,Y) = {cov_manual:.2f}")

# NumPy (ddof=0: 모집단 공분산)
cov_matrix = np.cov(X, Y, ddof=0)
print(f"NumPy Cov(X,Y) = {cov_matrix[0,1]:.2f}")

# === 상관계수 ===
print("\n=== 상관계수 ===")
corr = np.corrcoef(X, Y)[0, 1]
print(f"상관계수 ρ = {corr:.4f}")

# 수동 계산
rho = cov_manual / (X.std() * Y.std())
print(f"수동 계산 ρ = {rho:.4f}")

# === 공분산 행렬 ===
print("\n=== 공분산 행렬 ===")
data = np.random.randn(100, 3)  # 100개 샘플, 3개 변수
cov_mat = np.cov(data.T)
print(f"공분산 행렬 shape: {cov_mat.shape}")
print(f"대각선 (분산): {np.diag(cov_mat)}")
print(f"대칭 확인: {np.allclose(cov_mat, cov_mat.T)}")

# === PCA ===
print("\n=== PCA ===")
# 상관관계가 있는 데이터 생성
X1 = np.random.randn(500)
X2 = 0.8 * X1 + 0.2 * np.random.randn(500)  # X1과 상관관계

data = np.stack([X1, X2], axis=1)
print(f"PCA 전 공분산:\n{np.cov(data.T).round(3)}")

# PCA 적용
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
print(f"PCA 후 공분산:\n{np.cov(data_pca.T).round(3)}")
print("→ 비대각선이 ≈ 0 (상관관계 제거됨)")

# === 무상관 ≠ 독립 ===
print("\n=== 무상관 ≠ 독립 ===")
X = np.random.uniform(-1, 1, 10000)
Y = X ** 2  # Y는 X에 완전 종속

print(f"Cov(X, X²) = {np.cov(X, Y)[0,1]:.4f}")  # ≈ 0
print(f"Corr(X, X²) = {np.corrcoef(X, Y)[0,1]:.4f}")  # ≈ 0
print("→ 무상관이지만, Y=X²로 완전히 종속!")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 활용 |
|------|------|-------------|
| **공분산** | $\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ | PCA, Whitening |
| **상관계수** | $\text{Cov}(X,Y) / (\sigma_X \sigma_Y)$ | 특성 분석, Barlow Twins |
| **공분산 행렬** | $\Sigma_{ij} = \text{Cov}(X_i, X_j)$ | PCA, 다변량 가우시안 |

---

## 핵심 통찰

```
1. 공분산 = 두 변수의 "함께 변하는 정도"
   - 양수: 같은 방향으로 변함
   - 음수: 반대 방향으로 변함
   - 0: 선형 관계 없음

2. 상관계수 = 공분산의 정규화 버전
   - -1 ~ 1 범위로 비교 가능
   - 단위에 무관

3. 공분산 행렬 = 여러 변수 간의 관계를 한 행렬로
   - PCA의 핵심 재료
   - 다변량 가우시안의 모양 결정

4. 무상관 ≠ 독립! (중요!)
```

---

## 다음 단계

여러 변수의 **결합 분포**가 궁금하다면?
→ [결합/조건부 분포](/ko/docs/math/probability/joint-conditional)로!

**다변량 가우시안** 분포를 배우고 싶다면?
→ [다변량 가우시안](/ko/docs/math/probability/multivariate-gaussian)으로!

---

## 관련 콘텐츠

- [기댓값과 분산](/ko/docs/math/probability/expectation) - 선수 지식
- [확률 변수](/ko/docs/math/probability/random-variable) - 선수 지식
- [다변량 가우시안](/ko/docs/math/probability/multivariate-gaussian) - 공분산 행렬 활용
- [행렬](/ko/docs/math/linear-algebra/matrix) - 공분산 행렬의 수학
- [고유값 분해](/ko/docs/math/linear-algebra/eigenvalue) - PCA의 수학
