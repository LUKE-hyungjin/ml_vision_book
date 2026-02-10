---
title: "고유값/고유벡터"
weight: 6
math: true
---

# 고유값과 고유벡터 (Eigenvalue & Eigenvector)

{{% hint info %}}
**선수지식**: [행렬 연산](/ko/docs/math/linear-algebra/matrix)
{{% /hint %}}

> **한 줄 요약**: 고유값/고유벡터는 **행렬이 정말 하는 일**을 보여줍니다. PCA, 학습 안정성, 모델 압축의 핵심입니다.

## 왜 고유값을 배워야 하나요?

### 문제 상황 1: "PCA가 뭘 하는 건가요?"

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)  # 784차원 → 50차원
```

→ PCA는 **공분산 행렬의 고유벡터** 방향으로 데이터를 투영합니다.

### 문제 상황 2: "왜 학습이 불안정한가요?"

```python
# Loss가 진동하거나 발산함
for epoch in range(100):
    loss = train_step()
    print(f"Epoch {epoch}: Loss = {loss}")  # 왜 수렴 안 하지?
```

→ **Hessian의 고유값**이 너무 크거나 범위가 넓으면 학습이 불안정합니다.

### 문제 상황 3: "Spectral Normalization이 뭔가요?"

```python
# GAN에서 안정성을 위해 사용
conv = SpectralNorm(nn.Conv2d(64, 128, 3))
```

→ 가중치 행렬의 **최대 고유값(스펙트럼 노름)**을 1로 제한합니다.

---

## 고유값/고유벡터란?

### 핵심 정의

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

- $A$: 정사각 행렬 (n × n)
- $\mathbf{v}$: **고유벡터** (eigenvector) - 0이 아닌 벡터
- $\lambda$: **고유값** (eigenvalue) - 스칼라

### 직관적 이해

**"행렬 A로 변환했을 때, 방향은 그대로이고 크기만 λ배 변하는 벡터"**

```python
import torch
import numpy as np

# 행렬 A
A = torch.tensor([[3., 1.],
                  [0., 2.]])

# 일반적인 벡터는 방향이 변함
v1 = torch.tensor([1., 1.])
Av1 = A @ v1
print(f"원래: {v1}")      # [1, 1]
print(f"변환 후: {Av1}")  # [4, 2] - 방향이 바뀜!

# 고유벡터는 방향이 유지됨
v2 = torch.tensor([1., 0.])  # A의 고유벡터
Av2 = A @ v2
print(f"원래: {v2}")      # [1, 0]
print(f"변환 후: {Av2}")  # [3, 0] - 방향 유지, 크기 3배!
# λ = 3
```

### 비유: 줄다리기 방향

행렬 A가 데이터를 "잡아당기는" 변환이라고 생각하면:
- **고유벡터** = 잡아당기는 방향 (힘이 작용하는 축)
- **고유값** = 각 방향으로 얼마나 세게 당기는지

{{< figure src="/images/math/linear-algebra/ko/eigenvalue-concept.jpeg" caption="일반 벡터는 방향이 바뀌지만, 고유벡터는 방향이 유지되고 크기만 λ배 변한다" >}}

---

## 기하학적 의미

### 행렬 변환 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

# 행렬 정의
A = np.array([[2, 1],
              [1, 2]])

# 단위원 위의 점들
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# 변환 적용
transformed = A @ circle

# 고유값/고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"고유값: {eigenvalues}")  # [3, 1]
print(f"고유벡터:\n{eigenvectors}")

# 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(circle[0], circle[1], 'b-', label='원래')
plt.plot(transformed[0], transformed[1], 'r-', label='변환 후')
plt.legend()
plt.axis('equal')
plt.title('단위원의 변환')

plt.subplot(1, 2, 2)
# 고유벡터 방향 표시
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    plt.arrow(0, 0, vec[0]*val, vec[1]*val, head_width=0.1,
              color=f'C{i}', label=f'λ={val:.1f}')
plt.legend()
plt.axis('equal')
plt.title('고유벡터 방향')
plt.show()
```

### 해석

- 원이 **타원**으로 변환됨
- 타원의 장축/단축 방향 = **고유벡터**
- 장축/단축 길이 = **고유값**

---

## 고유값 계산 방법

### 특성 방정식 (Characteristic Equation)

$$
\det(A - \lambda I) = 0
$$

**유도**:
$$
A\mathbf{v} = \lambda\mathbf{v} \\
A\mathbf{v} - \lambda\mathbf{v} = 0 \\
(A - \lambda I)\mathbf{v} = 0
$$

$\mathbf{v} \neq 0$이려면 $(A - \lambda I)$가 역행렬이 없어야 함 → $\det = 0$

### 2×2 예제

$$
A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}
$$

특성 방정식:
$$
\det\begin{bmatrix} 4-\lambda & 2 \\ 1 & 3-\lambda \end{bmatrix} = 0
$$

$$
(4-\lambda)(3-\lambda) - 2 = 0 \\
\lambda^2 - 7\lambda + 10 = 0 \\
(\lambda - 5)(\lambda - 2) = 0
$$

**고유값**: $\lambda_1 = 5$, $\lambda_2 = 2$

```python
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"고유값: {eigenvalues}")  # [5. 2.]
```

### 고유벡터 구하기

$\lambda = 5$일 때:
$$
(A - 5I)\mathbf{v} = 0 \\
\begin{bmatrix} -1 & 2 \\ 1 & -2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = 0
$$

$-v_1 + 2v_2 = 0$ → $v_1 = 2v_2$

**고유벡터**: $\mathbf{v}_1 = [2, 1]$ (정규화 가능)

---

## 고유값의 성질

### 1. 대각합 (Trace)

$$
\text{tr}(A) = \sum_i a_{ii} = \sum_i \lambda_i
$$

**고유값들의 합 = 대각선 원소들의 합**

```python
A = np.array([[4, 2], [1, 3]])
print(f"Trace: {np.trace(A)}")  # 7
print(f"고유값 합: {np.sum(np.linalg.eigvals(A))}")  # 7
```

### 2. 행렬식 (Determinant)

$$
\det(A) = \prod_i \lambda_i
$$

**고유값들의 곱 = 행렬식**

```python
print(f"Det: {np.linalg.det(A)}")  # 10
print(f"고유값 곱: {np.prod(np.linalg.eigvals(A))}")  # 10
```

### 3. 대칭 행렬의 특별한 성질

**실대칭 행렬 ($A = A^T$)**:
- 모든 고유값이 **실수**
- 고유벡터들이 **서로 직교**

```python
# 대칭 행렬
A_sym = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A_sym)
print(f"고유값: {eigenvalues}")  # [3. 1.] - 실수!

# 직교성 확인
v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]
print(f"내적: {np.dot(v1, v2):.10f}")  # ≈ 0 (직교)
```

**딥러닝 적용**: 공분산 행렬, Hessian 모두 대칭행렬!

---

## 딥러닝에서의 적용

### 1. PCA (Principal Component Analysis)

**목표**: 고차원 데이터를 저차원으로 압축하되, 정보 손실 최소화

**원리**: 공분산 행렬의 고유벡터 방향으로 투영

```python
import numpy as np

def pca_from_scratch(X, n_components):
    # 1. 평균 제거 (중심화)
    X_centered = X - X.mean(axis=0)

    # 2. 공분산 행렬 계산
    cov = X_centered.T @ X_centered / (len(X) - 1)

    # 3. 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh는 대칭행렬용

    # 4. 고유값 큰 순서로 정렬
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 상위 k개 고유벡터 선택
    W = eigenvectors[:, :n_components]

    # 6. 투영
    X_reduced = X_centered @ W

    # 설명된 분산 비율
    explained_variance = eigenvalues[:n_components] / eigenvalues.sum()

    return X_reduced, W, explained_variance

# 사용 예시
X = np.random.randn(100, 50)  # 50차원 데이터 100개
X_reduced, W, var_ratio = pca_from_scratch(X, n_components=5)
print(f"원본: {X.shape}")      # (100, 50)
print(f"압축: {X_reduced.shape}")  # (100, 5)
print(f"설명된 분산: {var_ratio.sum():.2%}")  # e.g., 45%
```

**직관**: 고유값이 큰 방향 = 데이터가 많이 퍼져있는 방향 = 정보가 많은 방향

### 2. 학습 안정성 분석 (Hessian의 고유값)

**Hessian 행렬**: Loss 함수의 2차 미분

$$
H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}
$$

**고유값의 의미**:
- **모든 고유값 > 0**: 극소점 (학습 성공 가능)
- **일부 고유값 < 0**: 안장점 (탈출 필요)
- **고유값 범위가 큼**: 학습 어려움 (conditioning 나쁨)

```python
# Condition Number = 최대 고유값 / 최소 고유값
# 이 값이 크면 학습이 어려움

def compute_condition_number(H):
    eigenvalues = np.linalg.eigvalsh(H)  # 대칭행렬
    return np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

# 예시: 잘 conditioned된 Hessian
H_good = np.array([[2, 0], [0, 1]])  # 고유값: 2, 1
print(f"Condition (good): {compute_condition_number(H_good)}")  # 2

# 예시: 나쁘게 conditioned된 Hessian
H_bad = np.array([[100, 0], [0, 0.01]])  # 고유값: 100, 0.01
print(f"Condition (bad): {compute_condition_number(H_bad)}")  # 10000
```

**Learning Rate와의 관계**:
$$
\eta < \frac{2}{\lambda_{\max}}
$$

최대 고유값이 크면 작은 learning rate 필요!

### 3. Spectral Normalization (GAN 안정화)

**문제**: GAN의 Discriminator가 너무 강하면 학습 불안정

**해결**: 가중치 행렬의 스펙트럼 노름(최대 특이값)을 1로 제한

$$
W_{SN} = \frac{W}{\sigma(W)}
$$

여기서 $\sigma(W)$는 W의 최대 특이값

```python
import torch
import torch.nn as nn

# PyTorch의 Spectral Normalization
conv = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3))

# 내부적으로:
# 1. W의 최대 특이값 σ 계산 (power iteration으로 근사)
# 2. W = W / σ 로 정규화
# 3. 결과: 모든 입력에 대해 ||Wx|| ≤ ||x||

# 수동 구현
def spectral_norm(W, n_iterations=1):
    # Power iteration으로 최대 특이값 근사
    u = torch.randn(W.shape[0])
    for _ in range(n_iterations):
        v = W.T @ u
        v = v / torch.norm(v)
        u = W @ v
        u = u / torch.norm(u)
    sigma = u @ W @ v
    return W / sigma
```

**직관**: Lipschitz 상수를 1로 제한 → 출력 변화가 입력 변화를 넘지 않음

### 4. Power Iteration (큰 행렬의 최대 고유값)

실제로 수백만 차원의 행렬을 고유값 분해하기는 불가능. **Power Iteration**으로 최대 고유값만 근사:

```python
def power_iteration(A, num_iterations=100):
    """
    최대 고유값과 해당 고유벡터를 반복적으로 계산
    """
    # 랜덤 벡터로 시작
    b = torch.randn(A.shape[0])
    b = b / torch.norm(b)

    for _ in range(num_iterations):
        # A를 곱하면 최대 고유값 방향으로 "당겨짐"
        Ab = A @ b
        # 정규화
        b = Ab / torch.norm(Ab)

    # 최대 고유값 계산
    eigenvalue = (b @ A @ b) / (b @ b)
    return eigenvalue, b

# 테스트
A = torch.tensor([[3., 1.], [1., 3.]])
eigenvalue, eigenvector = power_iteration(A)
print(f"최대 고유값 (근사): {eigenvalue:.4f}")  # ≈ 4
print(f"실제 고유값: {torch.linalg.eigvalsh(A)}")  # [2, 4]
```

---

## 특수한 경우들

### 양의 정부호 행렬 (Positive Definite)

**모든 고유값 > 0**

$$
\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq 0
$$

**딥러닝에서**: Loss의 Hessian이 positive definite면 극소점!

```python
# Positive definite 확인
def is_positive_definite(A):
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > 0)

A = np.array([[2, 1], [1, 2]])
print(f"Positive definite: {is_positive_definite(A)}")  # True
```

### 직교 행렬 (Orthogonal Matrix)

$$
Q^T Q = I
$$

**모든 고유값의 절대값 = 1** (회전/반사만 수행, 크기 불변)

```python
# 회전 행렬 (직교)
theta = np.pi / 4
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

eigenvalues = np.linalg.eigvals(Q)
print(f"고유값: {eigenvalues}")
print(f"절대값: {np.abs(eigenvalues)}")  # [1, 1]
```

---

## 코드로 확인: 전체 파이프라인

```python
import numpy as np
import torch

print("=== 기본 고유값 분해 ===")
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"행렬 A:\n{A}")
print(f"고유값: {eigenvalues}")
print(f"고유벡터:\n{eigenvectors}")

# 검증: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nλ_{i+1} = {lam:.2f}")
    print(f"Av   = {Av}")
    print(f"λv   = {lam_v}")
    print(f"일치: {np.allclose(Av, lam_v)}")

print("\n=== PCA 예시 ===")
# 2D 데이터 생성 (한 방향으로 길쭉한 분포)
np.random.seed(42)
X = np.random.randn(100, 2) @ np.array([[3, 1], [1, 0.5]])

# PCA
X_centered = X - X.mean(axis=0)
cov = X_centered.T @ X_centered / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

print(f"공분산 행렬:\n{cov}")
print(f"고유값: {eigenvalues}")  # 작은 것, 큰 것 순서
print(f"큰 고유값 방향이 데이터가 가장 퍼진 방향!")

print("\n=== 학습 안정성 ===")
# Condition number 예시
H1 = np.array([[1, 0], [0, 1]])  # Identity
H2 = np.array([[100, 0], [0, 1]])  # Bad conditioning

def condition_number(H):
    eigs = np.linalg.eigvalsh(H)
    return np.max(eigs) / np.min(eigs)

print(f"H1 condition number: {condition_number(H1)}")  # 1 (ideal)
print(f"H2 condition number: {condition_number(H2)}")  # 100 (bad)
print("Condition number가 크면 learning rate 선택이 어려움!")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 적용 |
|------|------|------------|
| 고유값 분해 | $A\mathbf{v} = \lambda\mathbf{v}$ | PCA, 안정성 분석 |
| Trace | $\sum \lambda_i$ | 행렬의 특성 파악 |
| Determinant | $\prod \lambda_i$ | 역행렬 존재 여부 |
| Condition Number | $\lambda_{max}/\lambda_{min}$ | 학습 난이도 |
| Spectral Norm | $\sigma_{max}$ | GAN 안정화 |

## 핵심 통찰

1. **고유벡터 = 행렬이 당기는 방향**: 변환해도 방향이 바뀌지 않는 특별한 축
2. **고유값 = 당기는 세기**: 그 방향으로 얼마나 늘어나는지
3. **PCA = 데이터 분산 방향 찾기**: 공분산의 고유벡터가 주성분
4. **Condition Number = 학습 난이도**: 클수록 최적화 어려움

---

## 다음 단계

고유값 분해는 정사각 행렬에만 적용됩니다. 일반 행렬을 분해하려면?

→ [SVD](/ko/docs/math/linear-algebra/svd): 모든 행렬을 분해하는 방법

## 관련 콘텐츠

- [행렬 연산](/ko/docs/math/linear-algebra/matrix) - 행렬의 기초
- [SVD](/ko/docs/math/linear-algebra/svd) - 고유값 분해의 일반화
- [Gradient](/ko/docs/math/calculus/gradient) - Hessian과 고유값
- [최적화 수학](/ko/docs/math/calculus/optimization) - Condition number와 학습률
