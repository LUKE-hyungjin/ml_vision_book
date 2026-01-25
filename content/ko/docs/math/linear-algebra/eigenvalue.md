---
title: "고유값/고유벡터"
weight: 2
math: true
---

# 고유값과 고유벡터 (Eigenvalue & Eigenvector)

## 개요

행렬의 본질적인 특성을 나타내는 개념으로, PCA나 모델 안정성 분석에 사용됩니다.

## 정의

행렬 A에 대해 다음을 만족하는 벡터 v와 스칼라 λ:

$$
Av = \lambda v
$$

- **v**: 고유벡터 (eigenvector) - 방향이 변하지 않는 벡터
- **λ**: 고유값 (eigenvalue) - 스케일 변화량

### 직관적 이해

행렬 A를 선형 변환으로 볼 때, 고유벡터는 변환 후에도 방향이 유지되는 특별한 방향입니다.

## 특성 방정식

$$
\det(A - \lambda I) = 0
$$

이 방정식의 해가 고유값이 됩니다.

## 구현

```python
import torch
import numpy as np

# NumPy
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"고유값: {eigenvalues}")        # [5. 2.]
print(f"고유벡터:\n{eigenvectors}")

# PyTorch
A = torch.tensor([[4., 2.], [1., 3.]])
eigenvalues, eigenvectors = torch.linalg.eig(A)
```

## 딥러닝에서의 활용

### 1. PCA (Principal Component Analysis)

공분산 행렬의 고유벡터 → 주성분 방향

```python
# 데이터 중심화
X_centered = X - X.mean(axis=0)
# 공분산 행렬
cov = X_centered.T @ X_centered / len(X)
# 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(cov)
# 상위 k개 주성분 선택
top_k = eigenvectors[:, :k]
X_reduced = X_centered @ top_k
```

### 2. 학습 안정성 분석

Hessian 행렬의 고유값으로 손실 함수의 곡률 분석:
- 모든 고유값 > 0: 극소점
- 고유값 중 음수 존재: 안장점
- 고유값 범위가 크면 학습 불안정

### 3. 스펙트럼 정규화

가중치 행렬의 최대 고유값(스펙트럼 노름)을 제한:

$$
W_{normalized} = \frac{W}{\sigma(W)}
$$

여기서 σ(W)는 W의 최대 특이값

## 관련 콘텐츠

- [행렬 연산](/ko/docs/math/linear-algebra/matrix)
- [SVD](/ko/docs/math/linear-algebra/svd) - 고유값 분해의 일반화
- [Batch Normalization](/ko/docs/math/normalization/batch-norm)
