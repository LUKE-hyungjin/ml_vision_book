---
title: "행렬 연산"
weight: 1
math: true
---

# 행렬 연산 (Matrix Operations)

## 개요

딥러닝에서 데이터와 파라미터는 모두 텐서(다차원 배열)로 표현되며, 모든 연산은 행렬/텐서 연산으로 구성됩니다.

## 기본 연산

### 행렬 곱셈

$$
C = AB, \quad c_{ij} = \sum_k a_{ik} b_{kj}
$$

- A: (m, n) 행렬
- B: (n, p) 행렬
- C: (m, p) 행렬

### 전치 (Transpose)

$$
(A^T)_{ij} = A_{ji}
$$

### Fully Connected Layer

$$
y = Wx + b
$$

- x: 입력 벡터 (n,)
- W: 가중치 행렬 (m, n)
- b: 편향 벡터 (m,)
- y: 출력 벡터 (m,)

## 구현

```python
import torch

# Fully Connected Layer
x = torch.randn(32, 784)    # 배치 32, 입력 784
W = torch.randn(256, 784)   # 출력 256, 입력 784
b = torch.randn(256)

# 행렬 곱: (32, 784) @ (784, 256) = (32, 256)
y = x @ W.T + b

# PyTorch Linear Layer
fc = torch.nn.Linear(784, 256)
y = fc(x)  # 동일한 연산
```

## Batch 행렬 곱 (BMM)

Attention 등에서 배치 단위 행렬 곱이 필요합니다.

$$
C_i = A_i B_i, \quad i = 1, \ldots, B
$$

```python
# Batch Matrix Multiplication
A = torch.randn(32, 64, 128)  # (B, M, K)
B = torch.randn(32, 128, 64)  # (B, K, N)
C = torch.bmm(A, B)           # (B, M, N) = (32, 64, 64)
```

## 브로드캐스팅

차원이 다른 텐서 간 연산 시 자동으로 차원을 맞춥니다.

```python
A = torch.randn(32, 64, 128)  # (B, H, W)
b = torch.randn(128)          # (W,)
C = A + b  # b가 (1, 1, 128)로 확장되어 연산
```

## 관련 콘텐츠

- [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue)
- [SVD](/ko/docs/math/linear-algebra/svd)
- [Attention](/ko/docs/math/attention) - 행렬 곱의 활용
