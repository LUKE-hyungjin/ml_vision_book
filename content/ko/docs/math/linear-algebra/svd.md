---
title: "SVD"
weight: 3
math: true
---

# 특이값 분해 (Singular Value Decomposition)

## 개요

모든 행렬을 세 행렬의 곱으로 분해하는 방법으로, 모델 압축과 LoRA의 수학적 기반입니다.

## 정의

임의의 m × n 행렬 A에 대해:

$$
A = U \Sigma V^T
$$

- **U**: m × m 직교 행렬 (좌특이벡터)
- **Σ**: m × n 대각 행렬 (특이값, σ₁ ≥ σ₂ ≥ ... ≥ 0)
- **V**: n × n 직교 행렬 (우특이벡터)

### 직관적 이해

행렬 A의 작용을 세 단계로 분해:
1. **V^T**: 입력 공간에서 회전
2. **Σ**: 각 축 방향으로 스케일링
3. **U**: 출력 공간에서 회전

## 저랭크 근사 (Low-Rank Approximation)

상위 r개의 특이값만 사용하여 행렬을 근사:

$$
A \approx A_r = U_r \Sigma_r V_r^T
$$

이는 Frobenius 노름 관점에서 최적의 랭크-r 근사입니다 (Eckart-Young 정리).

## 구현

```python
import torch
import numpy as np

# NumPy SVD
A = np.random.randn(100, 50)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
# U: (100, 50), S: (50,), Vt: (50, 50)

# 저랭크 근사 (랭크 10)
r = 10
A_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
error = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
print(f"상대 오차: {error:.4f}")

# PyTorch SVD
A = torch.randn(100, 50)
U, S, Vt = torch.linalg.svd(A, full_matrices=False)
```

## 딥러닝에서의 활용

### 1. 모델 압축

큰 가중치 행렬을 저랭크 행렬로 근사하여 파라미터 수 감소:

```python
# 원본: W는 (1024, 1024) = 1M 파라미터
W = model.fc.weight.data

# SVD 분해
U, S, Vt = torch.linalg.svd(W, full_matrices=False)

# 랭크 64로 압축: 64 * 1024 * 2 = 131K 파라미터 (87% 감소)
r = 64
W_approx = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]
```

### 2. LoRA (Low-Rank Adaptation)

사전학습 가중치 W를 고정하고 저랭크 업데이트 ΔW만 학습:

$$
W' = W + \Delta W = W + BA
$$

- B: (d, r) 행렬
- A: (r, k) 행렬
- r << min(d, k)

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)
        self.A = torch.nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return x @ self.W.T + x @ self.A.T @ self.B.T
```

### 3. 이미지 압축

이미지를 행렬로 보고 SVD로 압축:

```python
from PIL import Image
import numpy as np

# 그레이스케일 이미지 로드
img = np.array(Image.open('image.jpg').convert('L'), dtype=float)

# SVD 분해
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# 상위 50개 특이값만 사용
k = 50
img_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

## 관련 콘텐츠

- [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue)
- [LoRA](/ko/docs/math/training/peft/lora) - SVD 기반 파인튜닝
- [모델 최적화](/ko/docs/engineering/deployment/optimization)
