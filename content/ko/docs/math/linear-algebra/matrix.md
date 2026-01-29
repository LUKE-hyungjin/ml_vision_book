---
title: "행렬 연산"
weight: 2
math: true
---

# 행렬 연산 (Matrix Operations)

> **한 줄 요약**: 행렬은 **벡터들의 모음**이자 **공간 변환**입니다. 딥러닝의 모든 레이어는 행렬 곱셈입니다.

## 왜 행렬을 배워야 하나요?

### 문제 상황 1: "nn.Linear가 뭘 하는 건가요?"

```python
layer = nn.Linear(784, 256)
output = layer(input)  # ← 내부에서 뭐가 일어나지?
```

→ 내부적으로 **행렬 곱셈**: $y = Wx + b$ (W는 256×784 행렬)

### 문제 상황 2: "왜 shape이 안 맞는다고 에러가 나나요?"

```python
A = torch.randn(32, 784)
B = torch.randn(256, 784)
C = A @ B  # RuntimeError: mat1 and mat2 shapes cannot be multiplied!
```

→ 행렬 곱셈의 **차원 규칙**을 모르면 항상 에러와 싸웁니다.

### 문제 상황 3: "Attention의 QK^T가 뭔가요?"

```python
# Transformer Attention
scores = torch.matmul(Q, K.transpose(-2, -1))  # QK^T가 뭐지?
```

→ 행렬 곱과 전치를 이해해야 Attention을 이해할 수 있습니다.

---

## 행렬이란 무엇인가?

### 정의 1: 숫자들의 2차원 배열

$$
A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}
$$

- **m × n 행렬**: m개의 행(row), n개의 열(column)
- $a_{ij}$: i번째 행, j번째 열의 원소

```python
import torch

# 3×4 행렬 (3행 4열)
A = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
], dtype=torch.float32)

print(f"Shape: {A.shape}")  # torch.Size([3, 4])
print(f"A[1, 2] = {A[1, 2]}")  # 7.0 (2번째 행, 3번째 열 - 0-indexed)
```

### 정의 2: 벡터들의 모음

**행렬 = 벡터들을 쌓아놓은 것**

```python
# 행 벡터들의 모음으로 보기
row_vectors = [
    torch.tensor([1, 2, 3, 4]),
    torch.tensor([5, 6, 7, 8]),
    torch.tensor([9, 10, 11, 12])
]
A = torch.stack(row_vectors)

# 열 벡터들의 모음으로 보기
col_vectors = [
    torch.tensor([1, 5, 9]),
    torch.tensor([2, 6, 10]),
    torch.tensor([3, 7, 11]),
    torch.tensor([4, 8, 12])
]
A = torch.stack(col_vectors, dim=1)
```

**딥러닝에서의 의미**:

| 관점 | 해석 | 예시 |
|------|------|------|
| 행 벡터 | 각 데이터 샘플 | 배치에서 각 이미지 |
| 열 벡터 | 각 특징(feature) | 임베딩의 각 차원 |

### 정의 3: 공간 변환 (가장 중요!)

**행렬 = 벡터를 다른 벡터로 바꾸는 함수**

$$
\mathbf{y} = A\mathbf{x}
$$

- 입력 벡터 $\mathbf{x}$를 출력 벡터 $\mathbf{y}$로 **변환**
- 행렬 A는 "어떻게 변환할지"를 정의

```python
# 2D 회전 변환 (45도)
import math
theta = math.pi / 4  # 45도

# 회전 행렬
R = torch.tensor([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]
])

# 벡터 [1, 0]을 45도 회전
x = torch.tensor([1.0, 0.0])
y = R @ x
print(f"회전 후: {y}")  # [0.707, 0.707]
```

**딥러닝에서의 의미**: Linear Layer = 공간 변환!

```python
# 784차원 → 256차원 변환
linear = nn.Linear(784, 256)
# 이것은 784차원 공간의 점을 256차원 공간의 점으로 "변환"하는 것
```

---

## 행렬 곱셈: 가장 중요한 연산

### 정의

$$
C = AB, \quad c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
$$

**직관**: C의 (i, j) 원소 = A의 i번째 **행**과 B의 j번째 **열**의 **내적**

```python
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2×2
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)  # 2×2

# 행렬 곱
C = A @ B  # 또는 torch.matmul(A, B)
print(C)
# [[19, 22],   ← [1,2]·[5,7]=19, [1,2]·[6,8]=22
#  [43, 50]]  ← [3,4]·[5,7]=43, [3,4]·[6,8]=50
```

### 차원 규칙 (절대 암기!)

$$
(m \times \mathbf{k}) \cdot (\mathbf{k} \times n) = (m \times n)
$$

- **안쪽 차원(k)은 같아야 함**: 그래야 내적 가능
- **바깥 차원(m, n)이 결과**: 결과 행렬의 shape

```python
# 올바른 예
A = torch.randn(32, 784)   # (32, 784)
B = torch.randn(784, 256)  # (784, 256)
C = A @ B                  # (32, 256) ✓

# 틀린 예
A = torch.randn(32, 784)   # (32, 784)
B = torch.randn(256, 784)  # (256, 784)
C = A @ B                  # Error! 784 ≠ 256
```

### 왜 순서가 중요한가?

$$
AB \neq BA
$$

행렬 곱셈은 **교환법칙이 성립하지 않습니다!**

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

print((A @ B).shape)  # (3, 5) ✓
# print((B @ A).shape)  # Error! (4, 5) @ (3, 4) 불가능
```

---

## 전치 (Transpose)

### 정의

$$
(A^T)_{ij} = A_{ji}
$$

행과 열을 뒤바꿉니다.

```python
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # (2, 3)

A_T = A.T  # 또는 A.transpose(0, 1)
print(A_T)
# [[1, 4],
#  [2, 5],
#  [3, 6]]  → (3, 2)
```

### 전치의 성질

$$
(AB)^T = B^T A^T
$$

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# (AB)^T = B^T A^T
left = (A @ B).T              # (5, 3)
right = B.T @ A.T             # (5, 4) @ (4, 3) = (5, 3)
print(torch.allclose(left, right))  # True
```

### 딥러닝에서의 전치

**1. Linear Layer에서**

```python
# PyTorch Linear: y = xW^T + b
linear = nn.Linear(784, 256)
print(linear.weight.shape)  # (256, 784) ← 출력 × 입력

x = torch.randn(32, 784)
# 내부적으로: x @ weight.T + bias
y = linear(x)  # (32, 256)
```

**2. Attention에서: QK^T**

```python
# Query와 Key의 유사도 계산
Q = torch.randn(32, 64, 128)  # (batch, seq_len, dim)
K = torch.randn(32, 64, 128)

# QK^T: 각 Query와 모든 Key의 내적
scores = Q @ K.transpose(-2, -1)  # (32, 64, 64)
# scores[b, i, j] = Q[b, i]와 K[b, j]의 유사도
```

---

## nn.Linear의 본질

### 수식

$$
\mathbf{y} = W\mathbf{x} + \mathbf{b}
$$

- $\mathbf{x}$: 입력 벡터 (n차원)
- $W$: 가중치 행렬 (m × n)
- $\mathbf{b}$: 편향 벡터 (m차원)
- $\mathbf{y}$: 출력 벡터 (m차원)

### 직접 구현

```python
# nn.Linear(784, 256)이 하는 일
class MyLinear:
    def __init__(self, in_features, out_features):
        # Xavier 초기화
        self.weight = torch.randn(out_features, in_features) * 0.01
        self.bias = torch.zeros(out_features)

    def forward(self, x):
        # y = xW^T + b
        return x @ self.weight.T + self.bias

# 사용
my_linear = MyLinear(784, 256)
x = torch.randn(32, 784)  # 배치 32개
y = my_linear.forward(x)  # (32, 256)
```

### 기하학적 의미

**Linear Layer = 공간 변환**

```python
# 784차원 공간 → 256차원 공간
#
# 생각해보세요:
# - MNIST 이미지: 28×28 = 784 픽셀
# - 각 이미지는 784차원 공간의 한 점
# - Linear Layer는 이 점을 256차원 공간으로 "투영"
# - 비슷한 숫자들은 비슷한 위치로 모임!
```

---

## Batch 처리

### 왜 Batch가 필요한가?

```python
# 하나씩 처리 (느림)
for image in images:
    output = model(image)  # 반복문 오버헤드

# 배치로 처리 (빠름) - GPU 병렬화 활용
outputs = model(torch.stack(images))
```

### Batch와 행렬 곱

```python
# 단일 샘플
x = torch.randn(784)        # (784,)
W = torch.randn(256, 784)   # (256, 784)
y = W @ x                   # (256,)

# 배치 처리
X = torch.randn(32, 784)    # (32, 784) - 32개 샘플
# y = XW^T
Y = X @ W.T                 # (32, 256) - 32개 출력
```

### Batch Matrix Multiplication (BMM)

**Attention에서 필수!**

```python
# 32개 헤드, 각각 64×128 행렬과 128×64 행렬을 곱해야 함
A = torch.randn(32, 64, 128)  # (batch, M, K)
B = torch.randn(32, 128, 64)  # (batch, K, N)

# 일반 matmul은 마지막 2차원에 대해 행렬 곱
C = torch.matmul(A, B)        # (32, 64, 64)

# bmm은 3D 텐서 전용
C = torch.bmm(A, B)           # (32, 64, 64)
```

### Einstein Summation (고급)

복잡한 텐서 연산을 간결하게:

```python
# 일반적인 배치 행렬 곱
A = torch.randn(32, 64, 128)
B = torch.randn(32, 128, 256)
C = torch.einsum('bik,bkj->bij', A, B)  # (32, 64, 256)

# Multi-head Attention에서
Q = torch.randn(32, 8, 64, 64)   # (batch, head, seq, dim)
K = torch.randn(32, 8, 64, 64)
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)  # (32, 8, 64, 64)
```

---

## Broadcasting

### 문제: 차원이 다른 텐서끼리 연산하려면?

```python
# 모든 샘플에 같은 bias 더하기
X = torch.randn(32, 256)  # (32, 256)
b = torch.randn(256)       # (256,)

# b를 (32, 256)으로 "방송"
Y = X + b  # 자동으로 작동!
```

### Broadcasting 규칙

1. 뒤에서부터 차원 비교
2. 차원이 같거나, 하나가 1이면 OK
3. 1인 차원이 다른 쪽에 맞춰 복제됨

```python
A = torch.randn(32, 64, 128)  # (32, 64, 128)
b = torch.randn(128)          # (128,)

# b의 shape 변화: (128,) → (1, 1, 128) → (32, 64, 128)
C = A + b  # 정상 작동

# 실패 케이스
b2 = torch.randn(64)          # (64,)
# C = A + b2  # Error! 128 ≠ 64
```

### 딥러닝에서 Broadcasting

```python
# BatchNorm의 scale/shift
x = torch.randn(32, 64, 224, 224)  # (N, C, H, W)
gamma = torch.randn(64)             # (C,) - 채널별 scale
beta = torch.randn(64)              # (C,) - 채널별 shift

# (64,) → (1, 64, 1, 1) → (32, 64, 224, 224)로 broadcast
gamma = gamma.view(1, -1, 1, 1)
beta = beta.view(1, -1, 1, 1)
y = gamma * x + beta
```

---

## 행렬의 Rank (랭크)

### 왜 Rank를 알아야 하나요?

```python
# LoRA에서
config = LoraConfig(r=8)  # r = rank, 왜 8이면 충분하지?

# 모델 압축에서
# 4096×4096 행렬을 더 작게 표현할 수 있을까?
```

→ **Rank**는 행렬이 담고 있는 "실제 정보량"입니다.

### 정의

**Rank = 선형 독립인 행(또는 열)의 개수**

$$
\text{rank}(A) = \text{선형 독립인 행의 수} = \text{선형 독립인 열의 수}
$$

```python
import torch

# Full rank 행렬 (모든 행이 독립)
A = torch.tensor([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
print(f"Rank: {torch.linalg.matrix_rank(A)}")  # 3

# Rank-deficient 행렬 (한 행이 다른 행의 배수)
B = torch.tensor([[1., 2., 3.],
                  [2., 4., 6.],  # = 2 × 첫 번째 행
                  [0., 1., 0.]])
print(f"Rank: {torch.linalg.matrix_rank(B)}")  # 2
```

### 직관: "실제 차원"

- **m × n 행렬의 최대 rank** = min(m, n)
- **Rank = 출력 공간의 실제 차원**

```python
# 3×3 행렬이지만 rank가 2라면?
# → 3차원 출력이라고 해도, 실제로는 2차원 평면 위의 점만 생성
```

### 딥러닝에서의 Rank

**1. LoRA의 핵심 가정: 변화는 저랭크**

```python
# 파인튜닝 시 가중치 변화 ΔW
# 논문의 발견: ΔW는 대부분 저랭크!

# 원본: 4096 × 4096
W = torch.randn(4096, 4096)

# LoRA: rank=8로 충분
# ΔW = B @ A where B: (4096, 8), A: (8, 4096)
# rank(ΔW) ≤ 8
```

**2. 모델 압축 가능성 판단**

```python
# 가중치 행렬의 effective rank 확인
def effective_rank(W, threshold=0.99):
    """특이값 기준 effective rank 계산"""
    U, S, Vt = torch.linalg.svd(W)
    total = S.sum()
    cumsum = torch.cumsum(S, dim=0)
    return (cumsum / total < threshold).sum().item() + 1

W = torch.randn(1000, 1000)
print(f"Effective rank: {effective_rank(W)}")
# 랜덤 행렬은 거의 full rank
# 학습된 가중치는 종종 effective rank가 낮음!
```

---

## 행렬식 (Determinant)

### 왜 행렬식을 알아야 하나요?

```python
# Normalizing Flow에서
log_prob = base_log_prob - log_det_jacobian  # 이게 뭐지?
```

→ **행렬식**은 변환이 공간을 얼마나 늘리거나 줄이는지를 나타냅니다.

### 정의

$$
\det(A) = |A|
$$

**2×2 행렬의 경우**:

$$
\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$

```python
A = torch.tensor([[3., 1.],
                  [2., 4.]])
det = torch.linalg.det(A)
print(f"Determinant: {det}")  # 10.0 = 3*4 - 1*2
```

### 기하학적 의미

**행렬식 = 부피 변화율**

- |det| > 1: 공간이 늘어남
- |det| < 1: 공간이 줄어듦
- det = 0: 차원이 줄어듦 (압축)
- det < 0: 뒤집힘 (반사)

```python
# 2배 확대 행렬
scale = torch.tensor([[2., 0.],
                      [0., 2.]])
print(f"Det: {torch.linalg.det(scale)}")  # 4.0 (면적 4배)

# 회전 행렬 (크기 보존)
import math
theta = math.pi / 4
rotation = torch.tensor([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
print(f"Det: {torch.linalg.det(rotation)}")  # 1.0
```

### 중요한 성질

$$
\det(AB) = \det(A) \cdot \det(B)
$$

$$
\det(A^{-1}) = \frac{1}{\det(A)}
$$

$$
\det(A) = \prod_{i} \lambda_i \quad \text{(고유값의 곱)}
$$

### 딥러닝에서의 행렬식

**Normalizing Flow**

```python
# 확률 분포 변환 시 행렬식이 필요
# p(x) = p(z) * |det(dz/dx)|

# 간단한 Affine Flow
class AffineFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        x = z * self.scale + self.shift
        # log|det J| = sum(log|scale|)
        log_det = torch.log(torch.abs(self.scale)).sum()
        return x, log_det
```

---

## 역행렬 (Inverse Matrix)

### 왜 역행렬을 알아야 하나요?

**개념적으로 중요하지만, 실제로 직접 계산하는 경우는 드뭅니다.**

```python
# 이론적으로:
# Ax = b 를 풀려면 x = A^{-1}b
#
# 실제로는:
x = torch.linalg.solve(A, b)  # 더 안정적!
```

### 정의

$$
AA^{-1} = A^{-1}A = I
$$

역행렬이 존재하려면:
- **정사각 행렬**이어야 함
- **det(A) ≠ 0** (= full rank)

```python
A = torch.tensor([[4., 7.],
                  [2., 6.]])

# 역행렬 계산
A_inv = torch.linalg.inv(A)
print(f"A^(-1):\n{A_inv}")

# 검증
print(f"A @ A^(-1) = I: {torch.allclose(A @ A_inv, torch.eye(2))}")
```

### 역행렬이 없는 경우

```python
# Singular matrix (det = 0)
B = torch.tensor([[1., 2.],
                  [2., 4.]])  # 두 번째 행 = 2 × 첫 번째 행

print(f"Det: {torch.linalg.det(B)}")  # 0.0
# torch.linalg.inv(B)  # Error!
```

### 딥러닝에서 역행렬을 피하는 이유

1. **수치적 불안정**: 작은 오차가 크게 증폭
2. **계산 비용**: O(n³) 복잡도
3. **대안이 있음**: solve, pseudo-inverse 등

```python
# 선형 시스템 풀기
A = torch.randn(100, 100)
b = torch.randn(100)

# 나쁜 방법 (불안정)
# x = torch.linalg.inv(A) @ b

# 좋은 방법 (안정적)
x = torch.linalg.solve(A, b)

# 비정사각 행렬: pseudo-inverse
A_rect = torch.randn(100, 50)
A_pinv = torch.linalg.pinv(A_rect)  # Moore-Penrose pseudo-inverse
```

### Newton's Method와 역행렬

2차 최적화에서는 Hessian의 역행렬이 필요:

$$
\theta_{new} = \theta - H^{-1} \nabla L
$$

**실제로는**:
- Hessian이 너무 커서 역행렬 계산 불가능
- Adam 같은 근사 방법 사용

---

## 특수 행렬

### 단위 행렬 (Identity Matrix)

$$
I = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}
$$

- $AI = IA = A$ (항등원)
- 아무것도 안 하는 변환

```python
I = torch.eye(3)
A = torch.randn(3, 3)
print(torch.allclose(A @ I, A))  # True
```

**딥러닝 적용: Residual Connection의 기초**

```python
# y = F(x) + x
# 행렬로 표현하면: y = (W + I)x
# → 항등 변환(I)에 작은 변화(W)를 더함
```

### 대각 행렬 (Diagonal Matrix)

```python
# 대각선만 값이 있는 행렬
d = torch.tensor([1, 2, 3])
D = torch.diag(d)
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]
```

**딥러닝 적용: Layer-wise Learning Rate**

```python
# 각 파라미터 그룹에 다른 학습률
# θ_new = θ - D @ gradient
# D가 대각행렬이면 각 차원에 다른 스케일 적용
```

### 대칭 행렬 (Symmetric Matrix)

$$
A = A^T
$$

```python
# Covariance Matrix는 항상 대칭
X = torch.randn(100, 5)
cov = X.T @ X / (X.shape[0] - 1)
print(torch.allclose(cov, cov.T))  # True
```

---

## 코드로 확인: 전체 파이프라인

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Linear Layer 이해하기 ===")

# 1. nn.Linear의 동작
linear = nn.Linear(784, 256)
print(f"Weight shape: {linear.weight.shape}")  # (256, 784)
print(f"Bias shape: {linear.bias.shape}")      # (256,)

# 2. 수동으로 같은 연산
x = torch.randn(32, 784)

# PyTorch Linear
y1 = linear(x)

# 수동 행렬 곱
y2 = x @ linear.weight.T + linear.bias

print(f"결과 일치: {torch.allclose(y1, y2, atol=1e-6)}")  # True

print("\n=== Attention Score 계산 ===")

# 3. Attention에서의 행렬 곱
batch, seq_len, dim = 2, 4, 8
Q = torch.randn(batch, seq_len, dim)
K = torch.randn(batch, seq_len, dim)
V = torch.randn(batch, seq_len, dim)

# QK^T / sqrt(d)
scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)
print(f"Score shape: {scores.shape}")  # (2, 4, 4)

# Softmax + V
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)
print(f"Output shape: {output.shape}")  # (2, 4, 8)

print("\n=== 차원 변환 추적 ===")

# 4. 네트워크를 통한 차원 변화 추적
x = torch.randn(32, 784)  # MNIST flatten
print(f"Input: {x.shape}")

x = nn.Linear(784, 512)(x)
print(f"After Linear1: {x.shape}")  # (32, 512)

x = nn.Linear(512, 256)(x)
print(f"After Linear2: {x.shape}")  # (32, 256)

x = nn.Linear(256, 10)(x)
print(f"After Linear3: {x.shape}")  # (32, 10) - 10 classes
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 적용 |
|------|------|------------|
| 행렬 곱 | $C = AB$ | Linear Layer, Attention |
| 전치 | $A^T$ | QK^T, Weight 접근 |
| 차원 규칙 | $(m,k)(k,n)=(m,n)$ | Shape 디버깅 |
| Rank | 선형 독립 벡터 수 | LoRA, 모델 압축 |
| 행렬식 | $\det(A)$ | Flow 모델, 부피 변화 |
| 역행렬 | $A^{-1}$ | 이론적 이해 (실제론 solve) |
| BMM | 배치 행렬 곱 | Multi-head Attention |
| Broadcasting | 차원 자동 확장 | Bias 더하기, 정규화 |

## 핵심 통찰

1. **Linear Layer = 행렬 곱**: `y = xW^T + b`
2. **차원 규칙 암기**: 안쪽 같아야 곱해짐, 바깥이 결과
3. **Rank = 실제 정보량**: 저랭크면 압축 가능 (LoRA의 핵심)
4. **행렬식 = 부피 변화**: Flow 모델에서 확률 변환에 필수
5. **역행렬은 피하기**: solve가 더 안정적

---

## 다음 단계

행렬이 무엇이고 어떻게 곱하는지 이해했습니다. 이제 **행렬의 본질**을 파헤칩니다.

→ [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue): 행렬이 정말 하는 일은?

## 관련 콘텐츠

- [벡터 기초](/ko/docs/math/linear-algebra/vector) - 행렬의 구성 요소
- [Attention](/ko/docs/math/attention) - QKV 행렬 연산
- [CNN](/ko/docs/architecture) - 합성곱도 행렬 곱으로 변환 가능
