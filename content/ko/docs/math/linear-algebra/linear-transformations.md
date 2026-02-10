---
title: "선형 변환"
weight: 4
math: true
---

# 선형 변환 (Linear Transformations)

{{% hint info %}}
**선수지식**: [행렬 연산](/ko/docs/math/linear-algebra/matrix) | [벡터 공간](/ko/docs/math/linear-algebra/vector-spaces)
{{% /hint %}}

> **한 줄 요약**: 선형 변환은 **행렬로 표현되는 공간 변환**입니다. 회전, 투영, 스케일링 — 딥러닝의 모든 레이어가 하는 일입니다.

## 왜 선형 변환을 배워야 하나요?

### 문제 상황 1: "Linear Layer가 정확히 뭘 하는 건가요?"

```python
layer = nn.Linear(784, 256)
output = layer(input)  # 784차원 → 256차원... 뭐가 일어나는 거지?
```

→ 784차원 공간의 점을 256차원 공간으로 **투영(projection)**합니다.

### 문제 상황 2: "이미지 증강에서 회전/반전은 어떻게 구현하나요?"

```python
# 이미지 회전, 반전, 전단... 이게 다 행렬이라고?
transformed = torchvision.transforms.RandomAffine(degrees=30)(image)
```

→ 모든 기하학적 변환은 **변환 행렬** 하나로 표현됩니다.

### 문제 상황 3: "왜 활성화 함수가 꼭 필요한가요?"

```python
# Linear만 쌓으면 안 되나?
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.Linear(512, 256),  # 이거 의미 없는 거 아닌가?
)
```

→ 선형 변환의 합성은 여전히 선형 변환! **비선형 활성화 없으면 깊은 네트워크가 의미 없습니다.**

---

## 선형 변환이란?

### 정의

함수 $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$이 **선형 변환**이려면:

$$
T(c_1\mathbf{u} + c_2\mathbf{v}) = c_1 T(\mathbf{u}) + c_2 T(\mathbf{v})
$$

두 가지 성질:
1. **가법성**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **동차성**: $T(c\mathbf{u}) = c \cdot T(\mathbf{u})$

### 핵심: 모든 선형 변환은 행렬이다

$T$가 선형 변환이면, 반드시 어떤 행렬 $A$가 존재하여:

$$
T(\mathbf{x}) = A\mathbf{x}
$$

```python
import torch

# 선형 변환 = 행렬 곱
A = torch.tensor([[2., 0.],
                  [0., 3.]])

def T(x):
    return A @ x

# 선형성 확인
u = torch.tensor([1., 2.])
v = torch.tensor([3., 4.])
c1, c2 = 2.0, 3.0

left = T(c1 * u + c2 * v)
right = c1 * T(u) + c2 * T(v)
print(f"T(c1*u + c2*v) = {left}")
print(f"c1*T(u) + c2*T(v) = {right}")
print(f"같은가? {torch.allclose(left, right)}")  # True
```

{{< figure src="/images/math/linear-algebra/ko/linear-transforms.jpeg" caption="주요 선형 변환: 단위 정사각형이 각 변환에 의해 어떻게 변하는지" >}}

---

## 주요 선형 변환 유형

### 1. 스케일링 (Scaling)

각 축 방향으로 크기를 늘리거나 줄이기:

$$
S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}
$$

```python
# x축 2배, y축 0.5배
scale = torch.tensor([[2., 0.],
                      [0., 0.5]])

v = torch.tensor([1., 1.])
print(f"원래: {v}")
print(f"스케일: {scale @ v}")  # [2, 0.5]
```

**딥러닝 적용**: BatchNorm의 γ (scale parameter)

```python
# BatchNorm: y = γ * x_normalized + β
# γ가 대각 행렬처럼 각 채널을 독립적으로 스케일
```

### 2. 회전 (Rotation)

원점을 중심으로 각도 θ만큼 회전:

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

```python
import math

def rotation_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])

# 45도 회전
R = rotation_matrix(math.pi / 4)
v = torch.tensor([1., 0.])
print(f"원래: {v}")
print(f"45도 회전: {R @ v}")  # [0.707, 0.707]

# 회전의 특징: 크기 보존, det = 1
print(f"det(R) = {torch.linalg.det(R):.4f}")  # 1.0
print(f"|v| = {torch.norm(v):.4f}")
print(f"|Rv| = {torch.norm(R @ v):.4f}")  # 같음!
```

**딥러닝 적용**: 이미지 증강의 회전

```python
# torchvision의 회전도 내부적으로 회전 행렬 사용
# RandomRotation(30) → R(θ)를 이미지 좌표에 적용
```

### 3. 반사 (Reflection)

축을 기준으로 뒤집기:

```python
# x축 기준 반사 (y좌표 뒤집기)
reflect_x = torch.tensor([[1., 0.],
                           [0., -1.]])

# y축 기준 반사 (x좌표 뒤집기)
reflect_y = torch.tensor([[-1., 0.],
                           [0., 1.]])

v = torch.tensor([3., 2.])
print(f"원래: {v}")
print(f"x축 반사: {reflect_x @ v}")  # [3, -2]
print(f"y축 반사: {reflect_y @ v}")  # [-3, 2]

# 반사의 특징: det = -1
print(f"det = {torch.linalg.det(reflect_x):.0f}")  # -1
```

**딥러닝 적용**: 이미지 증강의 Flip

```python
# RandomHorizontalFlip → y축 기준 반사 행렬 적용
```

### 4. 전단 (Shear)

한 축 방향으로 기울이기:

$$
\text{Shear}_x = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}
$$

```python
# x방향 전단 (k=0.5)
shear = torch.tensor([[1., 0.5],
                       [0., 1.]])

v = torch.tensor([1., 2.])
print(f"원래: {v}")
print(f"전단: {shear @ v}")  # [2, 2] (x가 y*0.5만큼 밀림)
```

### 5. 투영 (Projection) — 가장 중요!

고차원에서 저차원으로 "그림자 만들기":

$$
P = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}
$$

```python
# 벡터 u 방향으로의 투영
u = torch.tensor([1., 1.]) / (2**0.5)  # 단위 벡터

# 투영 행렬
P = torch.outer(u, u)  # uu^T (u가 단위벡터이므로 분모 불필요)
print(f"투영 행렬:\n{P}")

v = torch.tensor([3., 1.])
proj = P @ v
print(f"원래: {v}")
print(f"투영: {proj}")  # u 방향 성분만 남음

# 투영의 핵심 성질: 두 번 해도 같음 (P² = P)
print(f"P²=P? {torch.allclose(P @ P, P)}")  # True (멱등성)
```

**딥러닝 적용**: Attention은 투영의 연속!

```python
# Attention의 본질:
# 1. Q, K, V = 입력을 3개의 부분공간으로 투영
# 2. QK^T = 유사도 계산 (내적)
# 3. softmax(QK^T/√d) @ V = V를 가중 투영

# 각 Attention Head = 서로 다른 부분공간으로의 투영
```

---

## 변환의 합성

### 핵심: 행렬 곱 = 변환의 합성

$T_2 \circ T_1$을 적용하려면 $A_2 A_1$:

$$
T_2(T_1(\mathbf{x})) = A_2(A_1\mathbf{x}) = (A_2 A_1)\mathbf{x}
$$

```python
# 먼저 45도 회전, 그 다음 2배 확대
R = rotation_matrix(math.pi / 4)
S = torch.tensor([[2., 0.],
                  [0., 2.]])

# 합성 변환
T_combined = S @ R  # 주의: 오른쪽부터 적용!

v = torch.tensor([1., 0.])
# 방법 1: 순차 적용
v1 = R @ v      # 먼저 회전
v2 = S @ v1     # 그 다음 확대

# 방법 2: 합성 행렬로 한 번에
v3 = T_combined @ v

print(f"순차: {v2}")
print(f"합성: {v3}")
print(f"같은가? {torch.allclose(v2, v3)}")  # True
```

### 딥러닝의 핵심 한계: 선형의 합성은 선형

```python
# Linear Layer를 아무리 쌓아도...
W1 = torch.randn(256, 784)
W2 = torch.randn(128, 256)
W3 = torch.randn(10, 128)

# 세 층을 순차 적용
x = torch.randn(784)
y_sequential = W3 @ (W2 @ (W1 @ x))

# = 하나의 행렬로 축약 가능!
W_combined = W3 @ W2 @ W1  # (10, 784)
y_combined = W_combined @ x

print(f"순차: {y_sequential[:3]}")
print(f"합성: {y_combined[:3]}")
print(f"같은가? {torch.allclose(y_sequential, y_combined, atol=1e-4)}")  # True!

# → 3층 Linear = 1층 Linear와 같다!
# → 그래서 비선형 활성화 함수(ReLU)가 필수!
```

**비선형 활성화가 있으면?**

```python
import torch.nn.functional as F

x = torch.randn(784)

# 활성화 함수 포함
y1 = W1 @ x
y2 = W2 @ F.relu(y1)   # ReLU가 선형성을 깨뜨림!
y3 = W3 @ F.relu(y2)

# 이제 W3 @ W2 @ W1으로 축약 불가능!
# → 깊은 네트워크가 의미를 가짐
```

---

## 상 (Image)과 핵 (Kernel)

### 상 (Image) = 변환의 출력 범위

$$
\text{Im}(T) = \{T(\mathbf{x}) \mid \mathbf{x} \in \mathbb{R}^n\} = \text{행렬 A의 열공간}
$$

### 핵 (Kernel) = 0으로 보내지는 입력들

$$
\text{Ker}(T) = \{\mathbf{x} \mid T(\mathbf{x}) = \mathbf{0}\} = \text{행렬 A의 영공간}
$$

### 차원 정리 (Rank-Nullity Theorem)

$$
\dim(\text{Ker}) + \dim(\text{Im}) = n \quad \text{(입력 차원)}
$$

```python
# 3차원 → 2차원 투영 (z 성분 버리기)
A = torch.tensor([[1., 0., 0.],
                  [0., 1., 0.]])  # (2×3)

# Image: ℝ²의 전체 (rank = 2)
# Kernel: z축 방향 [0, 0, t] (nullity = 1)
# 확인: 2 + 1 = 3 ✓

print(f"Rank (dim Image): {torch.linalg.matrix_rank(A)}")  # 2
# Nullity = 3 - 2 = 1

# z축 벡터는 커널에 속함
z = torch.tensor([0., 0., 5.])
print(f"A @ z = {A @ z}")  # [0, 0] → 0으로 매핑됨!
```

**딥러닝 적용**: Linear Layer의 정보 손실

```python
# nn.Linear(784, 256): 784차원 → 256차원
# Kernel 차원 ≥ 784 - 256 = 528
# → 최소 528차원의 정보가 손실됨!
# → 하지만 중요한 정보만 살아남도록 학습됨
```

---

## 코드로 확인: 변환 시각화

```python
import torch
import numpy as np
import math

print("=== 변환 유형 비교 ===")
v = torch.tensor([1.0, 0.5])

transforms = {
    "원래": torch.eye(2),
    "2배 확대": torch.tensor([[2., 0.], [0., 2.]]),
    "x축 반사": torch.tensor([[1., 0.], [0., -1.]]),
    "45도 회전": rotation_matrix(math.pi / 4),
    "전단": torch.tensor([[1., 0.5], [0., 1.]]),
    "x축 투영": torch.tensor([[1., 0.], [0., 0.]]),
}

for name, A in transforms.items():
    result = A @ v
    det = torch.linalg.det(A) if A.shape[0] == A.shape[1] else float('nan')
    print(f"{name:10s}: {v.tolist()} → {result.tolist()}  (det={det:.2f})")

print("\n=== 선형 변환 합성 = 단일 행렬 ===")
# 회전 → 확대 → 전단
R = rotation_matrix(math.pi / 6)  # 30도 회전
S = torch.tensor([[1.5, 0.], [0., 1.5]])  # 1.5배 확대
H = torch.tensor([[1., 0.3], [0., 1.]])  # 전단

# 합성 (오른쪽부터 적용)
T = H @ S @ R

v = torch.tensor([1., 0.])
# 순차 적용
v1 = R @ v
v2 = S @ v1
v3 = H @ v2

# 합성 행렬로 한 번에
v_combined = T @ v

print(f"순차 적용: {v3}")
print(f"합성 행렬: {v_combined}")
print(f"일치: {torch.allclose(v3, v_combined)}")

print("\n=== 왜 비선형이 필요한가 ===")
import torch.nn.functional as F

W1 = torch.randn(64, 128)
W2 = torch.randn(32, 64)
x = torch.randn(128)

# 선형만: 축약 가능
y_linear = W2 @ (W1 @ x)
y_combined = (W2 @ W1) @ x
print(f"선형만 2층 = 1층? {torch.allclose(y_linear, y_combined, atol=1e-4)}")  # True

# ReLU 추가: 축약 불가능
y_relu = W2 @ F.relu(W1 @ x)
# (W2 @ W1) @ x와 다름! → 더 풍부한 표현력
```

---

## 핵심 정리

| 변환 유형 | 행렬 | det | 보존하는 것 | 딥러닝 적용 |
|----------|------|-----|-----------|------------|
| 스케일링 | 대각 행렬 | $\prod s_i$ | 축 방향 | BatchNorm γ |
| 회전 | 직교 행렬 | 1 | 크기, 각도 | 이미지 증강 |
| 반사 | 직교 행렬 | -1 | 크기 | 이미지 Flip |
| 전단 | 삼각 행렬 | 1 | 면적 | Affine 변환 |
| 투영 | $P^2=P$ | 0 | 부분공간 성분 | Attention, Linear Layer |

## 핵심 통찰

1. **모든 선형 변환 = 행렬 곱**: 행렬만 알면 변환의 모든 것을 안다
2. **합성 = 행렬 곱**: 여러 변환을 하나의 행렬로 축약 가능
3. **선형의 한계**: Linear Layer만 쌓으면 1층과 같다 → ReLU가 필수인 이유
4. **투영이 가장 중요**: Attention, PCA, Linear Layer 모두 투영의 변형

---

## 다음 단계

변환의 종류를 이해했습니다. 이제 **방정식 Ax = b를 푸는 방법**을 알아봅니다.

→ [선형 시스템](/ko/docs/math/linear-algebra/linear-systems): 카메라 캘리브레이션부터 최소제곱법까지

## 관련 콘텐츠

- [행렬 연산](/ko/docs/math/linear-algebra/matrix) - 행렬 곱, 전치, 역행렬
- [벡터 공간](/ko/docs/math/linear-algebra/vector-spaces) - 기저, 차원, 부분공간
- [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue) - 변환의 본질적 축
- [SVD](/ko/docs/math/linear-algebra/svd) - 최적의 저랭크 근사
