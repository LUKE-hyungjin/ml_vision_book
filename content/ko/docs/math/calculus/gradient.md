---
title: "Gradient"
weight: 2
math: true
---

# Gradient (기울기 벡터)

> **한 줄 요약**: Gradient는 "모든 방향 중 가장 가파르게 올라가는 방향"을 알려주는 벡터입니다.

## 왜 Gradient를 배워야 하나요?

### 문제 상황 1: "파라미터가 2개 이상이면 어떻게 하죠?"

```python
# 신경망의 한 레이어
layer = nn.Linear(10, 5)  # weight: 50개, bias: 5개 → 총 55개 파라미터
```

각각의 파라미터가 Loss에 미치는 영향을 **동시에** 알아야 합니다.
→ 55개의 미분값을 하나의 벡터로 모은 것이 **Gradient**

### 문제 상황 2: "어느 방향으로 이동해야 Loss가 가장 빨리 줄어들까요?"

```python
# 현재 파라미터: w1=2, w2=3
# Loss를 줄이려면 w1과 w2를 각각 얼마나 바꿔야 할까?
```

- w1만 바꾸기? w2만 바꾸기? 둘 다 바꾸기?
- Gradient는 **가장 효율적인 이동 방향**을 알려줍니다.

### 문제 상황 3: "optimizer.step()이 정확히 뭘 하는 건가요?"

```python
optimizer = SGD(model.parameters(), lr=0.01)
loss.backward()  # gradient 계산
optimizer.step()  # ← 이게 뭘 하는 거지?
```

→ `step()`은 **gradient의 반대 방향으로** 파라미터를 이동시킵니다.

---

## 편미분: Gradient의 재료

### 다변수 함수란?

입력이 여러 개인 함수:

$$
f(x, y) = x^2 + 3xy + y^2
$$

신경망의 Loss도 다변수 함수입니다:

$$
L(w_1, w_2, \ldots, w_n) = \text{손실값}
$$

### 편미분이란?

**다른 변수는 고정하고** 한 변수에 대해서만 미분:

$$
\frac{\partial f}{\partial x} = \text{"y를 상수로 보고, x에 대해 미분"}
$$

### 예시

$f(x, y) = x^2 + 3xy + y^2$ 의 편미분:

$$
\frac{\partial f}{\partial x} = 2x + 3y \quad \text{(y는 상수 취급)}
$$

$$
\frac{\partial f}{\partial y} = 3x + 2y \quad \text{(x는 상수 취급)}
$$

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

f = x**2 + 3*x*y + y**2  # 4 + 18 + 9 = 31

f.backward()

print(f"∂f/∂x = {x.grad.item()}")  # 2*2 + 3*3 = 13
print(f"∂f/∂y = {y.grad.item()}")  # 3*2 + 2*3 = 12
```

---

## Gradient의 정의

Gradient는 **모든 편미분을 모은 벡터**입니다:

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right)
$$

### 위 예시의 Gradient

$$
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) = (2x + 3y, 3x + 2y)
$$

$(x, y) = (2, 3)$에서:

$$
\nabla f = (13, 12)
$$

---

## Jacobian: 벡터 함수의 미분

### 왜 Jacobian이 필요한가?

Gradient는 **스칼라 출력** 함수에 대한 미분입니다:
- $f: \mathbb{R}^n \rightarrow \mathbb{R}$ (입력 n개 → 출력 1개)
- 예: Loss 함수

하지만 신경망 레이어는 **벡터를 출력**합니다:
- $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ (입력 n개 → 출력 m개)
- 예: Linear 레이어, Softmax

```python
layer = nn.Linear(10, 5)  # 입력 10차원 → 출력 5차원
# 각 출력이 각 입력에 어떻게 영향받는지?
```

### Jacobian의 정의

벡터 함수 $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$의 Jacobian:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

- **크기**: $m \times n$ 행렬
- **i행**: 출력 $f_i$가 각 입력에 얼마나 민감한가
- **j열**: 입력 $x_j$가 각 출력에 얼마나 영향을 주는가

### 예시: 2D → 2D 변환

$$
\mathbf{f}(x, y) = \begin{bmatrix} x^2 + y \\ xy \end{bmatrix}
$$

Jacobian:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\
\frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y}
\end{bmatrix} = \begin{bmatrix}
2x & 1 \\
y & x
\end{bmatrix}
$$

```python
import torch
from torch.autograd.functional import jacobian

def f(inputs):
    x, y = inputs[0], inputs[1]
    return torch.stack([x**2 + y, x*y])

inputs = torch.tensor([2.0, 3.0])
J = jacobian(f, inputs)

print("Jacobian:")
print(J)
# [[4., 1.],   ← ∂f₁/∂x=2*2=4, ∂f₁/∂y=1
#  [3., 2.]]   ← ∂f₂/∂x=y=3, ∂f₂/∂y=x=2
```

### 딥러닝에서의 Jacobian

**1. Linear 레이어의 Jacobian = Weight 행렬**

$\mathbf{y} = W\mathbf{x} + \mathbf{b}$일 때:

$$
J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = W
$$

```python
W = torch.randn(5, 10)  # 10 → 5
x = torch.randn(10, requires_grad=True)

y = W @ x

# Jacobian 계산
J = jacobian(lambda x: W @ x, x)
print(f"Jacobian shape: {J.shape}")  # [5, 10]
print(f"W == J: {torch.allclose(W, J)}")  # True
```

**2. Softmax의 Jacobian**

Softmax의 Jacobian은 대각 + 비대각 성분이 있음:

$$
\frac{\partial \text{softmax}_i}{\partial z_j} = \begin{cases}
p_i(1-p_i) & i = j \\
-p_i p_j & i \neq j
\end{cases}
$$

```python
def softmax(z):
    return torch.softmax(z, dim=0)

z = torch.tensor([2.0, 1.0, 0.1])
J_softmax = jacobian(softmax, z)

print("Softmax Jacobian:")
print(J_softmax)
# 대각: 양수, 비대각: 음수
# → 한 클래스 확률 증가 시 다른 클래스는 감소
```

**3. Flow 모델에서의 Jacobian**

Normalizing Flow에서는 Jacobian의 **행렬식(determinant)**이 중요:

$$
p_x(x) = p_z(f^{-1}(x)) \cdot |\det J_{f^{-1}}|
$$

```python
# 간단한 affine 변환
def affine_transform(x, scale, shift):
    return scale * x + shift

# log det Jacobian = log(scale)
scale = torch.tensor([2.0, 3.0])
log_det_J = torch.log(scale).sum()
print(f"log |det J| = {log_det_J.item()}")
```

---

## Hessian: 2차 미분 정보

### 왜 Hessian을 알아야 하나?

Gradient는 **1차 미분** = 기울기 (방향)
Hessian은 **2차 미분** = 곡률 (얼마나 휘어있는가)

```python
# 같은 gradient, 다른 Hessian
f1 = lambda x: x**2      # 볼록 (convex)
f2 = lambda x: -x**2     # 오목 (concave)
f3 = lambda x: x**3      # 안장점

# x=0에서 모두 gradient = 0
# 하지만 Hessian이 다름!
```

### Hessian의 정의

스칼라 함수 $f$의 Hessian:

$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

- **대칭 행렬**: $H_{ij} = H_{ji}$
- **대각 성분**: 각 방향의 곡률
- **고유값**: 주축 방향의 곡률

### 예시

$f(x, y) = x^2 + 3xy + 2y^2$

$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{bmatrix} = \begin{bmatrix}
2 & 3 \\
3 & 4
\end{bmatrix}
$$

```python
from torch.autograd.functional import hessian

def f(inputs):
    x, y = inputs[0], inputs[1]
    return x**2 + 3*x*y + 2*y**2

inputs = torch.tensor([1.0, 1.0])
H = hessian(f, inputs)

print("Hessian:")
print(H)
# [[2., 3.],
#  [3., 4.]]
```

### Hessian의 의미

**1. 고유값으로 곡률 판단**

```python
eigenvalues = torch.linalg.eigvalsh(H)
print(f"고유값: {eigenvalues}")

# 모두 양수 → 극소점 (볼록)
# 모두 음수 → 극대점 (오목)
# 부호 섞임 → 안장점
```

**2. Condition Number = 학습 난이도**

$$
\kappa = \frac{\lambda_{max}}{\lambda_{min}}
$$

```python
condition_number = eigenvalues.max() / eigenvalues.min()
print(f"Condition number: {condition_number:.2f}")

# 높으면 → 최적화 어려움 (가늘고 긴 골짜기)
# 낮으면 → 최적화 쉬움 (원형 골짜기)
```

### 딥러닝에서의 Hessian

**직접 계산하지 않는 이유**:
- 파라미터 N개 → Hessian은 N×N 행렬
- GPT-3: 175B 파라미터 → Hessian 저장 불가능!

**대신 근사 사용**:
- **Adam**: 2차 정보를 gradient의 제곱으로 근사
- **Natural Gradient**: Fisher Information으로 근사
- **Hessian-Free**: Hessian-벡터 곱만 계산

```python
# Adam은 Hessian 대각 성분을 근사
# v_t ≈ diag(H) 역할
optimizer = torch.optim.Adam(model.parameters())
# 내부적으로 gradient 제곱의 이동평균 사용
```

---

{{< figure src="/images/math/calculus/ko/gradient-comparison.svg" caption="Gradient, Jacobian, Hessian 비교: 입출력 타입과 딥러닝 적용" >}}

## Gradient vs Jacobian vs Hessian 정리

| 개념 | 입력 | 출력 | 결과 | 딥러닝 적용 |
|------|------|------|------|------------|
| **Gradient** | 벡터 | 스칼라 | 벡터 | Loss의 방향 |
| **Jacobian** | 벡터 | 벡터 | 행렬 | 레이어 변환 |
| **Hessian** | 벡터 | 스칼라 | 행렬 | 곡률, 최적화 |

```python
# 정리 코드
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Gradient: scalar → vector
loss = (x**2).sum()  # 스칼라
loss.backward()
print(f"Gradient: {x.grad}")  # [2, 4]

# Jacobian: vector → matrix
f = lambda x: torch.stack([x[0]**2, x[0]*x[1]])
J = jacobian(f, x)
print(f"Jacobian:\n{J}")

# Hessian: scalar → matrix
H = hessian(lambda x: (x**2).sum(), x)
print(f"Hessian:\n{H}")
```

---

## Gradient의 기하학적 의미

{{< figure src="/images/math/calculus/ko/gradient-contour.svg" caption="Gradient는 등고선에 수직이며, 반대 방향(-∇f)이 Loss 감소 방향" >}}

### 핵심 직관: 가장 가파른 오르막 방향

{{< mermaid >}}
graph TD
    A[현재 위치] --> B[Gradient 방향]
    B --> C[가장 빠르게 증가]
    A --> D[-Gradient 방향]
    D --> E[가장 빠르게 감소]
{{< /mermaid >}}

- **Gradient 방향**: 함수값이 가장 빠르게 **증가**하는 방향
- **Gradient 반대 방향**: 함수값이 가장 빠르게 **감소**하는 방향
- **Gradient 크기**: 얼마나 가파른가 (민감도)

### 등고선으로 이해하기

```
높은 곳 ●━━━━━━●
        ┃      ┃
        ┃  ↑   ┃  ← Gradient 방향 (오르막)
        ┃  │   ┃
낮은 곳 ●━━●━━━●
           ↓
         -Gradient 방향 (내리막)
```

Gradient는 항상 **등고선에 수직**입니다.

### 코드로 시각화

```python
import torch
import numpy as np

# 2D 함수: f(x, y) = x² + y²
def f(x, y):
    return x**2 + y**2

# 여러 점에서 gradient 계산
points = [
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
]

for px, py in points:
    x = torch.tensor([px], requires_grad=True)
    y = torch.tensor([py], requires_grad=True)

    z = f(x, y)
    z.backward()

    grad_mag = (x.grad**2 + y.grad**2).sqrt().item()
    print(f"점 ({px}, {py}): gradient = ({x.grad.item():.1f}, {y.grad.item():.1f}), 크기 = {grad_mag:.2f}")
```

출력:
```
점 (1.0, 0.0): gradient = (2.0, 0.0), 크기 = 2.00
점 (0.0, 1.0): gradient = (0.0, 2.0), 크기 = 2.00
점 (1.0, 1.0): gradient = (2.0, 2.0), 크기 = 2.83
점 (2.0, 1.0): gradient = (4.0, 2.0), 크기 = 4.47
```

**관찰**: 원점에서 멀수록 gradient 크기가 큼 (= 더 가파름)

---

## Gradient Descent: 핵심 알고리즘

Loss를 줄이려면 **Gradient의 반대 방향**으로 이동:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)
$$

- $\theta$: 파라미터 (weight, bias 등)
- $\eta$: Learning Rate (이동 거리 조절)
- $\nabla_\theta L$: Loss의 gradient

### 직관적 이해

```python
# 현재 산 중턱에 서있음
현재_위치 = (x, y)

# 가장 가파른 오르막 방향 측정
gradient = compute_gradient(현재_위치)

# 그 반대 방향(내리막)으로 이동
다음_위치 = 현재_위치 - learning_rate * gradient
```

### PyTorch로 구현

```python
import torch
import torch.nn as nn

# 간단한 최적화 문제: f(x) = (x-3)² 의 최솟값 찾기
x = torch.tensor([0.0], requires_grad=True)
lr = 0.1

print("Gradient Descent 시작!")
for step in range(20):
    # Forward
    loss = (x - 3) ** 2

    # Backward
    loss.backward()

    # Update (직접 구현)
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()

    if step % 5 == 0:
        print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")

print(f"\n최종: x = {x.item():.4f} (정답: 3)")
```

출력:
```
Gradient Descent 시작!
Step 0: x = 0.0000, loss = 9.0000
Step 5: x = 2.3534, loss = 0.4182
Step 10: x = 2.8965, loss = 0.0107
Step 15: x = 2.9839, loss = 0.0003

최종: x = 2.9975 (정답: 3)
```

---

## 신경망에서의 Gradient

### 전체 파라미터의 Gradient

신경망은 수백만 개의 파라미터가 있고, 각각의 gradient를 계산합니다:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),   # 200,960 파라미터
    nn.ReLU(),
    nn.Linear(256, 128),   # 32,896 파라미터
    nn.ReLU(),
    nn.Linear(128, 10)     # 1,290 파라미터
)

# 총 파라미터 수
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터: {total_params:,}")  # 235,146

# Forward & Backward
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), y)
loss.backward()

# 모든 파라미터의 gradient 확인
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: shape={list(param.shape)}, grad_norm={grad_norm:.4f}")
```

### Gradient의 크기가 중요한 이유

```python
# Gradient 크기 비교
for name, param in model.named_parameters():
    if 'weight' in name:
        avg_grad = param.grad.abs().mean().item()
        print(f"{name}: 평균 gradient = {avg_grad:.6f}")
```

**관찰**: 뒤쪽 레이어의 gradient가 앞쪽보다 큼 → **Vanishing Gradient 현상**

---

## Learning Rate의 역할

{{< figure src="/images/math/calculus/ko/learning-rate-effect.svg" caption="Learning Rate에 따른 학습 곡선: 너무 작으면 느리고, 적절하면 수렴, 너무 크면 발산" >}}

### Learning Rate가 너무 크면?

```python
x = torch.tensor([0.0], requires_grad=True)
lr = 2.0  # 너무 큼!

for step in range(10):
    loss = (x - 3) ** 2
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()
    print(f"Step {step}: x = {x.item():.2f}")
```

출력:
```
Step 0: x = 6.00   # 3을 지나침!
Step 1: x = -6.00  # 반대로 지나침!
Step 2: x = 18.00  # 발산!
...
```

### Learning Rate가 너무 작으면?

```python
x = torch.tensor([0.0], requires_grad=True)
lr = 0.001  # 너무 작음

for step in range(100):
    loss = (x - 3) ** 2
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()

print(f"100 step 후: x = {x.item():.4f}")  # 아직 0.5 근처...
```

### 적절한 Learning Rate

```
       Loss
        │
        │\
        │ \         ← lr 너무 작음 (느림)
        │  \
        │   \____   ← lr 적절함
        │    \
        │     \__   ← lr 약간 큼 (진동)
        │
        └────────── Iteration
```

---

## Gradient의 문제점들

### 1. Vanishing Gradient

깊은 네트워크에서 gradient가 점점 작아짐:

```python
# 10층 네트워크
model = nn.Sequential(*[nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())
                        for _ in range(10)])

x = torch.randn(1, 64)
y = torch.randn(1, 64)
loss = ((model(x) - y)**2).mean()
loss.backward()

# 각 층의 gradient 확인
for i, layer in enumerate(model):
    grad_norm = layer[0].weight.grad.norm().item()
    print(f"Layer {i}: gradient norm = {grad_norm:.8f}")
```

**해결책**: ReLU, BatchNorm, Skip Connection

### 2. Exploding Gradient

gradient가 폭발적으로 커짐:

```python
# gradient clipping으로 해결
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Saddle Point

gradient = 0인데 최솟값이 아님:

$$
f(x, y) = x^2 - y^2
$$

```python
x = torch.tensor([0.0], requires_grad=True)
y = torch.tensor([0.0], requires_grad=True)

f = x**2 - y**2
f.backward()

print(f"Gradient at (0,0): ({x.grad.item()}, {y.grad.item()})")  # (0, 0)
# 하지만 (0,0)은 최솟값이 아님! (안장점)
```

---

## 코드로 확인: 2D 최적화

```python
import torch

def rosenbrock(x, y):
    """로젠브록 함수: 최적화하기 어려운 대표적 함수"""
    return (1 - x)**2 + 100 * (y - x**2)**2

# 시작점
x = torch.tensor([-1.0], requires_grad=True)
y = torch.tensor([1.0], requires_grad=True)

lr = 0.001
history = []

for step in range(5000):
    loss = rosenbrock(x, y)
    loss.backward()

    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
        x.grad.zero_()
        y.grad.zero_()

    if step % 1000 == 0:
        print(f"Step {step}: ({x.item():.4f}, {y.item():.4f}), loss = {loss.item():.4f}")

print(f"\n최종: ({x.item():.4f}, {y.item():.4f})")
print(f"정답: (1.0, 1.0)")
```

---

## 핵심 정리

| 개념 | 정의 | 크기 | 딥러닝 적용 |
|------|------|------|------------|
| 편미분 | 한 변수에 대한 미분 | 스칼라 | 각 파라미터의 영향 |
| Gradient | 모든 편미분의 벡터 | n벡터 | 파라미터 업데이트 방향 |
| Jacobian | 벡터→벡터 함수의 미분 | m×n 행렬 | 레이어 변환, Flow 모델 |
| Hessian | 2차 미분 (곡률) | n×n 행렬 | 최적화 난이도, Adam |

## 핵심 통찰

1. **Gradient = 방향 + 크기**: 어디로, 얼마나 바꿔야 하는가
2. **항상 반대로 이동**: Loss를 줄이려면 -gradient
3. **Jacobian = 변환의 민감도**: 레이어가 입력을 어떻게 변형하는가
4. **Hessian = 곡률 정보**: 학습이 쉬운가 어려운가
5. **Learning Rate가 중요**: 너무 크면 발산, 너무 작으면 느림

---

## 다음 단계

Gradient는 한 레이어의 미분입니다. 하지만 신경망은 **여러 레이어가 연결**되어 있습니다.

→ [Chain Rule](/ko/docs/math/calculus/chain-rule): 연결된 함수의 gradient

## 관련 콘텐츠

- [미분 기초](/ko/docs/math/calculus/basics) - 단일 변수 미분
- [Chain Rule](/ko/docs/math/calculus/chain-rule) - 합성 함수의 미분
- [최적화 수학](/ko/docs/math/calculus/optimization) - Taylor 전개와 2차 최적화
- [SGD](/ko/docs/math/training/optimizer/sgd) - Gradient 기반 최적화
- [Adam](/ko/docs/math/training/optimizer/adam) - 적응적 학습률 (Hessian 근사)
