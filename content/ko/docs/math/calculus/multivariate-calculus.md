---
title: "다변수 미적분"
weight: 6
math: true
---

# 다변수 미적분 (Multivariate Calculus)

{{< figure src="/images/math/calculus/ko/jacobian-backprop.jpeg" caption="순전파와 역전파: 야코비안 행렬을 통한 VJP 체인이 Backpropagation의 본질" >}}

{{% hint info %}}
**선수지식**: [Gradient](/ko/docs/math/calculus/gradient) | [Chain Rule](/ko/docs/math/calculus/chain-rule)
{{% /hint %}}

> **한 줄 요약**: 야코비안은 "벡터 함수의 gradient"이고, 헤시안은 "gradient의 gradient"입니다. Backpropagation의 수학적 본질이 여기에 있습니다.

## 왜 다변수 미적분을 배워야 하나요?

### 문제 상황 1: "출력이 여러 개인 함수는 gradient가 어떻게 되나요?"

```python
# 입력 3차원 → 출력 2차원 함수
linear = nn.Linear(3, 2)
x = torch.randn(3)
y = linear(x)  # y는 2차원

# gradient가 스칼라가 아니라 행렬?
```

→ 입력 $n$개, 출력 $m$개이면 미분은 **$m \times n$ 행렬** = **야코비안(Jacobian)**

### 문제 상황 2: "Loss surface의 곡률은 어떻게 측정하나요?"

```python
# 어떤 방향은 가파르고, 어떤 방향은 완만한 Loss surface
# → 곡률 정보가 있으면 더 좋은 최적화가 가능
```

→ **헤시안(Hessian)** = 2차 도함수 행렬 = 곡률 정보

### 문제 상황 3: "loss.backward()의 내부 수학이 궁금해요"

```python
# 여러 층을 통과하는 gradient 계산
y = f3(f2(f1(x)))
loss = L(y, target)
loss.backward()
# 내부적으로 Jacobian들의 곱으로 gradient가 전파됨
```

→ Backpropagation = **야코비안의 연쇄 곱셈**

---

## 야코비안 행렬 (Jacobian Matrix)

### 정의

벡터 함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$의 야코비안:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

**크기**: $m \times n$ (출력 차원 × 입력 차원)

**각 행**: 하나의 출력 $f_i$에 대한 gradient
**각 열**: 하나의 입력 $x_j$가 모든 출력에 미치는 영향

### 특수한 경우들

| 입력 | 출력 | 미분 | 이름 | 크기 |
|------|------|------|------|------|
| 스칼라 | 스칼라 | $f'(x)$ | 도함수 | $1 \times 1$ |
| 벡터 | 스칼라 | $\nabla f$ | gradient | $1 \times n$ |
| 벡터 | 벡터 | $J$ | **야코비안** | $m \times n$ |

```python
import torch
from torch.autograd.functional import jacobian

# 벡터 함수: f(x) = [x₁² + x₂, x₁·x₂ + x₂²]
def f(x):
    return torch.stack([
        x[0]**2 + x[1],
        x[0] * x[1] + x[1]**2
    ])

x = torch.tensor([2.0, 3.0])

# PyTorch로 야코비안 계산
J = jacobian(f, x)
print(f"야코비안:\n{J}")
# [[2*x₁, 1    ],     [[4, 1],
#  [x₂,   x₁+2*x₂]]    [3, 8]]

# 수동 계산으로 검증
print(f"\n수동 계산:")
print(f"∂f₁/∂x₁ = 2·x₁ = {2*x[0]}")      # 4
print(f"∂f₁/∂x₂ = 1")                       # 1
print(f"∂f₂/∂x₁ = x₂ = {x[1]}")            # 3
print(f"∂f₂/∂x₂ = x₁ + 2·x₂ = {x[0] + 2*x[1]}")  # 8
```

### 딥러닝 적용: Linear Layer의 야코비안

```python
# y = Wx + b에서 야코비안은 무엇인가?
W = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])  # (2, 3)

def linear(x):
    return W @ x

x = torch.randn(3)
J = jacobian(linear, x)
print(f"야코비안:\n{J}")
print(f"\nW:\n{W}")
# 야코비안 = W 자체!
# ∂(Wx)/∂x = W
```

**핵심**: Linear Layer의 야코비안 = 가중치 행렬 $W$

---

## 다변수 Chain Rule과 Backpropagation

### Chain Rule의 행렬 형태

$\mathbf{y} = f(\mathbf{x})$, $\mathbf{z} = g(\mathbf{y})$이면:

$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = J_g \cdot J_f
$$

**야코비안의 곱 = 합성 함수의 야코비안**

```python
# 합성 함수: z = g(f(x))
def f(x):
    return torch.stack([x[0]*x[1], x[0]+x[1]**2])

def g(y):
    return torch.stack([y[0]+y[1], y[0]*y[1]])

def composed(x):
    return g(f(x))

x = torch.tensor([1.0, 2.0])

# 방법 1: 합성 함수의 야코비안 직접 계산
J_composed = jacobian(composed, x)

# 방법 2: 야코비안의 곱
J_f = jacobian(f, x)
y = f(x)
J_g = jacobian(g, y)
J_product = J_g @ J_f

print(f"직접 계산:\n{J_composed}")
print(f"\n곱으로 계산:\n{J_product}")
print(f"\n일치: {torch.allclose(J_composed, J_product)}")  # True
```

### Backpropagation = VJP (Vector-Jacobian Product)

실제 backprop은 야코비안을 **통째로** 계산하지 않습니다.

Loss가 스칼라이므로 우리가 필요한 것:

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot J_f = \mathbf{v}^T J_f
$$

→ **벡터 × 야코비안** 곱만 계산하면 됨 (VJP)

```python
# VJP: 야코비안 전체를 구하지 않고 벡터-야코비안 곱만 계산
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Forward
y = torch.stack([x[0]**2 + x[1], x[0] * x[1]])
loss = y.sum()  # 간단한 Loss

# Backward: ∂L/∂x = v^T · J_f  (v = ∂L/∂y = [1, 1])
loss.backward()

print(f"∂L/∂x = {x.grad}")
# ∂L/∂x₁ = ∂(x₁²+x₂)/∂x₁ + ∂(x₁x₂)/∂x₁ = 2x₁ + x₂ = 4
# ∂L/∂x₂ = ∂(x₁²+x₂)/∂x₂ + ∂(x₁x₂)/∂x₂ = 1 + x₁ = 2
```

### 왜 VJP가 효율적인가?

| 방법 | 필요한 것 | 비용 | 사용처 |
|------|----------|------|--------|
| 야코비안 전체 | $J$ ($m \times n$) | $O(mn)$ | 소규모 분석 |
| **VJP** | $\mathbf{v}^T J$ ($1 \times n$) | $O(n)$ | **Backprop** |
| JVP | $J\mathbf{v}$ ($m \times 1$) | $O(m)$ | Forward-mode AD |

→ 딥러닝에서 출력은 항상 스칼라 Loss → **VJP가 최적!**

---

## 헤시안 행렬 (Hessian Matrix)

### 정의

스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$의 2차 도함수 행렬:

$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

**크기**: $n \times n$ (입력 차원 × 입력 차원)

**의미**: Loss surface의 **곡률(curvature)** 정보

```python
from torch.autograd.functional import hessian

# f(x) = x₁² + 3x₁x₂ + 2x₂²
def f(x):
    return x[0]**2 + 3*x[0]*x[1] + 2*x[1]**2

x = torch.tensor([1.0, 2.0])
H = hessian(f, x)
print(f"헤시안:\n{H}")
# [[2, 3],    ← ∂²f/∂x₁² = 2, ∂²f/∂x₁∂x₂ = 3
#  [3, 4]]    ← ∂²f/∂x₂∂x₁ = 3, ∂²f/∂x₂² = 4
```

### 헤시안의 성질

| 성질 | 의미 | 딥러닝 해석 |
|------|------|------------|
| **대칭** ($H = H^T$) | 편미분 순서 무관 | 항상 성립 |
| **양정치** (고유값 > 0) | 볼록(convex) | 극소점 |
| **부정치** (고유값 < 0) | 오목(concave) | 극대점 |
| **부정부호** (혼합 부호) | 안장점(saddle) | 고차원에서 흔함 |

```python
# 안장점 예시: f(x, y) = x² - y²
def saddle(x):
    return x[0]**2 - x[1]**2

x = torch.tensor([0.0, 0.0])
H = hessian(saddle, x)
eigenvalues = torch.linalg.eigvalsh(H)
print(f"헤시안:\n{H}")        # [[2, 0], [0, -2]]
print(f"고유값: {eigenvalues}")  # [-2, 2] → 부정부호 → 안장점!
```

### 딥러닝 적용: 왜 안장점이 중요한가?

고차원 Loss surface에서:
- 극소점: 모든 방향에서 위로 볼록 (모든 고유값 > 0)
- 안장점: 어떤 방향은 위로, 어떤 방향은 아래로 (고유값 혼합)
- **고차원에서는 안장점이 극소점보다 훨씬 많음!**

```python
# 100차원에서 각 고유값이 ±1일 확률이 같다면:
# P(모든 고유값 > 0) = (1/2)^100 ≈ 0  (극소점이 거의 없음!)
# → SGD의 noise가 안장점을 탈출하는 데 도움
```

---

## 다변수 테일러 전개

### 2차까지의 근사

$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T H \Delta\mathbf{x}$

```python
# 다변수 2차 근사 예시
def f(x):
    return x[0]**2 + x[0]*x[1] + x[1]**2

x0 = torch.tensor([1.0, 1.0], requires_grad=True)

# 정확한 값
f_exact = f(torch.tensor([1.1, 1.2]))

# 2차 근사
f0 = f(x0)
f0.backward()
grad = x0.grad.clone()

H = hessian(f, x0.detach())
dx = torch.tensor([0.1, 0.2])

f_approx = f0.item() + grad @ dx + 0.5 * dx @ H @ dx
print(f"정확한 값: {f_exact:.6f}")
print(f"2차 근사:  {f_approx:.6f}")
print(f"오차:      {abs(f_exact - f_approx):.6f}")
```

### Adam의 Hessian 대각 근사

Adam optimizer는 Hessian의 **대각 원소만** 근사합니다:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
$$

- $m_t$ ≈ gradient (1차)
- $v_t$ ≈ gradient²의 이동 평균 ≈ **Hessian 대각 원소의 근사**
- $\frac{1}{\sqrt{v_t}}$가 각 파라미터별 **곡률의 역수** 역할

```python
# Adam vs SGD 비교: 곡률이 다른 함수에서
# f(x, y) = 100x² + y² (x방향이 100배 가파름)
def steep_loss(params):
    return 100 * params[0]**2 + params[1]**2

# SGD: 모든 방향에 같은 learning rate → 진동
# Adam: 가파른 방향은 작게, 완만한 방향은 크게 → 빠른 수렴
for opt_name in ['SGD', 'Adam']:
    params = torch.tensor([1.0, 1.0], requires_grad=True)
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD([params], lr=0.005)
    else:
        optimizer = torch.optim.Adam([params], lr=0.1)

    for step in range(100):
        optimizer.zero_grad()
        loss = steep_loss(params)
        loss.backward()
        optimizer.step()

    print(f"{opt_name:5s}: x={params[0].item():.4f}, y={params[1].item():.4f}, "
          f"loss={steep_loss(params).item():.4f}")
```

---

## 좌표 변환과 야코비안

### 변수 변환의 야코비안

확률분포에서 변수를 변환할 때 야코비안 행렬식이 필요합니다:

$$
p_Y(\mathbf{y}) = p_X(\mathbf{x}) \cdot |\det J^{-1}|
$$

```python
# 극좌표 변환: (r, θ) → (x, y) = (r·cos θ, r·sin θ)
import math

def polar_to_cartesian(rtheta):
    r, theta = rtheta[0], rtheta[1]
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)])

rtheta = torch.tensor([2.0, math.pi/4])
J = jacobian(polar_to_cartesian, rtheta)
det_J = torch.linalg.det(J)
print(f"야코비안:\n{J}")
print(f"|det(J)| = |{det_J.item():.4f}| = {abs(det_J.item()):.4f}")
# |det(J)| = r → 적분에서 r·dr·dθ가 나오는 이유!
```

**딥러닝 적용**: Normalizing Flow

```python
# Normalizing Flow: 간단한 분포를 복잡한 분포로 변환
# log p(x) = log p(z) - log|det(∂f/∂z)|
# 야코비안 행렬식이 확률 보정에 사용됨!
```

---

## 핵심 정리

| 개념 | 크기 | 의미 | 딥러닝 적용 |
|------|------|------|------------|
| Gradient | $1 \times n$ | 스칼라 함수의 미분 | optimizer.step() |
| **야코비안** | $m \times n$ | 벡터 함수의 미분 | Backprop의 본질 |
| **헤시안** | $n \times n$ | 2차 도함수 | 곡률, Adam |
| VJP | $1 \times n$ | 벡터 × 야코비안 | 실제 backprop 구현 |
| det(J) | 스칼라 | 부피 변화율 | Normalizing Flow |

## 핵심 통찰

1. **Backprop = VJP의 연쇄**: 야코비안 전체 대신 벡터 곱만 계산 → 효율적
2. **Hessian = 곡률**: 가파른 방향은 조심히, 완만한 방향은 과감히 → Adam
3. **고차원 = 안장점 천국**: 극소점보다 안장점이 훨씬 많음 → SGD noise가 도움
4. **det(J) = 부피 변화**: 확률분포의 변수 변환에 필수 → Normalizing Flow

---

## 다음 단계

다변수 미적분의 체계를 이해했습니다. 이제 이 도구들을 활용한 **역전파 알고리즘**을 자세히 봅시다.

→ [Backpropagation](/ko/docs/math/calculus/backpropagation): VJP를 활용한 효율적 gradient 계산

## 관련 콘텐츠

- [Gradient](/ko/docs/math/calculus/gradient) — 야코비안의 특수한 경우
- [Chain Rule](/ko/docs/math/calculus/chain-rule) — 야코비안 곱의 직관
- [Backpropagation](/ko/docs/math/calculus/backpropagation) — VJP의 실제 구현
- [최적화 수학](/ko/docs/math/calculus/optimization) — Hessian과 2차 최적화
