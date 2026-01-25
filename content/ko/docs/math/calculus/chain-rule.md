---
title: "Chain Rule"
weight: 2
math: true
---

# Chain Rule (연쇄 법칙)

## 개요

합성 함수의 미분을 계산하는 규칙으로, 역전파(Backpropagation)의 수학적 기반입니다.

## 정의

y = f(g(x)) 일 때:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

### 다변수 확장

z = f(x, y)이고 x = x(t), y = y(t) 일 때:

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}
$$

## 직관적 이해

Chain Rule은 **기여도의 곱**으로 이해할 수 있습니다:

- x가 1 변할 때 g가 얼마나 변하는가? → dg/dx
- g가 1 변할 때 y가 얼마나 변하는가? → dy/dg
- x가 1 변할 때 y가 얼마나 변하는가? → (dy/dg) × (dg/dx)

## 예시

$y = (3x + 2)^4$ 의 미분:

1. u = 3x + 2 로 치환
2. y = u⁴

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 4u^3 \cdot 3 = 12(3x+2)^3
$$

## 신경망에서의 Chain Rule

신경망은 합성 함수입니다:

$$
\hat{y} = f_L(f_{L-1}(\cdots f_1(x)))
$$

손실 L에 대한 첫 번째 층 가중치 W₁의 gradient:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_{L-1}} \cdot \ldots \cdot \frac{\partial h_1}{\partial W_1}
$$

## 구현

```python
import torch

# y = (3x + 2)^4
x = torch.tensor([1.0], requires_grad=True)

u = 3 * x + 2       # u = 5
y = u ** 4          # y = 625

y.backward()
print(f"dy/dx = {x.grad}")  # 1500.0 = 12 * (3*1+2)^3 = 12 * 125

# 수동 계산 검증
du_dx = 3
dy_du = 4 * (3 * 1 + 2) ** 3  # 4 * 125 = 500
print(f"Chain rule: {dy_du * du_dx}")  # 1500.0
```

### 2층 신경망 예시

```python
import torch
import torch.nn as nn

# 2층 네트워크: x -> h -> y
x = torch.randn(1, 10)
W1 = torch.randn(10, 5, requires_grad=True)
W2 = torch.randn(5, 1, requires_grad=True)
target = torch.tensor([[1.0]])

# 순전파
h = torch.relu(x @ W1)  # 첫 번째 층
y = h @ W2               # 두 번째 층
loss = (y - target) ** 2

# 역전파 - PyTorch가 Chain Rule 자동 적용
loss.backward()

# dL/dW1 = dL/dy * dy/dh * dh/dW1 (Chain Rule)
print(f"W1 gradient: {W1.grad.shape}")
print(f"W2 gradient: {W2.grad.shape}")
```

## Vanishing/Exploding Gradient

Chain Rule의 연속 적용으로 발생하는 문제:

$$
\frac{\partial L}{\partial W_1} = \prod_{i=1}^{L} \frac{\partial h_i}{\partial h_{i-1}} \cdot \frac{\partial L}{\partial \hat{y}}
$$

- 각 항이 < 1 이면: **Vanishing Gradient** (기울기 소실)
- 각 항이 > 1 이면: **Exploding Gradient** (기울기 폭발)

### 해결책
- [ResNet](/ko/docs/architecture/cnn/resnet) - Skip Connection
- [Batch Normalization](/ko/docs/math/normalization/batch-norm)
- [LSTM/GRU](/ko/docs/architecture) - Gate 메커니즘

## 관련 콘텐츠

- [Gradient](/ko/docs/math/calculus/gradient) - 기본 미분
- [Backpropagation](/ko/docs/math/calculus/backpropagation) - Chain Rule의 적용
- [ResNet](/ko/docs/architecture/cnn/resnet) - Gradient 흐름 개선
