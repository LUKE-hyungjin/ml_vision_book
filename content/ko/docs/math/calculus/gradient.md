---
title: "Gradient"
weight: 1
math: true
---

# Gradient (기울기)

## 개요

Gradient는 다변수 함수가 가장 빠르게 증가하는 방향을 나타내는 벡터입니다. 딥러닝에서는 손실 함수를 최소화하기 위해 gradient의 **반대 방향**으로 이동합니다.

## 정의

스칼라 함수 f(x₁, x₂, ..., xₙ)의 gradient:

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right)
$$

### 편미분

각 변수에 대해 다른 변수를 상수로 취급하고 미분:

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}
$$

## 예시

$f(x, y) = x^2 + 2xy + y^2$ 일 때:

$$
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) = (2x + 2y, 2x + 2y)
$$

## Gradient Descent

파라미터를 gradient의 반대 방향으로 업데이트:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

- θ: 모델 파라미터
- η: 학습률 (learning rate)
- L: 손실 함수

### 직관적 이해

- Gradient는 언덕을 오르는 가장 가파른 방향
- 손실을 최소화하려면 **내려가야** 함
- 따라서 gradient의 **반대** 방향으로 이동

## 구현

```python
import torch

# 간단한 함수: f(x, y) = x^2 + y^2
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)

f = x**2 + y**2  # f = 25
f.backward()     # gradient 계산

print(f"df/dx = {x.grad}")  # 6.0 (= 2*3)
print(f"df/dy = {y.grad}")  # 8.0 (= 2*4)
```

### 신경망에서의 Gradient

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# 순전파
pred = model(x)
loss = nn.MSELoss()(pred, y)

# 역전파 - 모든 파라미터의 gradient 계산
loss.backward()

# gradient 확인
print(f"Weight gradient shape: {model.weight.grad.shape}")
print(f"Bias gradient shape: {model.bias.grad.shape}")
```

## Gradient의 기하학적 의미

- **크기**: 함수 변화의 민감도
- **방향**: 가장 빠르게 증가하는 방향
- **Gradient = 0**: 극점 (극소, 극대, 안장점)

## 관련 콘텐츠

- [Chain Rule](/ko/docs/math/calculus/chain-rule) - 합성 함수의 gradient
- [Backpropagation](/ko/docs/math/calculus/backpropagation) - 효율적 gradient 계산
- [SGD](/ko/docs/math/training/optimizer/sgd) - Gradient 기반 최적화
