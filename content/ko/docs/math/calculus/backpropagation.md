---
title: "Backpropagation"
weight: 3
math: true
---

# Backpropagation (역전파)

## 개요

역전파는 Chain Rule을 효율적으로 적용하여 모든 파라미터의 gradient를 계산하는 알고리즘입니다. 딥러닝 학습의 핵심입니다.

## 왜 역전파인가?

신경망의 파라미터 수: 수백만 ~ 수십억 개

각 파라미터에 대해 손실의 gradient를 개별 계산하면 매우 비효율적입니다. 역전파는 **중간 계산을 재사용**하여 O(파라미터 수)로 모든 gradient를 한 번에 계산합니다.

## 알고리즘

### 1. 순전파 (Forward Pass)

입력에서 출력까지 계산하며 중간값 저장:

```
x → [Layer 1] → h₁ → [Layer 2] → h₂ → ... → ŷ → [Loss] → L
        ↓             ↓
      W₁ 저장       W₂ 저장
```

### 2. 역전파 (Backward Pass)

출력에서 입력 방향으로 gradient 계산:

```
dL/dL=1 ← dL/dŷ ← dL/dh₂ ← dL/dh₁ ← ...
                     ↓          ↓
                  dL/dW₂     dL/dW₁
```

## 수식

Layer l의 gradient 계산:

$$
\delta_l = \frac{\partial L}{\partial h_l} = \delta_{l+1} \cdot \frac{\partial h_{l+1}}{\partial h_l}
$$

가중치의 gradient:

$$
\frac{\partial L}{\partial W_l} = \delta_l \cdot \frac{\partial h_l}{\partial W_l}
$$

## 구현 (순수 Python)

```python
import numpy as np

class SimpleNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01

    def forward(self, x):
        # 중간값 저장 (역전파에 필요)
        self.x = x
        self.h = np.maximum(0, x @ self.W1)  # ReLU
        self.y = self.h @ self.W2
        return self.y

    def backward(self, dL_dy):
        # Layer 2 gradient
        dL_dW2 = self.h.T @ dL_dy
        dL_dh = dL_dy @ self.W2.T

        # ReLU backward
        dL_dh[self.h <= 0] = 0

        # Layer 1 gradient
        dL_dW1 = self.x.T @ dL_dh

        return dL_dW1, dL_dW2

# 사용 예시
net = SimpleNetwork(10, 64, 1)
x = np.random.randn(32, 10)
y_true = np.random.randn(32, 1)

# Forward
y_pred = net.forward(x)
loss = np.mean((y_pred - y_true) ** 2)

# Backward
dL_dy = 2 * (y_pred - y_true) / len(y_pred)  # MSE gradient
dW1, dW2 = net.backward(dL_dy)

# Update (SGD)
lr = 0.01
net.W1 -= lr * dW1
net.W2 -= lr * dW2
```

## PyTorch의 자동 미분

PyTorch는 연산 그래프를 자동으로 구성하고 역전파를 수행합니다:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Forward
pred = model(x)
loss = nn.MSELoss()(pred, y)

# Backward (자동)
loss.backward()

# 모든 파라미터의 gradient가 계산됨
for name, param in model.named_parameters():
    print(f"{name}: {param.grad.shape}")
```

## 계산 그래프

PyTorch는 연산을 그래프로 기록합니다:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3 + 1

# 그래프: x → (square) → y → (mul 3) → (add 1) → z
z.backward()
print(x.grad)  # dz/dx = d(3x²+1)/dx = 6x = 12
```

## 메모리 효율화: Gradient Checkpointing

중간 활성화를 저장하지 않고 필요할 때 재계산:

```python
from torch.utils.checkpoint import checkpoint

class EfficientModel(nn.Module):
    def forward(self, x):
        # 메모리 절약: 역전파 시 재계산
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        return x
```

## 관련 콘텐츠

- [Gradient](/ko/docs/math/calculus/gradient) - 미분의 기초
- [Chain Rule](/ko/docs/math/calculus/chain-rule) - 역전파의 수학적 기반
- [SGD](/ko/docs/math/training/optimizer/sgd) - 계산된 gradient 활용
- [Adam](/ko/docs/math/training/optimizer/adam) - 적응적 학습률
