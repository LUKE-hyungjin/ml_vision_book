---
title: "Backpropagation"
weight: 4
math: true
---

# Backpropagation (역전파)

> **한 줄 요약**: Backpropagation은 수백만 파라미터의 gradient를 **효율적으로** 계산하는 알고리즘입니다.

## 왜 Backpropagation을 배워야 하나요?

### 문제 상황 1: "loss.backward()는 어떻게 작동하나요?"

```python
loss = criterion(model(x), y)
loss.backward()  # ← 이 한 줄이 어떻게 수백만 gradient를 계산할까?
```

→ **Backpropagation** 알고리즘이 모든 파라미터의 gradient를 한 번에 계산합니다.

### 문제 상황 2: "왜 파라미터마다 따로 미분하면 안 되나요?"

GPT-3: 1750억 개 파라미터

- **순진한 방법**: 각 파라미터마다 수치 미분 → 1750억 × 2회 forward pass → **불가능**
- **Backpropagation**: 1회 forward + 1회 backward → **2회로 끝!**

### 문제 상황 3: "Custom Layer를 만들었는데 학습이 안 됩니다"

```python
class MyLayer(nn.Module):
    def forward(self, x):
        # 이상한 연산...
        return result
    # backward를 정의 안 함 → gradient가 흐르지 않음!
```

→ Backpropagation을 이해해야 custom layer를 제대로 만들 수 있습니다.

---

## Backpropagation의 핵심 아이디어

### 효율성의 비밀: 중간 계산 재사용

**비효율적인 방법** (수치 미분):
```
∂L/∂w₁: 전체 네트워크 2번 계산
∂L/∂w₂: 전체 네트워크 2번 계산
∂L/∂w₃: 전체 네트워크 2번 계산
...
```
→ 파라미터 N개면 **2N번** forward pass

**효율적인 방법** (Backpropagation):
```
Forward: 입력 → 출력 (중간값 저장)
Backward: 출력 → 입력 (중간값 재사용)
```
→ 파라미터 수와 **무관하게 2번**

### 비유: 공장 조립 라인

**Forward Pass** = 조립 라인
```
부품 → [공정1] → 중간품A → [공정2] → 중간품B → [공정3] → 완제품
        저장         저장          저장
```

**Backward Pass** = 불량 원인 추적
```
완제품 불량 ← [공정3] ← 중간품B ← [공정2] ← 중간품A ← [공정1] ← 부품
              ↓재사용     ↓재사용      ↓재사용
```

각 공정의 중간 결과를 **저장해뒀으니** 빠르게 추적 가능!

---

## 알고리즘 단계별 이해

{{< figure src="/images/math/calculus/ko/forward-backward-pass.svg" caption="Backpropagation 3단계: Forward Pass → Backward Pass → Parameter Update" >}}

### 1단계: Forward Pass

입력에서 출력까지 계산하며 **중간값 저장**:

```python
def forward(x, W1, W2):
    # 각 단계 저장
    z1 = x @ W1              # 저장: x, W1, z1
    a1 = relu(z1)            # 저장: z1, a1
    z2 = a1 @ W2             # 저장: a1, W2, z2
    return z2

# 시각화
#  x  →  [W1]  → z1 → [ReLU] → a1 → [W2] → z2 → [Loss] → L
#  ↓저장   ↓저장    ↓저장        ↓저장    ↓저장    ↓저장
```

### 2단계: Backward Pass

출력에서 입력 방향으로 Chain Rule 적용:

```python
def backward(dL_dz2, cache):
    x, W1, z1, a1, W2, z2 = cache

    # ∂L/∂W2
    dL_dW2 = a1.T @ dL_dz2

    # ∂L/∂a1
    dL_da1 = dL_dz2 @ W2.T

    # ∂L/∂z1 (ReLU backward)
    dL_dz1 = dL_da1 * (z1 > 0)

    # ∂L/∂W1
    dL_dW1 = x.T @ dL_dz1

    return dL_dW1, dL_dW2
```

**핵심**: 한 번 계산한 $\frac{\partial L}{\partial a_1}$을 **재사용**하여 $\frac{\partial L}{\partial W_1}$과 $\frac{\partial L}{\partial z_1}$ 모두 계산!

---

## 완전한 예시: 2층 네트워크

```python
import numpy as np

class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 가중치 초기화
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        """Forward pass: 중간값 저장"""
        self.x = x

        # Layer 1
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = self.z2

        return self.out

    def backward(self, dL_dout):
        """Backward pass: Chain Rule 적용"""
        batch_size = self.x.shape[0]

        # Layer 2 gradients
        self.dW2 = self.a1.T @ dL_dout / batch_size
        self.db2 = np.mean(dL_dout, axis=0)
        dL_da1 = dL_dout @ self.W2.T

        # ReLU backward
        dL_dz1 = dL_da1 * (self.z1 > 0)

        # Layer 1 gradients
        self.dW1 = self.x.T @ dL_dz1 / batch_size
        self.db1 = np.mean(dL_dz1, axis=0)

        return self.dW1, self.db1, self.dW2, self.db2

    def update(self, lr):
        """Gradient descent update"""
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


# 학습 예시
np.random.seed(42)
net = TwoLayerNet(2, 64, 1)

# 간단한 데이터: y = x1 + x2
X = np.random.randn(100, 2)
y = (X[:, 0:1] + X[:, 1:2])

print("=== 학습 시작 ===")
for epoch in range(1000):
    # Forward
    pred = net.forward(X)

    # Loss (MSE)
    loss = np.mean((pred - y) ** 2)

    # Backward
    dL_dout = 2 * (pred - y) / len(y)  # MSE gradient
    net.backward(dL_dout)

    # Update
    net.update(lr=0.1)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

print(f"\n최종 Loss: {loss:.6f}")
```

---

## PyTorch의 Autograd

PyTorch는 연산 **그래프**를 자동으로 만들고 역전파합니다.

{{< figure src="/images/math/calculus/ko/computation-graph.svg" caption="계산 그래프: 각 노드의 local gradient를 역방향으로 곱해 최종 gradient 계산" >}}

### 계산 그래프란?

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 연산 = 그래프 노드
z = x * y        # 곱셈 노드
w = z + x        # 덧셈 노드
loss = w ** 2    # 제곱 노드

# 그래프 시각화:
#     x ──┬──→ [*] ──→ z ──→ [+] ──→ w ──→ [²] ──→ loss
#         │              ↗
#     y ──┘          x ──┘
```

### 자동 역전파

```python
loss.backward()

print(f"∂loss/∂x = {x.grad.item()}")
print(f"∂loss/∂y = {y.grad.item()}")

# 해석적 계산:
# w = xy + x = x(y + 1)
# loss = w²
# ∂loss/∂w = 2w = 2 * 8 = 16
# ∂w/∂x = y + 1 = 4 (x가 두 경로로 영향)
# ∂loss/∂x = 16 * 4 = 64
# ∂w/∂y = x = 2
# ∂loss/∂y = 16 * 2 = 32
```

### 실행 결과 확인

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

z = x * y        # 6
w = z + x        # 8
loss = w ** 2    # 64

loss.backward()

print(f"z = {z.item()}, w = {w.item()}, loss = {loss.item()}")
print(f"∂loss/∂x = {x.grad.item()}")  # 64
print(f"∂loss/∂y = {y.grad.item()}")  # 32
```

---

## 각 연산의 Backward

### Linear (행렬 곱)

Forward: $Y = XW + b$

Backward:
$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T
$$
$$
\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}
$$
$$
\frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial Y}
$$

```python
class LinearBackward:
    @staticmethod
    def forward(x, W, b):
        return x @ W + b

    @staticmethod
    def backward(dL_dy, x, W):
        dL_dx = dL_dy @ W.T
        dL_dW = x.T @ dL_dy
        dL_db = dL_dy.sum(axis=0)
        return dL_dx, dL_dW, dL_db
```

### ReLU

Forward: $Y = \max(0, X)$

Backward:
$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot \mathbf{1}_{X > 0}
$$

```python
class ReLUBackward:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(dL_dy, x):
        return dL_dy * (x > 0)
```

### Softmax + Cross-Entropy

Forward: $L = -\sum y_i \log(\text{softmax}(z)_i)$

Backward (결합하면 간단):
$$
\frac{\partial L}{\partial z_i} = \text{softmax}(z)_i - y_i
$$

```python
class SoftmaxCrossEntropyBackward:
    @staticmethod
    def forward(logits, targets):
        probs = softmax(logits)
        loss = -np.sum(targets * np.log(probs + 1e-10))
        return loss, probs

    @staticmethod
    def backward(probs, targets):
        # 놀랍도록 간단!
        return probs - targets
```

**핵심 인사이트**: Softmax + Cross-Entropy의 gradient는 단순히 **(예측 확률 - 정답)**!

---

## 메모리와 계산 트레이드오프

### 문제: 중간값 저장 = 메모리 사용

```python
# GPT-3 학습 시 활성화값 메모리
batch_size = 8
seq_len = 2048
hidden = 12288
layers = 96

# 각 층마다 저장해야 하는 활성화값
activation_memory = batch_size * seq_len * hidden * layers * 4  # float32
print(f"활성화 메모리: {activation_memory / 1e9:.1f} GB")  # 수백 GB!
```

### 해결책 1: Gradient Checkpointing

일부 층의 중간값만 저장, 나머지는 **재계산**:

```python
from torch.utils.checkpoint import checkpoint

class EfficientBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # 메모리 절약: forward 중간값 저장 안 함
        # backward 시 재계산
        return checkpoint(self.block, x, use_reentrant=False)
```

**트레이드오프**:
- 메모리 ↓ (중간값 저장 안 함)
- 계산 ↑ (backward 시 재계산)

### 해결책 2: Mixed Precision

Float16으로 계산하여 메모리 절반:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()

    with autocast():  # Float16 사용
        output = model(x)
        loss = criterion(output, y)

    scaler.scale(loss).backward()  # 스케일링된 backward
    scaler.step(optimizer)
    scaler.update()
```

---

## Custom Autograd Function

직접 forward/backward 정의:

```python
import torch
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx: backward에서 사용할 정보 저장
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # 저장한 정보 불러오기
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 사용
my_relu = MyReLU.apply
x = torch.randn(5, requires_grad=True)
y = my_relu(x)
y.sum().backward()
print(f"x: {x.data}")
print(f"grad: {x.grad}")
```

### Straight-Through Estimator (STE)

Forward와 Backward가 다른 함수 (Quantization에서 사용):

```python
class StraightThroughEstimator(Function):
    """forward는 round, backward는 identity"""
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Round의 gradient는 0이지만, 그냥 통과시킴
        return grad_output

# Binary Neural Network에서 사용
class BinaryWeight(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)  # +1 또는 -1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # |input| <= 1인 경우만 gradient 전파
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input
```

---

## Gradient 흐름 디버깅

### Gradient 확인하기

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Forward & Backward
output = model(x)
loss = nn.MSELoss()(output, y)
loss.backward()

# Gradient 분석
print("=== Gradient 분석 ===")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad = param.grad
        print(f"{name}:")
        print(f"  shape: {list(grad.shape)}")
        print(f"  mean:  {grad.mean().item():.6f}")
        print(f"  std:   {grad.std().item():.6f}")
        print(f"  max:   {grad.abs().max().item():.6f}")
        print(f"  zeros: {(grad == 0).sum().item()}/{grad.numel()}")
```

### Hook으로 중간 gradient 확인

```python
def gradient_hook(name):
    def hook(grad):
        print(f"{name}: mean={grad.mean():.4f}, std={grad.std():.4f}")
        return grad
    return hook

# 중간 activation에 hook 등록
activations = {}

def forward_hook(name):
    def hook(module, input, output):
        activations[name] = output
        output.register_hook(gradient_hook(name))
    return hook

# 모든 ReLU에 hook 등록
for name, module in model.named_modules():
    if isinstance(module, nn.ReLU):
        module.register_forward_hook(forward_hook(name))

# Forward & Backward
output = model(torch.randn(32, 10))
loss = output.sum()
loss.backward()
```

---

## 수치 미분으로 검증

Backpropagation이 정확한지 확인:

```python
def numerical_gradient(f, x, eps=1e-5):
    """수치 미분으로 gradient 계산"""
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus.view(-1)[i] += eps
        x_minus.view(-1)[i] -= eps
        grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# 테스트
def f(W):
    return ((x @ W) ** 2).sum()

W = torch.randn(10, 5, requires_grad=True)
x = torch.randn(3, 10)

# 자동 미분
loss = f(W)
loss.backward()
auto_grad = W.grad.clone()

# 수치 미분
num_grad = numerical_gradient(f, W)

# 비교
diff = (auto_grad - num_grad).abs().max()
print(f"최대 차이: {diff.item():.10f}")
print(f"검증 통과: {diff.item() < 1e-4}")
```

---

## 핵심 정리

| 단계 | 하는 일 | 저장하는 것 |
|------|---------|------------|
| Forward | 입력 → 출력 계산 | 중간 활성화값 |
| Backward | 출력 → 입력 gradient | 파라미터 gradient |

| 연산 | Forward | Backward |
|------|---------|----------|
| Linear (Y=XW) | Y = X @ W | dX = dY @ W.T, dW = X.T @ dY |
| ReLU | max(0, X) | dY * (X > 0) |
| Softmax+CE | -log(softmax) | softmax - target |

## 핵심 통찰

1. **효율성**: 중간값 재사용으로 O(N)에서 O(1)로
2. **자동화**: PyTorch가 그래프 구축 + 역전파 자동 수행
3. **메모리 트레이드오프**: Checkpointing으로 메모리 ↓ 계산 ↑
4. **디버깅**: Hook으로 gradient 흐름 확인

---

## 요약: 딥러닝 학습의 전체 그림

```
┌─────────────────────────────────────────────────────────────┐
│                    Forward Pass                              │
│  x → [Layer 1] → a₁ → [Layer 2] → a₂ → ... → ŷ → [Loss] → L │
│        ↓저장        ↓저장                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Backward Pass                             │
│  ∂L/∂W₁ ← [Layer 1] ← ∂L/∂a₁ ← [Layer 2] ← ... ← ∂L/∂ŷ ← 1 │
│             ↑재사용           ↑재사용                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Parameter Update                          │
│           W ← W - η · ∂L/∂W  (Gradient Descent)             │
└─────────────────────────────────────────────────────────────┘
```

이제 `loss.backward()`가 무엇을 하는지 완전히 이해했습니다!

---

## 관련 콘텐츠

- [미분 기초](/ko/docs/math/calculus/basics) - 미분의 기본 개념
- [Gradient](/ko/docs/math/calculus/gradient) - 다변수 미분
- [Chain Rule](/ko/docs/math/calculus/chain-rule) - 역전파의 수학적 기반
- [SGD](/ko/docs/math/training/optimizer/sgd) - 계산된 gradient 활용
- [Adam](/ko/docs/math/training/optimizer/adam) - 적응적 최적화
