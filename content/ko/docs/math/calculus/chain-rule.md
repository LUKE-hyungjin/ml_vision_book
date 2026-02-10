---
title: "Chain Rule"
weight: 5
math: true
---

# Chain Rule (연쇄 법칙)

> **한 줄 요약**: Chain Rule은 "연결된 함수의 미분"을 계산합니다. 이것이 없으면 딥러닝은 불가능합니다.

## 왜 Chain Rule을 배워야 하나요?

### 문제 상황 1: "신경망은 함수가 연결된 것인데, 어떻게 미분하죠?"

```python
# 신경망 = 함수들의 연결
output = softmax(linear2(relu(linear1(x))))
#        ↑       ↑      ↑      ↑
#       f₄      f₃     f₂     f₁
```

- `linear1`의 가중치가 최종 `output`에 어떤 영향을 주는지 알아야 합니다
- 중간에 `relu`, `linear2`, `softmax`가 있어서 **직접 연결되어 있지 않습니다**
- → **Chain Rule**로 연결 고리를 따라가며 미분!

### 문제 상황 2: "왜 깊은 네트워크는 학습이 안 되나요?"

```python
# 100층 네트워크
model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(100)])
# 첫 번째 층의 gradient가 거의 0...
```

→ Chain Rule이 100번 적용되면서 gradient가 사라집니다 (**Vanishing Gradient**)

### 문제 상황 3: "ResNet의 Skip Connection은 왜 효과적인가요?"

```python
# ResNet의 핵심
output = F.relu(self.conv(x) + x)  # ← 왜 +x를 더하면 학습이 잘 될까?
```

→ Chain Rule 관점에서 **gradient가 직접 흐르는 경로**가 생기기 때문입니다.

---

## Chain Rule이란?

{{< figure src="/images/math/calculus/ko/chain-rule-flow.png" caption="Chain Rule: 환율 변환처럼 연결된 미분을 곱하면 전체 미분이 된다" >}}

### 핵심 아이디어: 연결 고리 따라가기

**상황**: $y$가 $u$에 의존하고, $u$가 $x$에 의존

$$
x \rightarrow u \rightarrow y
$$

**질문**: $x$가 변하면 $y$가 얼마나 변하는가?

**답**:
1. $x$가 1 변할 때 $u$가 얼마나 변하는가? → $\frac{du}{dx}$
2. $u$가 1 변할 때 $y$가 얼마나 변하는가? → $\frac{dy}{du}$
3. $x$가 1 변할 때 $y$가 얼마나 변하는가? → **곱하면 됩니다!**

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

### 직관적 비유: 환율 변환

- 달러 → 엔: 1달러 = 150엔
- 엔 → 원: 1엔 = 9원
- **달러 → 원**: 1달러 = 150 × 9 = **1350원**

Chain Rule도 똑같습니다:
- x → u: $\frac{du}{dx}$ 배
- u → y: $\frac{dy}{du}$ 배
- **x → y**: $\frac{dy}{du} \times \frac{du}{dx}$ 배

---

## 수학적 정의

### 단일 변수 Chain Rule

$y = f(g(x))$ 일 때:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(g(x)) \cdot g'(x)
$$

여기서 $u = g(x)$

### 예시 1: $(3x + 2)^4$

**분해**:
- 안쪽 함수: $u = g(x) = 3x + 2$
- 바깥 함수: $y = f(u) = u^4$

**Chain Rule 적용**:
$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 4u^3 \cdot 3 = 12(3x+2)^3
$$

```python
import torch

x = torch.tensor([1.0], requires_grad=True)

u = 3*x + 2      # u = 5
y = u ** 4       # y = 625

y.backward()

print(f"dy/dx = {x.grad.item()}")  # 1500
print(f"검증: 12 * (3*1+2)³ = 12 * 125 = {12 * 125}")  # 1500
```

### 예시 2: $\sin(x^2)$

**분해**:
- 안쪽: $u = x^2$
- 바깥: $y = \sin(u)$

**Chain Rule**:
$$
\frac{dy}{dx} = \cos(x^2) \cdot 2x
$$

```python
x = torch.tensor([2.0], requires_grad=True)

y = torch.sin(x**2)  # sin(4)

y.backward()

print(f"dy/dx = {x.grad.item():.4f}")
print(f"검증: cos(4) * 4 = {torch.cos(torch.tensor(4.0)).item() * 4:.4f}")
```

---

## 다변수 Chain Rule

### 여러 입력, 여러 중간값

$z = f(x, y)$이고, $x = x(t)$, $y = y(t)$일 때:

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}
$$

**직관**: 모든 경로의 기여를 **더합니다**

```
    x(t) ─────┐
              ├──→ z
    y(t) ─────┘

dz/dt = (z가 x를 통해 받는 영향) + (z가 y를 통해 받는 영향)
```

### 예시: $z = x^2 y$, where $x = t^2$, $y = t^3$

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}
$$

$$
= 2xy \cdot 2t + x^2 \cdot 3t^2
$$

$t=1$일 때 ($x=1$, $y=1$):

$$
\frac{dz}{dt} = 2(1)(1)(2) + (1)^2(3) = 4 + 3 = 7
$$

```python
t = torch.tensor([1.0], requires_grad=True)

x = t ** 2
y = t ** 3
z = x**2 * y

z.backward()

print(f"dz/dt = {t.grad.item()}")  # 7.0
```

---

## 신경망에서의 Chain Rule

### 2층 네트워크

```
입력 x → [Linear + ReLU] → h → [Linear] → output → [Loss] → L
         ↑                      ↑
        W₁                     W₂
```

$\frac{\partial L}{\partial W_1}$을 구하려면 Chain Rule 연속 적용:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial h} \cdot \frac{\partial h}{\partial W_1}
$$

### 코드로 확인

```python
import torch
import torch.nn as nn

# 2층 네트워크 수동 구현
class TwoLayerNet:
    def __init__(self):
        self.W1 = torch.randn(2, 3, requires_grad=True)
        self.W2 = torch.randn(3, 1, requires_grad=True)

    def forward(self, x):
        self.x = x
        self.h = torch.relu(x @ self.W1)  # 중간값 저장
        self.out = self.h @ self.W2
        return self.out

# Forward
net = TwoLayerNet()
x = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[5.0]])

output = net.forward(x)
loss = (output - target) ** 2

# Backward - PyTorch가 Chain Rule 자동 적용
loss.backward()

print("W1의 gradient shape:", net.W1.grad.shape)
print("W2의 gradient shape:", net.W2.grad.shape)

# Chain Rule 수동 검증
print("\n=== Chain Rule 수동 검증 ===")
dL_dout = 2 * (output - target)  # Loss → output
dout_dh = net.W2.T               # output → h
dL_dh = dL_dout @ dout_dh        # Loss → h

# ReLU gradient (h > 0이면 1, 아니면 0)
dh_drelu = (net.h > 0).float()
dL_drelu = dL_dh * dh_drelu      # Loss → ReLU 출력

# W1에 대한 gradient
dL_dW1_manual = x.T @ dL_drelu

print(f"PyTorch: {net.W1.grad[0, 0].item():.4f}")
print(f"수동 계산: {dL_dW1_manual[0, 0].item():.4f}")
```

---

## Vanishing Gradient 문제

{{< figure src="/images/math/calculus/ko/vanishing-gradient.png" caption="Sigmoid, ReLU, ResNet의 Gradient 흐름 비교" >}}

### 원인: Chain Rule의 연속 곱셈

L개 층이 있을 때:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdot \ldots \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}
$$

**문제**: 각 항이 1보다 작으면?

$$
0.5 \times 0.5 \times 0.5 \times \ldots = 0.5^L \approx 0
$$

### Sigmoid의 문제

Sigmoid의 미분: $\sigma'(x) = \sigma(x)(1-\sigma(x))$

최댓값이 0.25 (x=0일 때):

```python
import torch.nn.functional as F

# Sigmoid 미분의 최댓값
x = torch.linspace(-5, 5, 100)
sigmoid = torch.sigmoid(x)
sigmoid_grad = sigmoid * (1 - sigmoid)

print(f"Sigmoid gradient 최댓값: {sigmoid_grad.max().item():.4f}")  # 0.25
```

10층 네트워크라면: $0.25^{10} = 9.5 \times 10^{-7}$ → 거의 0!

### 실험: 깊은 네트워크의 gradient

```python
import torch
import torch.nn as nn

def check_gradient_flow(activation, num_layers=10):
    """깊은 네트워크의 gradient 흐름 확인"""
    layers = []
    for _ in range(num_layers):
        layers.extend([nn.Linear(64, 64), activation()])
    model = nn.Sequential(*layers)

    x = torch.randn(1, 64)
    y = torch.randn(1, 64)
    loss = ((model(x) - y)**2).mean()
    loss.backward()

    # 각 층의 gradient 확인
    grads = []
    for i, layer in enumerate(model):
        if hasattr(layer, 'weight'):
            grads.append(layer.weight.grad.abs().mean().item())

    return grads

# Sigmoid vs ReLU
print("=== Sigmoid ===")
grads_sigmoid = check_gradient_flow(nn.Sigmoid)
for i, g in enumerate(grads_sigmoid):
    print(f"Layer {i}: gradient = {g:.8f}")

print("\n=== ReLU ===")
grads_relu = check_gradient_flow(nn.ReLU)
for i, g in enumerate(grads_relu):
    print(f"Layer {i}: gradient = {g:.8f}")
```

**결과**: Sigmoid는 앞쪽 레이어 gradient가 거의 0, ReLU는 상대적으로 유지

---

## 해결책들

### 1. ReLU 활성화 함수

$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

- 양수 영역에서 gradient = 1 → 곱해도 줄어들지 않음

```python
# ReLU 사용
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),  # Sigmoid 대신
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

### 2. Skip Connection (ResNet)

$$
y = F(x) + x
$$

Chain Rule 적용:

$$
\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1
$$

**핵심**: $+1$ 덕분에 gradient가 **항상 1 이상!**

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)  # ← Skip Connection

# Gradient 흐름 확인
block = ResidualBlock(64)
x = torch.randn(1, 64, requires_grad=True)
y = block(x)
y.sum().backward()

print(f"Input gradient norm: {x.grad.norm().item():.4f}")
# Skip connection 덕분에 gradient가 살아있음!
```

### 3. Batch Normalization

각 층의 출력을 정규화하여 gradient 스케일 유지:

```python
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.BatchNorm1d(64),  # ← 추가
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

---

## Exploding Gradient 문제

### 원인: 각 항이 1보다 크면?

$$
2 \times 2 \times 2 \times \ldots = 2^L \rightarrow \infty
$$

### 해결: Gradient Clipping

```python
# Gradient 크기 제한
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

```python
# 예시: RNN 학습
model = nn.LSTM(64, 64, num_layers=3)
optimizer = torch.optim.Adam(model.parameters())

for x, y in dataloader:
    loss = criterion(model(x), y)
    loss.backward()

    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()
```

---

## 활성화 함수별 Chain Rule 효과

| 활성화 함수 | 미분 범위 | Vanishing? | Exploding? |
|-------------|-----------|------------|------------|
| Sigmoid | [0, 0.25] | 심각 | X |
| Tanh | [0, 1] | 있음 | X |
| ReLU | {0, 1} | 가능 (Dead ReLU) | X |
| Leaky ReLU | {0.01, 1} | 거의 없음 | X |
| GELU | 연속 | 약간 | X |

### 각 활성화 함수의 gradient

```python
x = torch.linspace(-3, 3, 100, requires_grad=True)

activations = {
    'Sigmoid': torch.sigmoid,
    'Tanh': torch.tanh,
    'ReLU': F.relu,
    'LeakyReLU': lambda x: F.leaky_relu(x, 0.01),
    'GELU': F.gelu
}

for name, act_fn in activations.items():
    x_copy = x.clone().requires_grad_(True)
    y = act_fn(x_copy)
    y.sum().backward()

    print(f"{name}: gradient range [{x_copy.grad.min():.3f}, {x_copy.grad.max():.3f}]")
```

---

## 코드로 확인: Chain Rule 분해

```python
import torch
import torch.nn.functional as F

# 복잡한 함수: softmax(W2 @ relu(W1 @ x))
x = torch.tensor([[1.0, 2.0]])
W1 = torch.tensor([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]], requires_grad=True)
W2 = torch.tensor([[0.1, 0.2],
                   [0.3, 0.4],
                   [0.5, 0.6]], requires_grad=True)
target = torch.tensor([1])

# Forward Pass (각 단계 저장)
z1 = x @ W1                    # Linear 1
a1 = F.relu(z1)                # ReLU
z2 = a1 @ W2                   # Linear 2
output = F.softmax(z2, dim=1)  # Softmax
loss = F.cross_entropy(z2, target)

print("=== Forward Pass ===")
print(f"x:      {x}")
print(f"z1:     {z1}")
print(f"a1:     {a1}")
print(f"z2:     {z2}")
print(f"output: {output}")
print(f"loss:   {loss.item():.4f}")

# Backward Pass
loss.backward()

print("\n=== Gradients (Chain Rule 결과) ===")
print(f"∂L/∂W1:\n{W1.grad}")
print(f"∂L/∂W2:\n{W2.grad}")
```

---

## 핵심 정리

| 개념 | 의미 | 딥러닝 적용 |
|------|------|------------|
| Chain Rule | 연결된 함수의 미분 | 역전파의 수학적 기반 |
| 곱셈 규칙 | 각 연결의 미분을 곱함 | 층별 gradient 전파 |
| Vanishing | gradient가 0에 수렴 | 앞 레이어 학습 불가 |
| Exploding | gradient가 발산 | 학습 불안정 |

## 핵심 통찰

1. **신경망 = 합성 함수**: Chain Rule이 필수
2. **곱셈의 누적**: 많이 곱할수록 극단값으로
3. **ReLU의 장점**: gradient가 1 (감쇠 없음)
4. **Skip Connection**: gradient에 +1 보장

---

## 다음 단계

Chain Rule은 수학적 원리입니다. 실제 구현에서는 이것을 **효율적으로** 계산해야 합니다.

→ [Backpropagation](/ko/docs/math/calculus/backpropagation): 효율적인 gradient 계산

## 관련 콘텐츠

- [Gradient](/ko/docs/math/calculus/gradient) - 편미분과 gradient
- [Backpropagation](/ko/docs/math/calculus/backpropagation) - Chain Rule의 효율적 구현
- [ResNet](/ko/docs/architecture/cnn/resnet) - Skip Connection으로 Vanishing 해결
- [BatchNorm](/ko/docs/components/normalization/batch-norm) - Gradient 흐름 개선
