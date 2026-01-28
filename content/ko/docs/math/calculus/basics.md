---
title: "미분 기초"
weight: 1
math: true
---

# 미분 기초 (Derivative Basics)

> **한 줄 요약**: 미분은 "조금 바꾸면 결과가 얼마나 달라지는가?"를 계산합니다.

## 왜 미분을 배워야 하나요?

### 문제 상황 1: "모델이 틀렸는데, 어떤 파라미터를 바꿔야 하나요?"

신경망에는 수백만 개의 파라미터(weight, bias)가 있습니다.

```python
model = ResNet50()  # 약 2500만 개의 파라미터
```

모델이 틀린 예측을 했을 때:
- 모든 파라미터를 무작위로 바꿔보기? → 평생 걸립니다
- **각 파라미터가 손실에 얼마나 영향을 주는지 계산** → 미분!

### 문제 상황 2: "Loss가 줄어들려면 weight를 키워야 하나요, 줄여야 하나요?"

```python
# loss = 2.5 일 때
# weight를 0.001 키우면 loss가 어떻게 될까?
```

- 미분 값 > 0: weight를 키우면 loss도 커짐 → **줄여야 함**
- 미분 값 < 0: weight를 키우면 loss가 줄어듦 → **키워야 함**

### 문제 상황 3: "Learning rate를 얼마로 해야 하나요?"

```python
optimizer = SGD(params, lr=???)  # 얼마가 적당할까?
```

Learning rate는 "미분 값에 비례해서 얼마나 이동할 것인가"입니다.
미분을 모르면 learning rate의 의미도 모릅니다.

---

## 미분이란 무엇인가?

### 핵심 아이디어: 순간 변화율

**평균 변화율** (중학교 수준):
- 서울에서 부산까지 400km를 4시간에 갔다
- 평균 속도 = 400km ÷ 4시간 = **100km/h**

**순간 변화율** (미분):
- 지금 이 순간 속도계에 찍힌 속도
- 120km/h로 달리다가 100km/h로 줄였다가...
- 매 순간의 속도가 **순간 변화율**

### 수학적 정의

함수 $f(x)$의 $x$에서의 미분:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

**풀어서 설명하면:**

1. x를 아주 조금(h만큼) 바꿔봅니다
2. f(x)가 얼마나 달라졌는지 봅니다: $f(x+h) - f(x)$
3. 바꾼 양으로 나눕니다: $\frac{f(x+h) - f(x)}{h}$
4. h를 아주 아주 작게 만듭니다: $\lim_{h \to 0}$

---

## 수치적으로 이해하기

미분을 코드로 직접 계산해봅시다.

```python
def f(x):
    return x ** 2  # f(x) = x²

def numerical_derivative(f, x, h=1e-5):
    """수치 미분: 미분의 정의를 그대로 구현"""
    return (f(x + h) - f(x)) / h

# x=3에서 f(x)=x²의 미분값
x = 3.0
derivative = numerical_derivative(f, x)
print(f"f(x) = x² at x=3")
print(f"수치 미분: {derivative:.4f}")  # 6.0000
print(f"해석적 미분: {2 * x}")         # 6 (f'(x) = 2x)
```

**결과**: $f'(3) = 6$

**해석**: x=3에서 x를 1 증가시키면, $f(x)$는 **약 6** 증가합니다.

### 검증해보기

```python
# 실제로 확인해보면:
print(f"f(3) = {f(3)}")        # 9
print(f"f(3.001) = {f(3.001)}")  # 9.006001

# 차이: 0.006001 ≈ 0.001 * 6 (변화량 * 미분값)
```

---

## 딥러닝에서 미분의 의미

신경망에서 미분은 **"이 파라미터가 손실에 얼마나 기여하는가?"**를 알려줍니다.

```python
import torch

# 간단한 모델: y = wx + b
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])
y_true = torch.tensor([10.0])

# Forward: 예측
y_pred = w * x + b  # 2*3 + 1 = 7

# Loss 계산
loss = (y_pred - y_true) ** 2  # (7-10)² = 9

# Backward: 미분 계산
loss.backward()

print(f"예측: {y_pred.item()}, 정답: {y_true.item()}")
print(f"Loss: {loss.item()}")
print(f"∂Loss/∂w = {w.grad.item()}")  # w가 loss에 미치는 영향
print(f"∂Loss/∂b = {b.grad.item()}")  # b가 loss에 미치는 영향
```

출력:
```
예측: 7.0, 정답: 10.0
Loss: 9.0
∂Loss/∂w = -18.0
∂Loss/∂b = -6.0
```

**해석:**
- $\frac{\partial L}{\partial w} = -18$: w를 1 증가시키면 Loss가 18 **감소**
- $\frac{\partial L}{\partial b} = -6$: b를 1 증가시키면 Loss가 6 **감소**

→ 둘 다 음수이므로, **w와 b를 키워야** Loss가 줄어듭니다!

---

## 기본 미분 공식

딥러닝에서 자주 쓰이는 함수들의 미분:

| 함수 $f(x)$ | 미분 $f'(x)$ | 딥러닝 적용 |
|-------------|-------------|-------------|
| $c$ (상수) | $0$ | bias는 직접 영향 없음 |
| $x$ | $1$ | Identity 함수 |
| $x^n$ | $nx^{n-1}$ | Weight decay (L2) |
| $e^x$ | $e^x$ | Softmax, Sigmoid |
| $\ln x$ | $\frac{1}{x}$ | Cross-entropy |
| $\sin x$ | $\cos x$ | Positional encoding |

### 미분 규칙

**덧셈 규칙:**
$$
(f + g)' = f' + g'
$$

**곱셈 규칙:**
$$
(f \cdot g)' = f' \cdot g + f \cdot g'
$$

**나눗셈 규칙:**
$$
\left(\frac{f}{g}\right)' = \frac{f' \cdot g - f \cdot g'}{g^2}
$$

---

## 딥러닝 활성화 함수의 미분

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

$$
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}
$$

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = F.relu(x)
y.sum().backward()

print(f"x:      {x.tolist()}")
print(f"ReLU:   {y.tolist()}")
print(f"Gradient: {x.grad.tolist()}")  # [0, 0, 0, 1, 1]
```

**문제점**: x < 0이면 gradient = 0 → **Dead ReLU** (뉴런이 죽음)

### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

```python
x = torch.tensor([-3.0, 0.0, 3.0], requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()

print(f"x:        {x.tolist()}")
print(f"Sigmoid:  {[f'{v:.4f}' for v in y.tolist()]}")
print(f"Gradient: {[f'{v:.4f}' for v in x.grad.tolist()]}")
```

**문제점**: x가 크거나 작으면 gradient ≈ 0 → **Vanishing Gradient**

### Softmax의 특별한 점

Softmax의 미분은 **다른 원소에도 영향**을 줍니다:

$$
\frac{\partial \text{softmax}_i}{\partial x_j} = \begin{cases}
p_i(1-p_i) & \text{if } i = j \\
-p_i p_j & \text{if } i \neq j
\end{cases}
$$

```python
x = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
y = F.softmax(x, dim=0)

# 첫 번째 원소에 대해 backward
y[0].backward()

print(f"Softmax: {[f'{v:.4f}' for v in y.tolist()]}")
print(f"x[0]에 대한 gradient:")
print(f"  ∂y[0]/∂x[0] = {x.grad[0]:.4f}")  # 양수 (자기 자신)
print(f"  ∂y[0]/∂x[1] = {x.grad[1]:.4f}")  # 음수 (다른 원소)
print(f"  ∂y[0]/∂x[2] = {x.grad[2]:.4f}")  # 음수 (다른 원소)
```

**의미**: 한 클래스의 확률을 높이면 다른 클래스 확률은 자동으로 낮아집니다.

---

## 코드로 확인: 수치 미분 vs 자동 미분

PyTorch의 자동 미분이 정확한지 수치 미분으로 검증:

```python
import torch

def check_gradient(f, x, h=1e-5):
    """수치 미분과 자동 미분 비교"""
    # 수치 미분
    x_plus = x.clone().detach() + h
    x_minus = x.clone().detach() - h
    numerical_grad = (f(x_plus) - f(x_minus)) / (2 * h)

    # 자동 미분
    x_auto = x.clone().requires_grad_(True)
    y = f(x_auto)
    y.backward()
    auto_grad = x_auto.grad

    print(f"수치 미분: {numerical_grad.item():.6f}")
    print(f"자동 미분: {auto_grad.item():.6f}")
    print(f"차이: {abs(numerical_grad - auto_grad).item():.10f}")

# 테스트: f(x) = x³ + 2x² + x
f = lambda x: x**3 + 2*x**2 + x
x = torch.tensor([2.0])

check_gradient(f, x)
# f'(x) = 3x² + 4x + 1 = 3(4) + 4(2) + 1 = 21
```

---

## Loss 함수의 미분

### MSE (Mean Squared Error)

$$
L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

$$
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

```python
y_pred = torch.tensor([2.5], requires_grad=True)
y_true = torch.tensor([3.0])

loss = (y_pred - y_true) ** 2
loss.backward()

print(f"예측: {y_pred.item()}, 정답: {y_true.item()}")
print(f"Loss: {loss.item()}")
print(f"Gradient: {y_pred.grad.item()}")  # 2 * (2.5 - 3.0) = -1.0
```

**해석**: gradient가 음수 → 예측값을 **키워야** Loss 감소

### Cross-Entropy (Softmax 출력 기준)

$$
L = -\sum_i y_i \log(\hat{y}_i)
$$

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
$$

(z는 softmax 이전 logit, $\hat{y}$는 softmax 출력)

```python
logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
target = torch.tensor([0])  # 정답: 첫 번째 클래스

loss = F.cross_entropy(logits, target)
loss.backward()

probs = F.softmax(logits, dim=1)
print(f"확률: {[f'{p:.4f}' for p in probs[0].tolist()]}")
print(f"Gradient: {[f'{g:.4f}' for g in logits.grad[0].tolist()]}")
# Gradient ≈ softmax(logits) - one_hot(target)
```

**핵심**: Cross-Entropy의 gradient는 단순히 **(예측 확률 - 정답)** 입니다!

---

## 핵심 정리

| 개념 | 의미 | 딥러닝 적용 |
|------|------|------------|
| 미분 | 입력 변화 → 출력 변화량 | 파라미터가 Loss에 미치는 영향 |
| 양수 gradient | 파라미터↑ → Loss↑ | 파라미터를 **줄여야** 함 |
| 음수 gradient | 파라미터↑ → Loss↓ | 파라미터를 **키워야** 함 |
| gradient = 0 | 변화해도 Loss 불변 | 학습 안 됨 (Dead ReLU 등) |

## 핵심 통찰

1. **미분 = 민감도 측정**: 이 파라미터가 결과에 얼마나 중요한가?
2. **Gradient의 부호 = 방향**: 키울지 줄일지 알려줌
3. **Gradient의 크기 = 민감도**: 얼마나 급하게 바꿔야 하는가?
4. **Gradient = 0이 문제**: Dead ReLU, Vanishing Gradient

---

## 다음 단계

미분의 기초를 이해했습니다. 하지만 신경망은 **변수가 수백만 개**입니다.

→ [Gradient](/ko/docs/math/calculus/gradient): 여러 변수에 대한 미분

## 관련 콘텐츠

- [Gradient](/ko/docs/math/calculus/gradient) - 다변수 함수의 미분
- [Chain Rule](/ko/docs/math/calculus/chain-rule) - 합성 함수의 미분
- [MSE Loss](/ko/docs/math/training/loss/mse) - 회귀 손실 함수
- [Cross-Entropy](/ko/docs/math/training/loss/cross-entropy) - 분류 손실 함수
