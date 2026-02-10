---
title: "테일러 급수"
weight: 3
math: true
---

# 테일러 급수 (Taylor Series)

{{< figure src="/images/math/calculus/ko/taylor-approximation.png" caption="테일러 근사: 차수가 높아질수록 원래 함수에 가까워지며, 각 차수의 근사가 딥러닝 기법으로 연결됨" >}}

{{% hint info %}}
**선수지식**: [미분 기초](/ko/docs/math/calculus/basics)
{{% /hint %}}

> **한 줄 요약**: 테일러 급수는 **어떤 함수든 다항식으로 근사**할 수 있다는 것입니다. Gradient Descent가 작동하는 이유, Softmax의 overflow, GELU의 근사식 — 모두 테일러 전개에서 나옵니다.

## 왜 테일러 급수를 배워야 하나요?

### 문제 상황 1: "Gradient Descent는 왜 작동하나요?"

```python
optimizer = SGD(model.parameters(), lr=0.01)
loss.backward()
optimizer.step()  # 왜 이렇게 하면 loss가 줄어들지?
```

→ Loss를 현재 지점에서 **1차 다항식으로 근사** → gradient 반대 방향이 최적

### 문제 상황 2: "Softmax에서 왜 최댓값을 빼나요?"

```python
logits = torch.tensor([1000.0, 1001.0, 999.0])
# torch.exp(logits)  ← overflow!

logits_safe = logits - logits.max()  # [-1, 0, -2]
torch.exp(logits_safe)  # 정상 작동
```

→ $e^x$의 테일러 전개를 보면 왜 큰 값에서 폭발하는지 알 수 있습니다.

### 문제 상황 3: "GELU 근사식의 0.044715는 어디서 온 건가요?"

```python
def gelu_approx(x):
    return 0.5 * x * (1 + torch.tanh(
        (2/torch.pi)**0.5 * (x + 0.044715 * x**3)
    ))
```

→ 가우시안 CDF의 **테일러 근사**에서 나온 계수입니다.

---

## 테일러 급수란?

### 핵심 아이디어: 미분값만으로 함수를 복원

점 $a$ 근처에서 함수 $f(x)$를 다항식으로 근사:

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n
$$

$$
= f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots
$$

**각 항의 의미:**
- $f(a)$ : 기준점에서의 **위치** (0차)
- $f'(a)(x-a)$ : **기울기** 정보 (1차 — 접선)
- $\frac{f''(a)}{2!}(x-a)^2$ : **곡률** 정보 (2차 — 얼마나 휘어있나)

### 비유: 내비게이션

현재 위치에서의 정보만으로 주변 도로를 예측:
- **0차**: "지금 해발 100m" → 평평한 판
- **1차**: "동쪽으로 올라감" → 기울어진 판
- **2차**: "점점 가팔라짐" → 휘어진 곡면
- **더 높은 차수**: 더 정확한 지형도

### $a = 0$인 특수한 경우 (매클로린 급수)

$$
f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \cdots
$$

---

## 핵심 함수들의 테일러 전개

### $e^x$: 지수 함수

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots = \sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

```python
import torch

def exp_taylor(x, n_terms=10):
    """e^x의 테일러 근사"""
    result = torch.zeros_like(x)
    term = torch.ones_like(x)  # x^0 / 0! = 1
    for n in range(n_terms):
        result += term
        term = term * x / (n + 1)  # 다음 항
    return result

# 항 수에 따른 근사 정확도
x = torch.tensor([1.0])
for n in [1, 2, 4, 8, 15]:
    approx = exp_taylor(x, n)
    error = abs(approx.item() - torch.e.item())
    print(f"{n:2d}항: e ≈ {approx.item():.8f}, 오차: {error:.2e}")
```

**왜 $e^x$가 큰 값에서 폭발하는가?**

$e^{100} = 1 + 100 + 5000 + 166667 + \cdots$ → 항이 폭발적으로 커짐!

→ Softmax에서 `logits - max(logits)` 하는 이유

### $\ln(1+x)$: 로그 함수

$$
\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots \quad (|x| \leq 1)
$$

```python
# x가 작을 때: ln(1+x) ≈ x
x = torch.tensor([0.01, 0.001, 0.0001])
print(f"ln(1+x): {torch.log(1 + x).tolist()}")
print(f"x:       {x.tolist()}")
# 거의 같음!
```

**딥러닝 적용**: Cross-Entropy에서 $-\log p$
- $p \approx 1$ → $-\log p \approx 1-p \approx 0$ (맞으면 Loss 작음)
- $p \approx 0$ → $-\log p \to \infty$ (틀리면 Loss 폭발)

### $\sin x$, $\cos x$

$$
\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots
$$

$$
\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots
$$

```python
# x가 작을 때: sin(x) ≈ x
x_small = torch.tensor([0.01, 0.1])
print(f"sin(x): {torch.sin(x_small).tolist()}")
print(f"x:      {x_small.tolist()}")
# 0.01 근처에서 거의 같음
```

---

## 딥러닝 핵심 응용

### 1. 1차 근사 → Gradient Descent

Loss $L(\theta)$를 현재 파라미터 $\theta$ 근처에서 1차 근사:

$$
L(\theta + \Delta\theta) \approx L(\theta) + \nabla L(\theta)^T \Delta\theta
$$

Loss를 줄이려면 → $\nabla L^T \Delta\theta < 0$ → $\Delta\theta = -\eta \nabla L$:

$$
L(\theta - \eta\nabla L) \approx L(\theta) - \eta\|\nabla L\|^2 \leq L(\theta)
$$

**항상 Loss가 감소합니다!** 이것이 Gradient Descent의 수학적 보장입니다.

```python
# Gradient Descent = 1차 테일러 근사의 반복 적용
w = torch.tensor([5.0], requires_grad=True)
lr = 0.1

for step in range(10):
    loss = (w - 2) ** 2  # 최솟값: w=2
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad  # Δw = -η·∇L
        w.grad.zero_()

    if step % 3 == 0:
        print(f"step {step}: w={w.item():.4f}, loss={loss.item():.4f}")
```

### 2. 2차 근사 → Newton's Method

Loss를 2차까지 근사:

$$
L(\theta + \Delta\theta) \approx L(\theta) + \nabla L^T \Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta
$$

**각 기호의 의미:**
- $\nabla L$ : Gradient (1차) — 어디로 갈지
- $H$ : **Hessian** (2차 도함수 행렬) — 얼마나 빠르게 변하는지
- 최적 이동: $\Delta\theta^* = -H^{-1}\nabla L$

```python
# Newton's Method (1D 예시)
# f(x) = (x-3)² + 0.1(x-3)⁴
def f(x): return (x-3)**2 + 0.1*(x-3)**4

# 1차 방법 (GD) vs 2차 방법 (Newton)
x_gd = torch.tensor([0.0])
x_newton = torch.tensor([0.0])
lr = 0.05

for step in range(20):
    # GD: 1차만 사용
    x_g = x_gd.clone().requires_grad_(True)
    loss_g = f(x_g)
    loss_g.backward()
    x_gd = x_gd - lr * x_g.grad

    # Newton: 2차도 사용 (수치 미분으로 Hessian 근사)
    x_n = x_newton.clone().requires_grad_(True)
    loss_n = f(x_n)
    loss_n.backward()
    grad = x_n.grad.item()

    h = 1e-4
    x_plus = (x_newton + h).requires_grad_(True)
    x_minus = (x_newton - h).requires_grad_(True)
    f(x_plus).backward()
    f(x_minus).backward()
    hessian = (x_plus.grad.item() - x_minus.grad.item()) / (2 * h)

    x_newton = x_newton - torch.tensor([grad / max(hessian, 1e-8)])

if True:
    print(f"GD 최종:     x={x_gd.item():.4f}")
    print(f"Newton 최종: x={x_newton.item():.4f}")
    print(f"정답: 3.0")
```

**왜 Newton's Method를 딥러닝에서 안 쓰나?**
- Hessian $H$는 $n \times n$ 행렬 (n = 파라미터 수)
- ResNet-50: n ≈ 25M → H는 25M × 25M → **메모리 불가능**
- Adam은 Hessian의 **대각 근사**를 사용 → 실용적인 타협

### 3. 활성화 함수 근사: GELU

GELU의 정의: $\text{GELU}(x) = x \cdot \Phi(x)$ ($\Phi$: 정규분포 CDF)

가우시안 CDF를 테일러 근사하면:

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

```python
import torch.nn.functional as F

x = torch.linspace(-3, 3, 100)

gelu_exact = F.gelu(x)
gelu_approx = 0.5 * x * (1 + torch.tanh(
    (2/torch.pi)**0.5 * (x + 0.044715 * x**3)
))

max_error = (gelu_exact - gelu_approx).abs().max()
print(f"GELU 근사 최대 오차: {max_error:.6f}")  # 매우 작음
```

### 4. 수치 안정성: log-sum-exp

$\log\sum_i e^{x_i}$를 안전하게 계산하는 트릭:

$$
\log\sum_i e^{x_i} = c + \log\sum_i e^{x_i - c}, \quad c = \max_i x_i
$$

```python
logits = torch.tensor([1000.0, 1001.0, 999.0])

# 안정적 계산
c = logits.max()
stable = c + torch.log(torch.exp(logits - c).sum())
print(f"log-sum-exp: {stable:.4f}")

# PyTorch 내장 (내부적으로 같은 트릭 사용)
print(f"torch: {torch.logsumexp(logits, dim=0):.4f}")
```

---

## 근사의 한계: 수렴 반경

테일러 급수는 **기준점 근처에서만** 정확합니다.

```python
# sin(x)의 테일러 근사: 기준점에서 멀어지면 부정확
def sin_taylor(x, n_terms=5):
    result = torch.zeros_like(x)
    for k in range(n_terms):
        n = 2 * k + 1
        sign = (-1) ** k
        factorial = 1.0
        for i in range(1, n + 1):
            factorial *= i
        result += sign * x ** n / factorial
    return result

x = torch.tensor([0.1, 1.0, 3.0, 6.0, 10.0])
for terms in [3, 5, 10]:
    approx = sin_taylor(x, terms)
    exact = torch.sin(x)
    errors = (approx - exact).abs()
    print(f"{terms}항: 오차 = {[f'{e:.4f}' for e in errors.tolist()]}")
# x=0.1: 매우 정확, x=10: 항을 많이 써야 정확
```

**딥러닝 시사점**: Learning rate가 너무 크면 1차 근사가 깨짐 → 발산!

---

## 핵심 정리

| 근사 차수 | 수식 | 의미 | 딥러닝 적용 |
|----------|------|------|------------|
| 0차 | $f(a)$ | 상수 | 현재 Loss 값 |
| 1차 | $f(a) + f'(a)(x-a)$ | 접선 | **Gradient Descent** |
| 2차 | $+ \frac{f''(a)}{2}(x-a)^2$ | 포물선 | Newton, **Adam** |
| $n$차 | 테일러 전개 전체 | 정확한 근사 | GELU 근사 등 |

## 핵심 통찰

1. **1차 근사 = GD**: Loss의 접선을 따라 내려감. learning rate가 작으면 잘 작동
2. **2차 근사 = 곡률**: 더 정확하지만 Hessian이 비쌈. Adam이 타협안
3. **근사는 가까울 때만 유효**: learning rate가 크면 근사가 깨짐 → 발산
4. **$e^x$ 폭발 = 테일러 항의 폭발**: Softmax overflow의 근본 원인

---

## 다음 단계

함수의 근사 방법을 이해했습니다. 이제 **여러 변수의 미분을 체계적으로** 다룹니다.

→ [Gradient](/ko/docs/math/calculus/gradient): 다변수 함수의 기울기 벡터

## 관련 콘텐츠

- [미분 기초](/ko/docs/math/calculus/basics) — 미분의 기본 개념
- [Gradient](/ko/docs/math/calculus/gradient) — 다변수 미분
- [최적화 수학](/ko/docs/math/calculus/optimization) — Gradient Descent, Newton's Method
