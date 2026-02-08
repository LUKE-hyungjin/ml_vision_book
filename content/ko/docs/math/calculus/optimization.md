---
title: "최적화 수학"
weight: 5
math: true
---

# 최적화 수학 (Optimization Mathematics)

> **한 줄 요약**: Gradient Descent가 왜 작동하는지, Adam이 왜 SGD보다 좋은지를 수학적으로 이해합니다.

## 왜 최적화 수학을 배워야 하나요?

### 문제 상황 1: "Gradient Descent는 왜 작동하나요?"

```python
# 왜 이게 최솟값을 찾아갈까?
theta = theta - lr * gradient
```

→ **Taylor 전개**로 증명할 수 있습니다.

### 문제 상황 2: "Learning Rate는 왜 중요한가요?"

```python
# lr=0.1은 되는데 lr=1.0은 발산?
optimizer = SGD(params, lr=0.1)   # OK
optimizer = SGD(params, lr=1.0)   # 발산!
```

→ **2차 미분(곡률)**과 관련이 있습니다.

### 문제 상황 3: "왜 Adam이 SGD보다 좋은가요?"

```python
# 대부분의 경우 Adam이 더 빠름
optimizer = Adam(params)   # 빠름
optimizer = SGD(params)    # 느림
```

→ Adam은 **2차 최적화를 근사**하기 때문입니다.

---

## Taylor 전개: 함수 근사의 핵심

{{< figure src="/images/math/calculus/ko/taylor-approximation.png" caption="Taylor 근사: 1차(접선)로 GD 작동 원리 증명, 2차(포물선)로 Newton 방법 유도" >}}

### 핵심 아이디어

어떤 함수든 **다항식으로 근사**할 수 있습니다.

$$
f(x + \delta) \approx f(x) + f'(x)\delta + \frac{1}{2}f''(x)\delta^2 + \ldots
$$

### 1차 근사 (Linear Approximation)

$$
f(x + \delta) \approx f(x) + f'(x)\delta
$$

**직관**: 현재 점에서 **접선**으로 함수를 근사

```python
import torch
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 2*x + 1

def f_linear_approx(x, x0):
    """x0 근처에서의 1차 근사"""
    f_x0 = f(x0)
    grad = 3*x0**2 - 2  # f'(x) = 3x² - 2
    return f_x0 + grad * (x - x0)

x0 = 1.0
x = torch.linspace(-1, 3, 100)

plt.plot(x, f(x), label='실제 함수')
plt.plot(x, f_linear_approx(x, x0), '--', label='1차 근사')
plt.scatter([x0], [f(x0)], color='red', s=100)
plt.legend()
plt.title('Taylor 1차 근사')
```

### 2차 근사 (Quadratic Approximation)

$$
f(x + \delta) \approx f(x) + f'(x)\delta + \frac{1}{2}f''(x)\delta^2
$$

**직관**: 접선 + **곡률** 정보

```python
def f_quadratic_approx(x, x0):
    """x0 근처에서의 2차 근사"""
    f_x0 = f(x0)
    grad = 3*x0**2 - 2      # f'(x)
    hess = 6*x0             # f''(x)
    delta = x - x0
    return f_x0 + grad * delta + 0.5 * hess * delta**2

plt.plot(x, f(x), label='실제 함수')
plt.plot(x, f_linear_approx(x, x0), '--', label='1차 근사')
plt.plot(x, f_quadratic_approx(x, x0), ':', label='2차 근사')
plt.legend()
```

**결과**: 2차 근사가 실제 함수에 더 가깝습니다!

---

## Gradient Descent가 작동하는 이유

### Taylor 전개로 증명

현재 점 $\theta$에서 $\theta + \delta$로 이동할 때:

$$
L(\theta + \delta) \approx L(\theta) + \nabla L(\theta)^T \delta
$$

**질문**: 어떤 $\delta$로 이동해야 Loss가 줄어들까?

**답**: $\delta = -\eta \nabla L(\theta)$ (gradient의 반대 방향)

$$
L(\theta - \eta \nabla L) \approx L(\theta) - \eta \|\nabla L\|^2 < L(\theta)
$$

$\|\nabla L\|^2 > 0$이므로 **Loss가 반드시 감소**!

```python
import torch

def loss_fn(theta):
    return (theta - 3)**2

theta = torch.tensor([0.0], requires_grad=True)
lr = 0.1

print("Gradient Descent 증명:")
for step in range(5):
    loss = loss_fn(theta)
    loss.backward()

    grad_norm_sq = (theta.grad ** 2).item()
    predicted_decrease = lr * grad_norm_sq

    with torch.no_grad():
        old_loss = loss.item()
        theta -= lr * theta.grad
        new_loss = loss_fn(theta).item()
        actual_decrease = old_loss - new_loss

    print(f"Step {step}: 예상 감소 = {predicted_decrease:.4f}, "
          f"실제 감소 = {actual_decrease:.4f}")

    theta.grad.zero_()
```

---

## Learning Rate와 곡률의 관계

### 2차 근사에서 최적의 step size

2차 근사: $L(\theta + \delta) \approx L + g^T\delta + \frac{1}{2}\delta^T H \delta$

이걸 $\delta$에 대해 최소화하면:

$$
\delta^* = -H^{-1}g
$$

이것이 **Newton's Method**입니다!

### Learning Rate의 상한

1차원에서, step size $\eta$가 수렴하려면:

$$
\eta < \frac{2}{|f''(x)|} = \frac{2}{\text{곡률}}
$$

**직관**:
- 곡률이 크면 (가파른 골짜기) → Learning Rate를 작게
- 곡률이 작으면 (평평한 골짜기) → Learning Rate를 크게 해도 됨

```python
def demonstrate_lr_curvature():
    """곡률과 Learning Rate의 관계"""

    # 곡률이 다른 두 함수
    def f1(x):
        return x**2           # f''(x) = 2, 낮은 곡률

    def f2(x):
        return 10 * x**2      # f''(x) = 20, 높은 곡률

    # 같은 Learning Rate 사용
    lr = 0.3

    for name, f, curvature in [('낮은 곡률', f1, 2), ('높은 곡률', f2, 20)]:
        x = torch.tensor([5.0], requires_grad=True)
        max_lr = 2 / curvature

        print(f"\n{name} (곡률={curvature}, 최대 lr={max_lr:.2f}):")
        print(f"  사용 lr={lr} {'< 최대' if lr < max_lr else '>= 최대 (발산 위험!)'}")

        for step in range(5):
            loss = f(x)
            loss.backward()
            with torch.no_grad():
                x -= lr * x.grad
                x.grad.zero_()
            print(f"  Step {step}: x = {x.item():.4f}")

demonstrate_lr_curvature()
```

---

## 1차 최적화 vs 2차 최적화

### 1차 최적화 (Gradient Descent)

$$
\theta_{t+1} = \theta_t - \eta \nabla L
$$

- **사용 정보**: Gradient (1차 미분)
- **장점**: 계산 간단
- **단점**: 곡률 무시 → 느림

### 2차 최적화 (Newton's Method)

$$
\theta_{t+1} = \theta_t - H^{-1} \nabla L
$$

- **사용 정보**: Gradient + Hessian (2차 미분)
- **장점**: 곡률 고려 → 빠름
- **단점**: Hessian 계산/저장 불가능 (N² 크기)

```python
def compare_1st_2nd_order():
    """1차 vs 2차 최적화 비교"""

    # 타원형 Loss: 방향에 따라 곡률이 다름
    def loss(x, y):
        return x**2 + 10*y**2  # y 방향이 10배 가파름

    # 1차: Gradient Descent
    x1, y1 = torch.tensor([5.0]), torch.tensor([5.0])
    lr = 0.05
    gd_path = [(x1.item(), y1.item())]

    for _ in range(20):
        x1.requires_grad_(True)
        y1.requires_grad_(True)
        l = loss(x1, y1)
        l.backward()
        with torch.no_grad():
            x1 -= lr * x1.grad
            y1 -= lr * y1.grad
        gd_path.append((x1.item(), y1.item()))

    # 2차: Newton (해석적 해)
    # H = [[2, 0], [0, 20]], H^-1 = [[0.5, 0], [0, 0.05]]
    x2, y2 = 5.0, 5.0
    newton_path = [(x2, y2)]

    for _ in range(3):  # Newton은 3번이면 수렴
        gx, gy = 2*x2, 20*y2  # gradient
        # H^-1 @ g
        dx = 0.5 * gx
        dy = 0.05 * gy
        x2 -= dx
        y2 -= dy
        newton_path.append((x2, y2))

    print(f"Gradient Descent: {len(gd_path)-1}회 반복")
    print(f"  시작: {gd_path[0]}")
    print(f"  끝: {gd_path[-1]}")

    print(f"\nNewton's Method: {len(newton_path)-1}회 반복")
    print(f"  시작: {newton_path[0]}")
    print(f"  끝: {newton_path[-1]}")

compare_1st_2nd_order()
```

**결과**: Newton은 3번, GD는 20번 이상 필요!

---

## Adam이 빠른 이유

{{< figure src="/images/math/calculus/ko/sgd-vs-adam.png" caption="SGD vs Adam: 타원형 Loss에서 SGD는 진동, Adam은 적응적 LR로 빠르게 수렴" >}}

### SGD의 문제

```
       y
       │    ← y 방향: 곡률 높음 (빠르게 진동)
       │ ~~~~
       │~    ~
───────┼──────── x  ← x 방향: 곡률 낮음 (천천히 이동)
       │~    ~
       │ ~~~~
```

SGD는 모든 방향에 같은 Learning Rate → 비효율적

### Adam의 해결책

**핵심 아이디어**: 각 파라미터마다 다른 Learning Rate

$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
$$

- $m_t$: Gradient의 이동 평균 (momentum)
- $v_t$: Gradient 제곱의 이동 평균 ≈ **Hessian 대각 성분**!

### 왜 $v_t$가 Hessian을 근사하는가?

$$
\mathbb{E}[g^2] \approx \text{곡률} = \frac{\partial^2 L}{\partial \theta^2}
$$

- Gradient가 큰 방향 → $v_t$ 큼 → Learning Rate 작게
- Gradient가 작은 방향 → $v_t$ 작음 → Learning Rate 크게

**결과**: 각 방향에 적응적인 Learning Rate!

```python
def visualize_adam_vs_sgd():
    """Adam vs SGD 비교"""

    def loss(params):
        x, y = params[0], params[1]
        return x**2 + 100*y**2  # y 방향이 100배 가파름

    # SGD
    params_sgd = torch.tensor([5.0, 5.0], requires_grad=True)
    opt_sgd = torch.optim.SGD([params_sgd], lr=0.01)
    sgd_losses = []

    for _ in range(100):
        opt_sgd.zero_grad()
        l = loss(params_sgd)
        l.backward()
        opt_sgd.step()
        sgd_losses.append(l.item())

    # Adam
    params_adam = torch.tensor([5.0, 5.0], requires_grad=True)
    opt_adam = torch.optim.Adam([params_adam], lr=0.5)
    adam_losses = []

    for _ in range(100):
        opt_adam.zero_grad()
        l = loss(params_adam)
        l.backward()
        opt_adam.step()
        adam_losses.append(l.item())

    print("100 iteration 후:")
    print(f"  SGD Loss:  {sgd_losses[-1]:.6f}")
    print(f"  Adam Loss: {adam_losses[-1]:.6f}")
    print(f"  Adam이 {sgd_losses[-1]/adam_losses[-1]:.1f}배 빠름!")

visualize_adam_vs_sgd()
```

---

## 실전: 최적화 선택 가이드

### 언제 무엇을 쓰나?

| 상황 | 추천 | 이유 |
|------|------|------|
| 일반적인 딥러닝 | **Adam** | 적응적 LR, 빠른 수렴 |
| 최종 fine-tuning | **SGD + momentum** | 더 좋은 일반화 |
| NLP (Transformer) | **AdamW** | Weight decay 분리 |
| 큰 배치 학습 | **LAMB** | 레이어별 적응 |

### Learning Rate 설정

```python
# 경험적 규칙
lr_sgd = 0.01 ~ 0.1      # SGD는 작게 시작
lr_adam = 0.001 ~ 0.0001  # Adam은 더 작게

# Learning Rate Finder 사용
from torch_lr_finder import LRFinder
finder = LRFinder(model, optimizer, criterion)
finder.range_test(train_loader)
finder.plot()  # 최적 LR 시각화
```

### Warmup이 필요한 이유

학습 초기에는 gradient가 불안정 → 작은 LR로 시작

```python
# Linear Warmup
def warmup_lr(step, warmup_steps, target_lr):
    if step < warmup_steps:
        return target_lr * step / warmup_steps
    return target_lr

# Warmup + Cosine Decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

---

## 핵심 정리

| 개념 | 의미 | 딥러닝 적용 |
|------|------|------------|
| Taylor 1차 | 접선 근사 | Gradient Descent 작동 원리 |
| Taylor 2차 | 곡률 포함 | Newton's Method |
| Hessian | 곡률 정보 | Learning Rate 상한 결정 |
| Adam의 $v_t$ | Hessian 근사 | 적응적 Learning Rate |

## 핵심 통찰

1. **GD가 작동하는 이유**: Taylor 1차 근사에서 Loss 감소 보장
2. **LR 상한**: $\eta < 2/\text{곡률}$, 곡률 크면 LR 작게
3. **Adam의 비밀**: Gradient 제곱으로 Hessian 대각 근사
4. **2차 최적화**: 빠르지만 메모리 문제 → Adam이 타협점

---

## 수식 요약

### Gradient Descent 수렴 조건

$$
L(\theta - \eta g) < L(\theta) \quad \text{when} \quad \eta < \frac{2}{\lambda_{max}(H)}
$$

### Newton's Method

$$
\theta_{new} = \theta - H^{-1} g
$$

### Adam Update

$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot \frac{m_t}{1-\beta_1^t}
$$

여기서 $v_t \approx \text{diag}(H)$

---

## 관련 콘텐츠

- [Gradient](/ko/docs/math/calculus/gradient) - 1차 미분과 Hessian
- [SGD](/ko/docs/components/training/optimizer/sgd) - 기본 최적화
- [Adam](/ko/docs/components/training/optimizer/adam) - 적응적 최적화
- [Learning Rate Scheduler](/ko/docs/components/training/optimizer) - LR 스케줄링
