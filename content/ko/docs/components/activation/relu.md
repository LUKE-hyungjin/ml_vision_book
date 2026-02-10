---
title: "ReLU"
weight: 1
math: true
---

# ReLU (Rectified Linear Unit)

{{% hint info %}}
**선수지식**: [미분 기초](/ko/docs/math/calculus/basics)
{{% /hint %}}

> **한 줄 요약**: ReLU는 음수를 0으로, 양수는 그대로 통과시키는 **가장 단순하고 널리 쓰이는 활성화 함수**입니다.

![활성화 함수 그래프 비교](/images/components/activation/ko/activation-comparison.png)

## 왜 ReLU가 필요한가?

### 문제 상황: "Sigmoid/Tanh로는 깊은 네트워크를 학습할 수 없습니다"

```python
# Sigmoid의 문제: gradient vanishing
# Sigmoid 미분의 최댓값 = 0.25

# 10개 레이어를 역전파하면:
gradient = 0.25 ** 10  # = 0.0000009536...
# → gradient가 거의 0이 되어 학습 불가!
```

**왜 이런 일이?** Sigmoid와 Tanh는 입력이 크거나 작을 때 미분값이 거의 0입니다:

```
Sigmoid:  σ'(x) → 0  when |x| > 4    ← 포화(saturation)
Tanh:     tanh'(x) → 0  when |x| > 2  ← 포화(saturation)
```

### 해결: "양수 영역에서는 gradient를 1로 유지하자!"

```
ReLU: f(x) = max(0, x)
       f'(x) = 1  (x > 0)     ← gradient가 줄어들지 않음!
       f'(x) = 0  (x < 0)     ← 음수는 비활성화
```

시험에 비유하면:
- **Sigmoid** = 0~100점을 0~1로 압축 → 95점과 100점의 차이가 거의 없음
- **ReLU** = 0점 미만은 0점, 나머지는 원점수 그대로 → 차이가 명확!

---

## 수식

### ReLU

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\\\ 0 & \text{if } x \leq 0 \end{cases}
$$

### 미분

$$
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\\\ 0 & \text{if } x < 0 \end{cases}
$$

**각 기호의 의미:**
- $x$ : 이전 레이어의 출력 (활성화 전 값)
- $\max(0, x)$ : 0과 x 중 큰 값 → 음수를 0으로 자름
- $x = 0$ 에서 미분은 수학적으로 정의되지 않지만, 구현에서는 0으로 처리

### 왜 이게 효과적인가?

```
Sigmoid gradient:  ← 레이어마다 0.25배씩 줄어듦
  Layer 10: 0.25¹⁰ = 0.0000009...

ReLU gradient:     ← 양수이면 1배 (줄어들지 않음!)
  Layer 10: 1¹⁰ = 1  (x > 0인 경우)
```

---

## 변형들

### Dead ReLU 문제

ReLU의 가장 큰 약점은 **Dead Neuron (죽은 뉴런)** 문제입니다:

```python
# 학습 중 큰 음수 입력을 받으면:
x = -100
relu(x) = 0          # 출력 0
relu'(x) = 0         # gradient도 0 → 파라미터 업데이트 없음!
# → 이 뉴런은 영원히 0만 출력 = "죽음"
```

### Leaky ReLU

$$
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\\\ \alpha x & \text{if } x \leq 0 \end{cases}
$$

```python
# α = 0.01 (기본값)
# 음수에서도 작은 gradient가 흐름 → Dead Neuron 방지
```

### PReLU (Parametric ReLU)

$$
\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\\\ a \cdot x & \text{if } x \leq 0 \end{cases}
$$

- $a$ 가 **학습 가능한 파라미터** → 네트워크가 최적의 기울기를 학습

### ReLU6

$$
\text{ReLU6}(x) = \min(\max(0, x), 6)
$$

```python
# MobileNet에서 사용
# 양수도 6으로 클리핑 → 양자화에 유리 (값의 범위 제한)
```

### 변형 비교

| 변형 | 음수 영역 | 장점 | 사용처 |
|------|----------|------|--------|
| ReLU | $0$ | 빠름, 단순 | 대부분의 CNN |
| Leaky ReLU | $0.01x$ | Dead neuron 방지 | GAN |
| PReLU | $ax$ (학습) | 최적 기울기 학습 | ResNet 논문 |
| ReLU6 | $0$, 상한 6 | 양자화 친화 | MobileNet |

---

## 구현

```python
import torch
import torch.nn as nn

# === PyTorch ReLU ===
relu = nn.ReLU()
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"입력:  {x.tolist()}")
print(f"ReLU:  {relu(x).tolist()}")
# 입력:  [-3.0, -1.0, 0.0, 1.0, 3.0]
# ReLU:  [0.0, 0.0, 0.0, 1.0, 3.0]


# === inplace 옵션 ===
relu_ip = nn.ReLU(inplace=True)  # 메모리 절약 (입력을 직접 수정)


# === 수동 구현 ===
def manual_relu(x):
    return torch.clamp(x, min=0)  # = max(0, x)


# === 변형들 ===
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
prelu = nn.PReLU(num_parameters=1)  # 학습 가능한 α
relu6 = nn.ReLU6()

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 7.0])
print(f"\n입력:       {x.tolist()}")
print(f"ReLU:       {relu(x).tolist()}")
print(f"LeakyReLU:  {leaky_relu(x).tolist()}")
print(f"ReLU6:      {relu6(x).tolist()}")
# ReLU:       [0.0, 0.0, 0.0, 1.0, 7.0]
# LeakyReLU:  [-0.02, -0.01, 0.0, 1.0, 7.0]
# ReLU6:      [0.0, 0.0, 0.0, 1.0, 6.0]
```

---

## CNN에서의 위치

### Conv → BN → ReLU (표준 순서)

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ResNet, VGG 등 거의 모든 CNN이 이 패턴
```

### 왜 BN 뒤에 ReLU인가?

```
Conv → BN → ReLU (표준)
  BN이 평균 0으로 맞춤 → 약 절반이 음수 → ReLU가 절반을 0으로 만듦
  → 적절한 sparsity (희소성)

Conv → ReLU → BN (비표준)
  ReLU가 음수를 다 지움 → BN의 입력이 양수로 편향됨
  → BN 정규화 효과 감소
```

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === Gradient Vanishing 비교 ===
print("=== Gradient 비교: Sigmoid vs ReLU ===")

# Sigmoid: 10개 레이어 역전파 시 gradient
sigmoid_grad = 0.25 ** 10  # Sigmoid 미분 최댓값 = 0.25
print(f"Sigmoid 10층 gradient: {sigmoid_grad:.10f}")

# ReLU: 양수 영역이면 gradient 유지
relu_grad = 1.0 ** 10  # ReLU 미분 = 1 (x > 0)
print(f"ReLU 10층 gradient:    {relu_grad:.1f}")


# === Dead Neuron 확인 ===
print("\n=== Dead Neuron 확인 ===")
relu = nn.ReLU()
x = torch.randn(1000)  # 표준정규분포
y = relu(x)
dead_ratio = (y == 0).float().mean()
print(f"표준정규분포 입력 → 비활성화 비율: {dead_ratio:.1%}")
# 약 50%가 비활성화 (정규분포의 음수 부분)


# === Sparsity (희소성) 효과 ===
print("\n=== Sparsity 효과 ===")
x = torch.randn(4, 64, 8, 8)  # (B, C, H, W)
y = relu(x)
sparsity = (y == 0).float().mean()
print(f"Conv 출력 → ReLU 후 sparsity: {sparsity:.1%}")
# 약 50% — 절반의 뉴런만 활성화 → 계산 효율적


# === 변형별 Dead Neuron 비교 ===
print("\n=== 변형별 비활성화(=0) 비율 ===")
x = torch.randn(10000)
for name, fn in [("ReLU", nn.ReLU()),
                 ("LeakyReLU", nn.LeakyReLU()),
                 ("ReLU6", nn.ReLU6())]:
    y = fn(x)
    zero_ratio = (y == 0).float().mean()
    print(f"{name:>10}: 완전 비활성화 {zero_ratio:.1%}")
```

---

## 핵심 정리

| 항목 | 내용 |
|------|------|
| **수식** | $\max(0, x)$ |
| **출력 범위** | $[0, \infty)$ |
| **미분** | 0 또는 1 (양수면 1) |
| **장점** | 빠름, gradient 유지, sparsity |
| **단점** | Dead Neuron (음수 영역 영구 비활성) |
| **주 사용처** | CNN (ResNet, VGG 등) |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| Conv-BN-ReLU | [ResNet](/ko/docs/architecture/cnn/resnet), [VGG](/ko/docs/architecture/cnn/vgg) | CNN의 표준 블록 |
| LeakyReLU | GAN (Discriminator) | Dead Neuron 방지 |
| ReLU6 | MobileNet | 양자화 친화적 |
| ReLU → GELU 교체 | Transformer | 더 부드러운 비선형성 |

---

## 관련 콘텐츠

- [미분 기초](/ko/docs/math/calculus/basics) — 선수 지식: 미분의 정의
- [GELU](/ko/docs/components/activation/gelu) — Transformer의 활성화 함수
- [Sigmoid](/ko/docs/components/activation/sigmoid) — 게이트/이진 분류의 활성화 함수
- [Swish/SiLU](/ko/docs/components/activation/swish-silu) — ReLU의 부드러운 대안
- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — ReLU 앞에서 정규화
