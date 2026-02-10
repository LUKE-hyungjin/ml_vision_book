---
title: "Sigmoid"
weight: 2
math: true
---

# Sigmoid (시그모이드)

{{% hint info %}}
**선수지식**: [미분 기초](/ko/docs/math/calculus/basics)
{{% /hint %}}

> **한 줄 요약**: Sigmoid는 임의의 실수를 **0~1 사이의 값**으로 압축하여, **확률이나 게이트 값**으로 해석할 수 있게 만드는 활성화 함수입니다.

## 왜 Sigmoid가 필요한가?

### 문제 상황: "네트워크 출력을 확률로 해석하고 싶습니다"

```python
# 네트워크 출력 (logit)
logit = 3.7  # 이 숫자가 "고양이일 확률"인가?
# → 범위가 (-∞, +∞)이므로 확률로 해석 불가!

# 필요한 것: 0~1 사이로 변환
prob = sigmoid(3.7)  # = 0.976
# → "97.6% 확률로 고양이" 라고 해석 가능!
```

### 해결: "S자 곡선으로 0~1에 매핑하자!"

수도꼭지에 비유하면:
- 입력이 **매우 큰 양수** → 수도꼭지를 끝까지 열음 → 출력 ≈ 1
- 입력이 **0** → 수도꼭지 반쯤 → 출력 = 0.5
- 입력이 **매우 큰 음수** → 수도꼭지를 끝까지 닫음 → 출력 ≈ 0

---

## 수식

### Sigmoid 함수

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### 미분

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

**각 기호의 의미:**
- $x$ : 입력값 (범위: $(-\infty, +\infty)$)
- $e^{-x}$ : 지수 감쇠 — $x$가 클수록 0에 가까워짐
- $\sigma(x)$ : 출력값 (범위: $(0, 1)$)

### 핵심 성질

| 입력 $x$ | $e^{-x}$ | $\sigma(x)$ |
|-----------|----------|-------------|
| $-\infty$ | $\infty$ | $0$ |
| $-2$ | $7.39$ | $0.12$ |
| $0$ | $1$ | $0.5$ |
| $2$ | $0.14$ | $0.88$ |
| $+\infty$ | $0$ | $1$ |

### 미분의 특성

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

- 최댓값: $x=0$일 때 $\sigma'(0) = 0.25$
- $|x|$가 커지면 → $\sigma'(x) \to 0$ — **Gradient Vanishing!**

```
x = 0:   σ'(0) = 0.5 × 0.5 = 0.25     ← 최대
x = 5:   σ'(5) = 0.993 × 0.007 = 0.007  ← 거의 0
x = -5:  σ'(-5) = 0.007 × 0.993 = 0.007  ← 거의 0
```

---

## 현대적 사용처

Sigmoid는 은닉층 활성화로는 거의 쓰이지 않지만, 특정 역할에서 **여전히 필수적**입니다.

### 1. 이진 분류 출력층

```python
# "고양이인가 아닌가?" → 확률로 변환
logit = model(image)           # 하나의 실수
prob = torch.sigmoid(logit)    # 0~1 확률
prediction = prob > 0.5        # 분류
```

### 2. 게이트 메커니즘

```python
# LSTM의 게이트
forget_gate = torch.sigmoid(Wf @ x + bf)   # 0~1: 얼마나 잊을지
input_gate = torch.sigmoid(Wi @ x + bi)     # 0~1: 얼마나 기억할지
output_gate = torch.sigmoid(Wo @ x + bo)    # 0~1: 얼마나 출력할지

# Attention에서의 게이트 (예: GLU)
gate = torch.sigmoid(linear1(x))
output = gate * linear2(x)  # 게이트로 정보 조절
```

### 3. 멀티라벨 분류

```python
# "이 이미지에 고양이가 있는가? 개가 있는가? 새가 있는가?"
# → 각 클래스를 독립적으로 판단
logits = model(image)           # (num_classes,)
probs = torch.sigmoid(logits)   # 각각 0~1 (독립!)
# probs = [0.95, 0.02, 0.87] → 고양이 O, 개 X, 새 O
```

### Sigmoid vs Softmax

| | Sigmoid | Softmax |
|---|---------|---------|
| 출력 범위 | 각각 (0, 1) | 각각 (0, 1), 합 = 1 |
| 클래스 관계 | **독립** | **경쟁** |
| 사용처 | 멀티라벨 / 이진 | 단일 분류 |
| 예시 | 태그 여러 개 동시 | 가장 가능성 높은 하나 |

---

## 구현

```python
import torch
import torch.nn as nn

# === PyTorch Sigmoid ===
sigmoid = nn.Sigmoid()
x = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
print(f"입력:    {x.tolist()}")
print(f"Sigmoid: {sigmoid(x).tolist()}")
# Sigmoid: [0.0067, 0.1192, 0.5, 0.8808, 0.9933]


# === 함수형 사용 ===
y = torch.sigmoid(x)  # nn.Sigmoid()과 동일


# === 수동 구현 ===
def manual_sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# === 수치 안정성 (Numerically Stable) ===
def stable_sigmoid(x):
    """큰 음수에서 exp overflow 방지"""
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),      # x ≥ 0: 표준 공식
        torch.exp(x) / (1 + torch.exp(x))  # x < 0: 변형 공식
    )
# PyTorch의 torch.sigmoid()은 이미 수치 안정적


# === 이진 분류 예시 ===
class BinaryClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        # Sigmoid는 BCEWithLogitsLoss 안에 포함됨

    def forward(self, x):
        return self.fc(x)  # logit 출력 (학습 시)

    def predict(self, x):
        return torch.sigmoid(self.fc(x))  # 확률 출력 (추론 시)

# 학습: BCEWithLogitsLoss = Sigmoid + BCE (더 수치 안정적)
criterion = nn.BCEWithLogitsLoss()
```

---

## 은닉층에서 Sigmoid를 쓰지 않는 이유

### Gradient Vanishing

```python
# Sigmoid 미분 최댓값 = 0.25
# 10개 레이어를 역전파하면:
gradient = 0.25 ** 10  # = 0.0000009536
# → gradient가 사라져 학습 불가!

# ReLU는 양수에서 미분 = 1 → gradient 유지
```

### Zero-centered 아님

```python
# Sigmoid 출력: 항상 양수 (0 ~ 1)
# → gradient 방향이 편향됨 (zig-zag 업데이트)

# 입력이 모두 양수이면:
# ∂L/∂w = ∂L/∂y × ∂y/∂w = ∂L/∂y × x
# x > 0 → gradient 부호가 ∂L/∂y 부호와 항상 같음
# → w 업데이트가 같은 방향으로만 → 비효율적
```

### exp 연산 비용

```
ReLU:    max(0, x)     → 비교 연산 1개
Sigmoid: 1/(1+e^(-x))  → exp, 덧셈, 나눗셈 → 느림
```

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === Sigmoid 출력 범위 확인 ===
print("=== Sigmoid 출력 범위 ===")
x = torch.linspace(-10, 10, 21)
y = torch.sigmoid(x)
for xi, yi in zip(x, y):
    bar = "█" * int(yi * 50)
    print(f"x={xi:>6.1f}: σ(x)={yi:.4f} {bar}")


# === Gradient Vanishing 확인 ===
print("\n=== Sigmoid Gradient ===")
x = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0], requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()
print(f"입력:     {x.data.tolist()}")
print(f"gradient: {x.grad.tolist()}")
# gradient: [0.0066, 0.1050, 0.2500, 0.1050, 0.0066]
# |x|가 커지면 gradient → 0


# === 이진 분류 실습 ===
print("\n=== 이진 분류 ===")
# 모델이 출력한 logit → 확률 변환
logits = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
probs = torch.sigmoid(logits)
preds = (probs > 0.5).int()
for l, p, pred in zip(logits, probs, preds):
    print(f"logit={l:>5.1f} → prob={p:.4f} → 예측={pred.item()}")


# === BCEWithLogitsLoss vs Sigmoid + BCELoss ===
print("\n=== Loss 비교 ===")
logit = torch.tensor([2.0], requires_grad=True)
target = torch.tensor([1.0])

# 방법 1: 수동 (수치 불안정할 수 있음)
loss1 = nn.BCELoss()(torch.sigmoid(logit), target)

# 방법 2: 합쳐진 버전 (수치 안정적)
logit2 = torch.tensor([2.0], requires_grad=True)
loss2 = nn.BCEWithLogitsLoss()(logit2, target)

print(f"BCELoss(sigmoid(x)):    {loss1.item():.6f}")
print(f"BCEWithLogitsLoss(x):   {loss2.item():.6f}")  # 같은 값, 더 안정적
```

---

## 핵심 정리

| 항목 | 내용 |
|------|------|
| **수식** | $\sigma(x) = 1 / (1 + e^{-x})$ |
| **출력 범위** | $(0, 1)$ |
| **미분 최댓값** | $0.25$ (at $x=0$) |
| **은닉층 사용** | X — Gradient Vanishing |
| **현대적 사용** | 이진 분류 출력, 게이트, 멀티라벨 |
| **Loss 조합** | `BCEWithLogitsLoss` 권장 |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| 이진 분류 출력 | 모든 이진 분류 모델 | logit → 확률 변환 |
| 게이트 메커니즘 | LSTM, GRU, GLU | 정보 흐름 조절 |
| 멀티라벨 분류 | Object Detection (클래스 독립 판단) | 독립적 확률 |
| Swish/SiLU | EfficientNet, LLaMA | $x \cdot \sigma(x)$ |

---

## 관련 콘텐츠

- [미분 기초](/ko/docs/math/calculus/basics) — 선수 지식: 미분, 지수함수
- [ReLU](/ko/docs/components/activation/relu) — 은닉층의 표준 활성화 함수
- [Softmax](/ko/docs/components/activation/softmax) — 다중 분류의 활성화 함수
- [Swish/SiLU](/ko/docs/components/activation/swish-silu) — Sigmoid 기반 활성화 함수
- [Cross-Entropy Loss](/ko/docs/components/training/loss/cross-entropy) — Sigmoid와 함께 사용되는 손실 함수
