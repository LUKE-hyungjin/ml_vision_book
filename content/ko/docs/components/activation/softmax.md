---
title: "Softmax"
weight: 3
math: true
---

# Softmax (소프트맥스)

{{% hint info %}}
**선수지식**: [확률 기초](/ko/docs/math/probability/basics) | [미분 기초](/ko/docs/math/calculus/basics)
{{% /hint %}}

> **한 줄 요약**: Softmax는 임의의 실수 벡터를 **합이 1인 확률 분포**로 변환하여, "가장 가능성 높은 클래스"를 선택할 수 있게 만드는 함수입니다.

![Softmax 변환 과정](/images/components/activation/ko/softmax-transform.png)

## 왜 Softmax가 필요한가?

### 문제 상황: "네트워크 출력을 클래스 확률로 해석하고 싶습니다"

```python
# 네트워크가 3개 클래스에 대해 출력한 점수 (logits)
logits = [2.0, 1.0, 0.1]
# → "고양이=2.0, 개=1.0, 새=0.1"
# → 이 숫자들이 확률인가? 아님! (합 = 3.1, 음수도 가능)

# 필요한 것: 합이 1이고, 각각 0~1인 확률 분포
probs = softmax([2.0, 1.0, 0.1])
# → [0.659, 0.243, 0.099]
# → "고양이 65.9%, 개 24.3%, 새 9.9%"
```

### 해결: "지수 함수로 양수를 만들고, 합으로 나누자!"

선거에 비유하면:
- **logits** = 각 후보의 인기 점수 (음수 가능, 범위 무제한)
- **exp(logits)** = 점수를 양수로 변환 (인기가 높을수록 기하급수적으로 커짐)
- **나누기** = 전체 대비 비율 계산 → 득표율!

---

## 수식

### Softmax 함수

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**각 기호의 의미:**
- $z_i$ : 클래스 $i$의 logit (네트워크 출력)
- $K$ : 전체 클래스 수
- $e^{z_i}$ : 지수 함수 — 음수도 양수로 만듦
- $\sum e^{z_j}$ : 모든 클래스의 지수 합 — 정규화 상수

### 핵심 성질

1. **출력 합 = 1**: $\sum_{i=1}^{K} \text{Softmax}(z_i) = 1$ — 확률 분포!
2. **순서 보존**: $z_i > z_j \Rightarrow \text{Softmax}(z_i) > \text{Softmax}(z_j)$
3. **지수적 차이 증폭**: 작은 logit 차이가 큰 확률 차이로

```
logits = [3.0, 1.0, 0.5]
exp    = [20.1, 2.72, 1.65]   ← 3.0과 1.0의 차이가 크게 증폭
sum    = 24.47
softmax = [0.82, 0.11, 0.07]  ← "거의 확신"
```

### Temperature Scaling

$$
\text{Softmax}(z_i; T) = \frac{e^{z_i / T}}{\sum_{j=1}^{K} e^{z_j / T}}
$$

- $T$ : 온도 (temperature)
- $T > 1$ : **부드러운** 분포 (불확실성 표현)
- $T < 1$ : **날카로운** 분포 (확신 강조)
- $T \to 0$ : argmax와 동일 (one-hot)
- $T = 1$ : 일반 Softmax

```
logits = [2.0, 1.0, 0.5]

T=0.5:  [0.88, 0.08, 0.04]  ← 날카로움 (확신)
T=1.0:  [0.64, 0.24, 0.12]  ← 표준
T=2.0:  [0.46, 0.31, 0.23]  ← 부드러움 (불확실)
T=∞:    [0.33, 0.33, 0.33]  ← 균등 분포
```

### Softmax의 미분 (Jacobian)

$$
\frac{\partial \text{Softmax}(z_i)}{\partial z_j} = \text{Softmax}(z_i)(\delta_{ij} - \text{Softmax}(z_j))
$$

- $\delta_{ij}$ : 크로네커 델타 ($i=j$이면 1, 아니면 0)
- 실무에서는 직접 계산할 일 없음 — `CrossEntropyLoss`가 처리

---

## 수치 안정성

### 문제: Overflow

```python
import torch

# 큰 logit → exp 계산 시 overflow!
z = torch.tensor([1000.0, 1001.0, 999.0])
# e^1000 = Inf → 계산 실패!
```

### 해결: Max Subtraction Trick

$$
\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

모든 logit에서 최댓값을 빼면 수학적으로 **동일한 결과**이면서 overflow를 방지합니다:

```python
z = torch.tensor([1000.0, 1001.0, 999.0])
z_shifted = z - z.max()  # [−1, 0, −2]
# e^0 = 1, e^−1 = 0.37, e^−2 = 0.14  ← 안전!
```

PyTorch의 `torch.softmax()`는 이 트릭을 자동으로 적용합니다.

---

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === PyTorch Softmax ===
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=-1)
print(f"logits: {logits.tolist()}")
print(f"probs:  {[f'{p:.4f}' for p in probs.tolist()]}")
print(f"합:     {probs.sum():.4f}")
# probs:  ['0.6590', '0.2424', '0.0986']
# 합:     1.0000


# === 수동 구현 (수치 안정적) ===
def manual_softmax(z):
    z_shifted = z - z.max()  # max subtraction trick
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum()


# === Temperature Scaling ===
def softmax_with_temperature(z, T=1.0):
    return F.softmax(z / T, dim=-1)

logits = torch.tensor([2.0, 1.0, 0.5])
for T in [0.5, 1.0, 2.0, 5.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"T={T:.1f}: {[f'{p:.3f}' for p in probs.tolist()]}")


# === 분류 모델에서의 사용 ===
class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        # Softmax는 CrossEntropyLoss 안에 포함!

    def forward(self, x):
        return self.fc(x)  # logits 출력 (학습 시)

    def predict(self, x):
        logits = self.fc(x)
        return F.softmax(logits, dim=-1)  # 확률 출력 (추론 시)

# 학습: CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()  # logit을 직접 받음 (Softmax 포함!)
```

---

## Softmax + Cross-Entropy

### 왜 합쳐서 쓰나?

```python
# 방법 1: 분리 (수치 불안정)
probs = F.softmax(logits, dim=-1)     # step 1
loss = -torch.log(probs[target])       # step 2: log(0)이면 -inf!

# 방법 2: 합친 버전 (수치 안정)
loss = F.cross_entropy(logits, target)  # LogSoftmax + NLLLoss
# log(softmax(z)) = z - log(Σexp(z)) → log(0) 문제 회피
```

**규칙**: 학습 시에는 `CrossEntropyLoss(logits, target)`을 사용하고, 추론 시에만 `softmax(logits)`를 호출합니다.

---

## Attention에서의 Softmax

Softmax는 분류뿐 아니라 **Attention** 메커니즘에서도 핵심 역할을 합니다:

```python
# Self-Attention
scores = Q @ K.T / sqrt(d_k)       # (T, T) 유사도 점수
weights = F.softmax(scores, dim=-1)  # 확률 분포로 변환!
output = weights @ V                 # 가중 합

# Softmax가 하는 일:
# 1. 유사도 점수를 양수로 만듦
# 2. 합이 1이 되게 정규화
# 3. → 각 토큰이 다른 토큰에 "얼마나 주목할지"를 결정
```

---

## 코드로 확인하기

```python
import torch
import torch.nn.functional as F

# === 합이 1인지 확인 ===
print("=== Softmax 기본 성질 ===")
logits = torch.tensor([3.0, 1.0, -1.0, 0.5])
probs = F.softmax(logits, dim=-1)
print(f"logits: {logits.tolist()}")
print(f"probs:  {[f'{p:.4f}' for p in probs.tolist()]}")
print(f"합:     {probs.sum():.6f}")  # 1.000000


# === Temperature 효과 ===
print("\n=== Temperature Scaling ===")
logits = torch.tensor([2.5, 2.0, 1.0])
for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = F.softmax(logits / T, dim=-1)
    print(f"T={T:>4.1f}: {[f'{p:.4f}' for p in probs.tolist()]}")
# T→0: argmax처럼 동작, T→∞: 균등분포


# === 수치 안정성 확인 ===
print("\n=== 수치 안정성 ===")
large_logits = torch.tensor([1000.0, 1001.0, 999.0])
probs = F.softmax(large_logits, dim=-1)  # PyTorch가 자동 처리
print(f"큰 logits: {large_logits.tolist()}")
print(f"probs:     {[f'{p:.4f}' for p in probs.tolist()]}")  # 정상 동작!


# === Sigmoid vs Softmax ===
print("\n=== Sigmoid vs Softmax ===")
logits = torch.tensor([2.0, 1.0, 0.5])

sig_probs = torch.sigmoid(logits)  # 각각 독립
soft_probs = F.softmax(logits, dim=-1)  # 경쟁

print(f"Sigmoid: {[f'{p:.4f}' for p in sig_probs.tolist()]}  합={sig_probs.sum():.4f}")
print(f"Softmax: {[f'{p:.4f}' for p in soft_probs.tolist()]}  합={soft_probs.sum():.4f}")
# Sigmoid: 합 ≠ 1, Softmax: 합 = 1


# === CrossEntropyLoss 사용법 ===
print("\n=== CrossEntropyLoss ===")
logits = torch.tensor([[2.0, 0.5, -1.0]])  # (1, 3) — 배치 1, 클래스 3
target = torch.tensor([0])  # 정답 = 클래스 0

loss = F.cross_entropy(logits, target)
probs = F.softmax(logits, dim=-1)
print(f"logits: {logits[0].tolist()}")
print(f"probs:  {[f'{p:.4f}' for p in probs[0].tolist()]}")
print(f"loss:   {loss.item():.4f}")
print(f"확인:   -log(prob[0]) = {-torch.log(probs[0, 0]).item():.4f}")
```

---

## 핵심 정리

| 항목 | 내용 |
|------|------|
| **수식** | $e^{z_i} / \sum e^{z_j}$ |
| **출력 범위** | $(0, 1)$, 합 = 1 |
| **입력** | 벡터 (logits) |
| **핵심 성질** | 확률 분포 변환, 순서 보존 |
| **수치 안정성** | max subtraction trick |
| **주 사용처** | 다중 분류 출력, Attention |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| 분류 출력 | 거의 모든 분류 모델 | logit → 확률 변환 |
| Temperature | Knowledge Distillation, 생성 모델 | 분포 조절 |
| Attention weights | Self-Attention | 주목도 분포 |
| CrossEntropyLoss | 학습 | Softmax + Log + NLLLoss |

---

## 관련 콘텐츠

- [확률 기초](/ko/docs/math/probability/basics) — 선수 지식: 확률 분포
- [미분 기초](/ko/docs/math/calculus/basics) — 선수 지식: 지수 함수, 미분
- [Sigmoid](/ko/docs/components/activation/sigmoid) — 이진 분류의 활성화 함수
- [Cross-Entropy Loss](/ko/docs/components/training/loss/cross-entropy) — Softmax와 결합된 손실 함수
- [Self-Attention](/ko/docs/components/attention/self-attention) — Softmax가 핵심인 메커니즘
