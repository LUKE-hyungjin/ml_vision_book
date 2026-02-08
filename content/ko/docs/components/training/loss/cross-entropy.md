---
title: "Cross-Entropy Loss"
weight: 1
math: true
---

# Cross-Entropy Loss

## 개요

Cross-Entropy는 두 확률분포 간의 차이를 측정합니다. 분류 문제의 표준 손실 함수입니다.

## 수식

**이진 분류**:
$$
L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

**다중 분류**:
$$
L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

One-hot 인코딩에서 정답 클래스 c*만 y=1이므로:
$$
L = -\log(\hat{y}_{c^*})
$$

## 직관적 이해

- 정답 클래스의 확률이 높을수록 → 손실 낮음
- 확률 1.0 → 손실 0
- 확률 0.01 → 손실 4.6
- 확률 0.0001 → 손실 9.2

**핵심**: 정답에 높은 확률을 할당하도록 학습

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch CrossEntropyLoss (Softmax + NLL Loss 결합)
criterion = nn.CrossEntropyLoss()

logits = torch.randn(32, 10)  # (batch, num_classes) - raw scores
targets = torch.randint(0, 10, (32,))  # class indices

loss = criterion(logits, targets)

# 수동 구현
def cross_entropy(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return F.nll_loss(log_probs, targets)

# 또는 더 명시적으로
def manual_cross_entropy(logits, targets):
    probs = F.softmax(logits, dim=-1)
    batch_size = logits.size(0)
    return -torch.log(probs[range(batch_size), targets]).mean()
```

## Softmax + Cross-Entropy

실제로는 Softmax와 Cross-Entropy를 합쳐서 계산 (수치 안정성):

```python
# 나쁜 방법 (수치 불안정)
probs = torch.softmax(logits, dim=-1)
loss = -torch.log(probs[range(B), targets]).mean()

# 좋은 방법 (PyTorch가 내부적으로 log-sum-exp trick 사용)
loss = F.cross_entropy(logits, targets)
```

## Label Smoothing

정답에 100% 확신 대신 약간의 불확실성 부여:

$$
y'_c = (1 - \alpha) \cdot y_c + \frac{\alpha}{C}
$$

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

효과:
- 과적합 방지
- 더 부드러운 확률 분포 학습
- Calibration 개선

## 관련 콘텐츠

- [Focal Loss](/ko/docs/components/training/loss/focal-loss) - 불균형 데이터용
- [확률분포](/ko/docs/math/probability/distribution) - Softmax
- [Classification](/ko/docs/task/classification)
