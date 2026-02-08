---
title: "Label Smoothing"
weight: 3
math: true
---

# Label Smoothing

## 개요

Label Smoothing은 정답 레이블을 "부드럽게" 만들어 모델이 과도하게 확신하는 것을 방지합니다.

## 문제: 과신뢰

Hard label (one-hot):
```
정답 클래스 = [0, 0, 1, 0, 0]  # 100% 확신
```

모델이 이를 완벽히 맞추려면:
- 정답 클래스 확률 → 1.0
- 다른 클래스 확률 → 0.0
- logit 차이 → ∞

**문제점**:
- 과적합 유발
- calibration 나쁨 (확률 신뢰 어려움)

## Label Smoothing

일부 확률을 다른 클래스에 분배:

$$
y'_i = \begin{cases}
1 - \alpha + \frac{\alpha}{K} & \text{if } i = \text{정답} \\
\frac{\alpha}{K} & \text{otherwise}
\end{cases}
$$

- α: smoothing factor (보통 0.1)
- K: 클래스 수

**예시** (α=0.1, K=5):
```
Hard:   [0, 0, 1, 0, 0]
Smooth: [0.02, 0.02, 0.92, 0.02, 0.02]
```

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 내장
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 수동 구현
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / self.num_classes

        # Soft labels
        soft_targets = torch.full_like(logits, smooth_value)
        soft_targets.scatter_(1, targets.unsqueeze(1), confidence + smooth_value)

        # Cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss
```

## 효과

1. **정규화**: 모델이 극단적 예측 피함
2. **Calibration 개선**: 예측 확률이 더 신뢰할 만함
3. **일반화**: 테스트 성능 향상

## Teacher Forcing과의 관계

Knowledge Distillation에서 soft label 사용과 유사:
- Label Smoothing: 균등 분포로 smoothing
- KD: Teacher 모델의 soft prediction 사용

## 일반적인 값

- **이미지 분류**: α = 0.1
- **Transformer**: α = 0.1
- **많은 클래스**: α = 0.1 ~ 0.2

## 관련 콘텐츠

- [Cross-Entropy](/ko/docs/components/training/loss/cross-entropy)
- [Dropout](/ko/docs/components/training/regularization/dropout)
- [Classification](/ko/docs/task/classification)
