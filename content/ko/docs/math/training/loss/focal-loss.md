---
title: "Focal Loss"
weight: 2
math: true
---

# Focal Loss

## 개요

Focal Loss는 클래스 불균형 문제를 해결하기 위해 쉬운 샘플의 가중치를 낮추고 어려운 샘플에 집중합니다.

## 문제: 클래스 불균형

Object Detection에서:
- 배경 (Negative): 수만 개
- 객체 (Positive): 수십 개

Cross-Entropy만 쓰면:
- 쉬운 배경 샘플이 손실 지배
- 어려운 객체 샘플 무시됨

## 수식

$$
FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- **p_t**: 정답 클래스의 예측 확률
- **α**: 클래스 가중치 (불균형 보정)
- **γ**: Focusing parameter (보통 2)

### (1-p_t)^γ의 역할

| p_t (정답 확률) | (1-p_t)² | 효과 |
|----------------|----------|------|
| 0.9 (쉬운 샘플) | 0.01 | 손실 100배 감소 |
| 0.5 (보통) | 0.25 | 손실 4배 감소 |
| 0.1 (어려운 샘플) | 0.81 | 거의 그대로 |

## 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 정답 클래스 확률

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weight (선택적)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        return loss.mean()

# 사용
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, targets)
```

## 이진 분류용 Focal Loss

```python
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_weight = alpha_t * (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')

        return (focal_weight * bce).mean()
```

## 하이퍼파라미터

- **γ = 0**: Cross-Entropy와 동일
- **γ = 2**: 가장 일반적 (논문 권장)
- **γ 증가**: 어려운 샘플에 더 집중

- **α**: 클래스 빈도의 역수로 설정 (예: positive가 1%면 α=0.99)

## 관련 콘텐츠

- [Cross-Entropy](/ko/docs/math/training/loss/cross-entropy)
- [Detection](/ko/docs/task/detection) - Focal Loss 탄생 배경
- [YOLO](/ko/docs/architecture/detection/yolo)
