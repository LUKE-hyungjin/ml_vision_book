---
title: "Dropout"
weight: 1
math: true
---

# Dropout

## 개요

Dropout은 학습 중 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.

## 동작 원리

**학습 시**:
- 각 뉴런을 확률 p로 0으로 설정
- 나머지 뉴런은 1/(1-p)로 스케일링 (inverted dropout)

**추론 시**:
- 모든 뉴런 사용
- 스케일링 불필요 (학습 때 이미 보정됨)

## 수식

$$
h' = \frac{1}{1-p} \cdot h \odot m, \quad m_i \sim \text{Bernoulli}(1-p)
$$

- p: dropout 확률 (보통 0.1 ~ 0.5)
- m: 마스크 (0 또는 1)
- ⊙: element-wise 곱

## 구현

```python
import torch
import torch.nn as nn

# PyTorch Dropout
dropout = nn.Dropout(p=0.5)

x = torch.randn(32, 512)

# 학습 모드
model.train()
y = dropout(x)  # 50% 뉴런이 0, 나머지 2배

# 추론 모드
model.eval()
y = dropout(x)  # 그대로 통과 (dropout 비활성화)
```

## 수동 구현

```python
class ManualDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)  # Inverted dropout
        return x
```

## Dropout 위치

```python
# 일반적인 배치
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)  # 활성화 후 dropout
        x = self.fc2(x)
        x = self.dropout(x)  # 두 번째 층 후에도
        return x
```

## DropPath (Stochastic Depth)

레이어 전체를 확률적으로 스킵:

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
            return x * mask / keep_prob
        return x

# ResNet/ViT에서 skip connection에 적용
class ResBlock(nn.Module):
    def forward(self, x):
        return x + self.drop_path(self.block(x))
```

## Monte Carlo Dropout

추론 시에도 dropout 유지 → 불확실성 추정:

```python
model.train()  # Dropout 활성화
predictions = []
for _ in range(100):
    pred = model(x)
    predictions.append(pred)

mean = torch.stack(predictions).mean(0)  # 예측값
std = torch.stack(predictions).std(0)    # 불확실성
```

## 관련 콘텐츠

- [Weight Decay](/ko/docs/components/training/regularization/weight-decay)
- [Batch Normalization](/ko/docs/components/normalization/batch-norm)
- [Transformer](/ko/docs/architecture/transformer)
