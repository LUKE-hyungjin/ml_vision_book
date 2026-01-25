---
title: "SGD"
weight: 1
math: true
---

# SGD (Stochastic Gradient Descent)

## 개요

SGD는 가장 기본적인 최적화 알고리즘으로, gradient 방향으로 파라미터를 업데이트합니다.

## 기본 SGD

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

```python
import torch

# 기본 SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

## SGD with Momentum

이전 업데이트 방향을 기억하여 진동 감소:

$$
v_t = \mu v_{t-1} + \nabla L(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

- **μ**: Momentum 계수 (보통 0.9)
- 효과: 골짜기에서 빠른 진행, 진동 감소

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

### 직관

- 공이 언덕을 굴러 내려가는 것과 유사
- 관성으로 인해 지역 최소점 탈출 가능
- 일관된 방향으로 가속

## Nesterov Momentum

"미리 보고" gradient 계산:

$$
v_t = \mu v_{t-1} + \nabla L(\theta_t - \eta \mu v_{t-1})
$$

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov 활성화
)
```

## Weight Decay

L2 정규화 효과:

$$
\theta_{t+1} = \theta_t - \eta(\nabla L + \lambda \theta_t)
$$

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4  # L2 regularization
)
```

## SGD vs Adam

| | SGD | Adam |
|---|---|---|
| 수렴 속도 | 느림 | 빠름 |
| 일반화 | 더 좋음 | 가끔 과적합 |
| 하이퍼파라미터 | 민감 | 덜 민감 |
| 메모리 | 적음 | 많음 |

**권장**:
- **CNN 대규모 학습**: SGD + Momentum
- **Transformer/빠른 실험**: Adam/AdamW

## 실전 설정

```python
# ImageNet 표준 설정
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,           # 큰 학습률
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# Cosine LR decay와 함께
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs
)
```

## 관련 콘텐츠

- [Adam](/ko/docs/math/training/optimizer/adam)
- [LR Scheduler](/ko/docs/math/training/optimizer/lr-scheduler)
- [Weight Decay](/ko/docs/math/training/regularization/weight-decay)
