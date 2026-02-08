---
title: "Weight Decay"
weight: 2
math: true
---

# Weight Decay (L2 Regularization)

## 개요

Weight Decay는 가중치가 커지는 것을 방지하여 모델이 단순한 해를 찾도록 유도합니다.

## 수식

손실 함수에 가중치 크기 페널티 추가:

$$
L_{total} = L_{data} + \frac{\lambda}{2} \sum_i w_i^2
$$

Gradient에 미치는 영향:

$$
\frac{\partial L_{total}}{\partial w} = \frac{\partial L_{data}}{\partial w} + \lambda w
$$

업데이트:

$$
w_{t+1} = w_t - \eta \left( \frac{\partial L}{\partial w} + \lambda w_t \right) = (1 - \eta\lambda)w_t - \eta \frac{\partial L}{\partial w}
$$

## L2 vs Weight Decay

**L2 정규화**: 손실에 추가 → optimizer가 gradient 계산
**Weight Decay**: optimizer에서 직접 감쇠

SGD에서는 동일하지만, Adam에서는 다름 (AdamW 필요)

## 구현

```python
import torch

# SGD with Weight Decay
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4  # λ
)

# AdamW (분리된 weight decay)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

## 파라미터별 Weight Decay

특정 파라미터에만 적용 (bias, norm에는 보통 적용 안 함):

```python
def get_parameter_groups(model, weight_decay=0.01):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

optimizer = torch.optim.AdamW(
    get_parameter_groups(model, weight_decay=0.01),
    lr=1e-4
)
```

## 베이지안 해석

Weight Decay = 가중치에 가우시안 사전 분포:

$$
P(w) \propto \exp(-\frac{\lambda}{2}w^2)
$$

MAP(Maximum A Posteriori) 추정과 동일

## 일반적인 값

| 모델 | Weight Decay |
|------|-------------|
| CNN (ImageNet) | 1e-4 |
| Transformer | 0.01 ~ 0.1 |
| Fine-tuning | 0.01 |
| ViT | 0.05 ~ 0.3 |

## 관련 콘텐츠

- [Dropout](/ko/docs/components/training/regularization/dropout)
- [Adam](/ko/docs/components/training/optimizer/adam)
- [베이즈 정리](/ko/docs/math/probability/bayes)
