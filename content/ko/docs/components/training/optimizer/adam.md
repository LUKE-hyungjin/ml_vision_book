---
title: "Adam"
weight: 2
math: true
---

# Adam & AdamW

## 개요

Adam은 적응적 학습률을 사용하는 optimizer로, 파라미터별로 다른 학습률을 적용합니다.

## Adam 알고리즘

**1차 모멘트** (Gradient의 지수이동평균):
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

**2차 모멘트** (Gradient 제곱의 지수이동평균):
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

**편향 보정**:
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

**업데이트**:
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

## 구현

```python
import torch

# 기본 Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),  # β1, β2
    eps=1e-8
)
```

## 적응적 학습률의 의미

$$
\text{effective lr} = \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}
$$

- Gradient가 일관되면 (v 작음) → 큰 스텝
- Gradient가 변동 크면 (v 큼) → 작은 스텝
- 파라미터별로 자동 조절

## AdamW (Weight Decay 분리)

Adam의 문제: Weight decay가 적응적 학습률과 결합됨

AdamW 해결책: Weight decay를 별도로 적용

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

```python
# AdamW (Transformer 학습에 권장)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01  # 분리된 weight decay
)
```

## Adam vs AdamW

| | Adam + L2 | AdamW |
|---|---|---|
| Weight decay | Gradient에 포함 | 별도 적용 |
| 적응적 학습률 영향 | 받음 | 안 받음 |
| Transformer 성능 | 낮음 | 높음 |

**결론**: 거의 모든 경우 AdamW 사용 권장

## 하이퍼파라미터

- **lr**: 1e-4 ~ 3e-4 (Transformer), 1e-3 (작은 모델)
- **β1**: 0.9 (기본)
- **β2**: 0.999 (기본), 0.95 (LLM에서 가끔)
- **weight_decay**: 0.01 ~ 0.1

## 8-bit Adam

메모리 절약을 위한 양자화 버전:

```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4
)
# 상태를 8비트로 저장 → 메모리 75% 절약
```

## 관련 콘텐츠

- [SGD](/ko/docs/components/training/optimizer/sgd)
- [LR Scheduler](/ko/docs/components/training/optimizer/lr-scheduler)
- [LoRA](/ko/docs/components/training/peft/lora) - AdamW와 함께 사용
