---
title: "Learning Rate Scheduler"
weight: 3
math: true
---

# Learning Rate Scheduler

## 개요

학습률을 학습 중 동적으로 조절하여 수렴을 개선합니다.

## 왜 필요한가?

- **초기**: 큰 학습률로 빠른 탐색
- **후기**: 작은 학습률로 세밀한 수렴
- 고정 학습률은 비효율적

## Step Decay

일정 간격으로 학습률 감소:

```python
# 30 에폭마다 0.1배
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# 학습 루프
for epoch in range(100):
    train(...)
    scheduler.step()  # 에폭 끝에 호출
```

## Multi-Step Decay

지정된 에폭에서 감소:

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[60, 80, 90],  # 이 에폭에서
    gamma=0.1                  # 0.1배
)
```

## Cosine Annealing

부드러운 코사인 곡선으로 감소:

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))
$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,        # 총 에폭
    eta_min=1e-6      # 최소 학습률
)
```

**장점**: 가장 널리 사용, 안정적 수렴

## Cosine with Warmup

초기에 학습률을 점진적으로 높임:

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,    # Warmup 스텝
    num_training_steps=10000  # 총 스텝
)

# 스텝마다 호출
for step, batch in enumerate(dataloader):
    loss.backward()
    optimizer.step()
    scheduler.step()  # 매 스텝!
```

## Warmup의 중요성

- **초기 불안정**: 큰 gradient로 발산 가능
- **해결**: 작은 lr → 점진적 증가
- **Transformer 필수**: Pre-LN 없으면 특히 중요

```
lr
 ^
 |    /‾‾‾‾‾‾‾‾‾‾\
 |   /            \
 |  /              \
 | /                \
 +---------------------> steps
   warmup   decay
```

## Linear Warmup + Linear Decay

```python
def get_linear_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0, (total_steps - step) / (total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## OneCycleLR

Warmup + Decay를 한 사이클로:

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=len(dataloader) * epochs,
    pct_start=0.3  # 30%까지 warmup
)
```

## 실전 가이드

| 모델 | 권장 스케줄러 |
|------|-------------|
| CNN (ImageNet) | Step 또는 Cosine |
| Transformer | Cosine + Warmup |
| Fine-tuning | Linear decay |
| 짧은 학습 | OneCycle |

## 관련 콘텐츠

- [SGD](/ko/docs/components/training/optimizer/sgd)
- [Adam](/ko/docs/components/training/optimizer/adam)
- [ViT](/ko/docs/architecture/transformer/vit)
