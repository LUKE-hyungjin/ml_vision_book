---
title: "Adam"
weight: 2
math: true
---

# Adam & AdamW

{{% hint info %}}
**선수지식**: [SGD](/ko/docs/components/training/optimizer/sgd), [Weight Decay](/ko/docs/components/training/regularization/weight-decay)
{{% /hint %}}

## 한 줄 요약
> **Adam은 파라미터마다 학습률을 자동 조절하는 optimizer이고, 실무에서는 대부분 AdamW(Weight Decay 분리 버전)를 사용합니다.**

## 왜 필요한가?
딥러닝 학습에서 가장 흔한 문제는 "같은 학습률을 모든 파라미터에 똑같이 적용하면 학습이 불안정해진다"는 점입니다.

- 어떤 파라미터는 gradient가 자주 크게 튀고
- 어떤 파라미터는 gradient가 매우 작아 거의 안 움직입니다.

Adam은 이 차이를 자동으로 반영해, 파라미터별로 스텝 크기를 다르게 조절합니다.
비유하면, 울퉁불퉁한 산길에서 **미끄러운 구간은 천천히, 평평한 구간은 빠르게** 걷도록 보폭을 자동 조절하는 것과 같습니다.

## 핵심 수식

시간 step $t$에서 gradient를 $g_t$라고 하면:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

초기 step에서 평균이 0 쪽으로 치우치는 문제를 고치기 위해 편향 보정을 합니다.

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

최종 업데이트:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

### 각 기호의 의미
- $\theta_t$ : 현재 파라미터
- $g_t$ : 현재 step의 gradient
- $m_t$ : gradient의 지수이동평균(1차 모멘트)
- $v_t$ : gradient 제곱의 지수이동평균(2차 모멘트)
- $\beta_1$ : 1차 모멘트 감쇠 계수(보통 0.9)
- $\beta_2$ : 2차 모멘트 감쇠 계수(보통 0.999)
- $\eta$ : 기본 학습률
- $\epsilon$ : 0 나누기 방지용 작은 상수

## 직관: Adam이 "자동 보폭 조절"을 하는 방법
Adam의 유효 학습률은 아래처럼 볼 수 있습니다.

$$
\text{effective lr} = \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}
$$

- $\hat{v}_t$가 크다(gradient 변동이 큼) → 분모가 커져 스텝이 작아짐(안정화)
- $\hat{v}_t$가 작다(gradient 변동이 작음) → 스텝이 상대적으로 커짐(학습 가속)

즉, Adam은 "진동이 큰 파라미터는 조심히", "안정적인 파라미터는 과감히" 업데이트합니다.

## AdamW가 필요한 이유
Adam에 단순 L2 정규화를 섞으면, 가중치 감소가 적응적 학습률과 뒤엉켜 의도와 다르게 동작할 수 있습니다.
AdamW는 Weight Decay를 업데이트식에서 **분리(decoupled)** 해서 적용합니다.

실무 관점 요약:
- Transformer/ViT/LLM 계열: 거의 기본 선택이 AdamW
- Adam을 써야 할 특별한 이유가 없다면 AdamW부터 시작

## 구현

```python
import torch

# AdamW (실무 기본 추천)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

## Adam vs AdamW

| 항목 | Adam + L2 | AdamW |
|---|---|---|
| Weight decay 적용 방식 | gradient에 섞여 적용 | 파라미터에 분리 적용 |
| 적응적 학습률과 상호작용 | 큼 | 작음(의도한 decay 유지) |
| Transformer 계열 기본 선택 | 보통 비권장 | 권장 |

## 시작 하이퍼파라미터 (초보자용)
- **lr**: `1e-4` (Transformer/ViT), 작은 모델은 `1e-3`도 자주 사용
- **betas**: `(0.9, 0.999)` 기본값으로 시작
- **weight_decay**: `0.01`부터 시작 후 과적합이면 증가 검토
- **eps**: 기본 `1e-8` 유지 (수치 불안정 시 로그 확인)

## 8-bit Adam (메모리 절약)
옵티마이저 상태(state)가 메모리를 많이 쓰는 대형 모델에서는 8-bit Adam을 사용하기도 합니다.

```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4,
)
```

장점: optimizer state 메모리 절감
주의: 라이브러리/하드웨어 조합에 따라 수치 거동이 달라질 수 있으니 짧은 검증 실험이 필요합니다.

## 디버깅 체크리스트
- [ ] loss가 초반부터 출렁이면 `lr`를 2~10배 낮춰봤는가?
- [ ] validation이 나빠지면 `weight_decay`가 너무 작은지 확인했는가?
- [ ] gradient norm이 자주 폭주하면 clip(`max_norm`)을 적용했는가?
- [ ] mixed precision 환경에서 NaN이 나면 scaler/autocast 설정을 점검했는가?
- [ ] Adam vs AdamW를 혼동해 잘못된 baseline을 비교하고 있지 않은가?

## 자주 하는 실수 (FAQ)
**Q1. Adam이 항상 SGD보다 좋은가요?**  
A. 아닙니다. 수렴 속도는 Adam/AdamW가 빠른 경우가 많지만, 태스크에 따라 최종 일반화 성능은 SGD가 더 나을 때도 있습니다.

**Q2. AdamW면 weight decay를 무조건 크게 주는 게 좋은가요?**  
A. 아닙니다. 너무 크면 underfitting이 생깁니다. 보통 0.01 전후에서 시작해 검증 성능으로 조정합니다.

**Q3. β1, β2를 자주 바꿔야 하나요?**  
A. 초보자 단계에서는 기본값 `(0.9, 0.999)`을 유지하고, 먼저 lr/weight_decay를 튜닝하는 편이 효율적입니다.

## 관련 콘텐츠
- [SGD](/ko/docs/components/training/optimizer/sgd)
- [LR Scheduler](/ko/docs/components/training/optimizer/lr-scheduler)
- [Weight Decay](/ko/docs/components/training/regularization/weight-decay)
- [LoRA](/ko/docs/components/training/peft/lora)
