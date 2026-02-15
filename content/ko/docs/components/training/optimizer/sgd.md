---
title: "SGD"
weight: 1
math: true
---

# SGD (Stochastic Gradient Descent)

{{% hint info %}}
**선수지식**: [Gradient](/ko/docs/math/calculus/gradient)
{{% /hint %}}

## 한 줄 요약
> **SGD는 손실을 줄이는 방향(기울기의 반대 방향)으로 파라미터를 조금씩 업데이트하는 가장 기본적인 최적화 방법입니다.**

## 왜 필요한가?
딥러닝 학습의 핵심은 "모델 파라미터를 어떻게 고칠지"를 반복해서 결정하는 것입니다.
SGD는 이 과정을 가장 단순하고 해석 가능하게 수행합니다.

비유하면, 눈을 가린 채 산을 내려올 때 현재 발밑의 경사(gradient)를 느끼고 한 걸음 내려가는 방식입니다.

- 장점: 구현이 단순, 메모리 사용량이 작음, 대규모 CNN에서 일반화가 좋은 경우가 많음
- 단점: 학습률 설정에 민감, 진동/느린 수렴이 발생할 수 있음

## 기본 SGD

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

**기호 설명**
- $\theta_t$: $t$번째 스텝의 모델 파라미터
- $L(\theta_t)$: 현재 파라미터에서의 손실 함수
- $\nabla L(\theta_t)$: 손실의 기울기(어느 방향으로 손실이 커지는지)
- $\eta$: 학습률(learning rate)

### 직관
- 기울기는 "오르막 방향"이므로, 그 반대 방향으로 가면 손실이 줄어듭니다.
- $\eta$가 너무 크면 발을 너무 크게 디뎌서 최솟값을 지나치고,
- $\eta$가 너무 작으면 너무 천천히 학습됩니다.

```python
import torch

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for x, y in dataloader:
    optimizer.zero_grad()        # 이전 step gradient 초기화
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()              # gradient 계산
    optimizer.step()             # 파라미터 업데이트
```

## 초보자가 가장 헷갈리는 포인트: SGD vs Mini-batch SGD
이론식은 샘플 1개(진짜 stochastic)처럼 보이지만, 실무에서는 보통 **mini-batch 평균 gradient**를 사용합니다.
PyTorch 기본 `DataLoader` 학습 루프가 바로 이 형태입니다.

$$
g_t = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell_i(\theta_t),
\qquad
\theta_{t+1} = \theta_t - \eta g_t
$$

- $B$: 배치 크기
- $\ell_i$: $i$번째 샘플의 loss
- $g_t$: 배치 평균 gradient

직관:
- 배치가 너무 작으면 gradient 노이즈가 커져 loss가 출렁이기 쉽습니다.
- 배치가 너무 크면 메모리는 많이 쓰고, 일반화가 오히려 둔해질 수 있습니다.

실무 시작점(경험칙):
- 메모리 여유가 없으면 배치를 줄이고, 필요하면 gradient accumulation으로 **유효 배치 크기**를 늘립니다.
- 배치 크기를 2배로 키웠다면 learning rate도 소폭 상향(예: 1.5~2배)하며 안정성을 확인합니다.

### 배치 크기 변경 시, 왜 lr를 함께 조정할까?
같은 데이터라도 배치가 커지면 gradient 평균이 더 안정적(노이즈 감소)이라 한 step을 조금 더 크게 가도 되는 경우가 많습니다.
초보자는 아래를 **출발점**으로만 사용하세요.

$$
\eta_2 = \eta_1 \times \frac{B_2}{B_1}
$$

- $\eta_1, \eta_2$: 변경 전/후 learning rate
- $B_1, B_2$: 변경 전/후 배치 크기

주의:
- 이 식은 "항상 정답"이 아니라 초기 탐색용입니다.
- 실제로는 warmup, augmentation 강도, 모델 구조에 따라 1:1 비율이 과할 수 있습니다.
- 그래서 실무에서는 선형 스케일링 값을 중심으로 2~3개 후보(lr 낮음/중간/높음)를 짧게 비교합니다.

## SGD with Momentum
이전 업데이트 방향을 일부 기억해 진동을 줄이고 진행을 빠르게 만듭니다.

$$
\begin{aligned}
v_t &= \mu v_{t-1} + \nabla L(\theta_t),\\
\theta_{t+1} &= \theta_t - \eta v_t
\end{aligned}
$$

- $\mu$: 모멘텀 계수(보통 0.9)
- 효과: 좁고 긴 골짜기(loss valley)에서 좌우 진동 감소

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)
```

## Nesterov Momentum
"지금 속도로 한 걸음 먼저 가본 지점"에서 gradient를 계산해 보정합니다.

$$
v_t = \mu v_{t-1} + \nabla L(\theta_t - \eta \mu v_{t-1})
$$

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True,
)
```

## Weight Decay와 함께 쓰기
가중치가 과도하게 커지는 것을 막아 과적합을 줄이는 데 도움을 줍니다.

$$
\theta_{t+1} = \theta_t - \eta(\nabla L + \lambda \theta_t)
$$

- $\lambda$: 가중치 감쇠 계수(weight decay strength)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)
```

실무에서는 bias/정규화 파라미터에 weight decay를 제외하는 경우가 많습니다.

```python
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or n.endswith(".bias"):
        no_decay.append(p)   # BN/LN scale, bias 등
    else:
        decay.append(p)

optimizer = torch.optim.SGD(
    [
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=0.01,
    momentum=0.9,
    nesterov=True,
)
```

## SGD vs Adam (실무 관점)

| 항목 | SGD | Adam/AdamW |
|---|---|---|
| 초기 수렴 속도 | 느린 편 | 빠른 편 |
| 일반화 성능 | 좋은 경우가 많음(CNN) | 빠른 실험에 유리 |
| 하이퍼파라미터 민감도 | 높음 | 상대적으로 낮음 |
| 메모리 사용량 | 적음 | 더 큼 |

## 실무 디버깅 체크리스트
- [ ] **학습률 확인**: loss가 발산하면 lr을 10배 낮춰 테스트
- [ ] **gradient 폭주 확인**: `grad_norm`이 비정상적으로 큰지 로깅
- [ ] **모멘텀 점검**: 과진동 시 momentum(예: 0.9→0.85) 조정
- [ ] **weight decay 분리 적용**: norm/bias 파라미터에 동일 decay 적용 여부 점검
- [ ] **스케줄러 연동 확인**: warmup 없이 큰 lr로 시작해 불안정해지지 않았는지 확인

## 5분 미니 실습: 학습률 10배 차이 체감하기
아래 코드는 같은 모델/데이터에서 학습률만 바꿔 20 step 동안 loss 추세를 비교합니다.
초보자에게 가장 중요한 감각인 "SGD는 lr에 얼마나 민감한가"를 빠르게 체험할 수 있습니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim


def run_once(lr: float):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    losses = []
    for _ in range(20):
        x = torch.randn(128, 32)
        y = torch.randint(0, 10, (128,))

        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        losses.append(float(loss))

    return losses

loss_lr_1e2 = run_once(1e-2)
loss_lr_1e1 = run_once(1e-1)

print("lr=1e-2 first/last:", round(loss_lr_1e2[0], 4), round(loss_lr_1e2[-1], 4))
print("lr=1e-1 first/last:", round(loss_lr_1e1[0], 4), round(loss_lr_1e1[-1], 4))
```

읽는 법:
- `lr=1e-2`는 대체로 안정적으로 감소합니다.
- `lr=1e-1`은 더 빨리 줄 수도 있지만, 데이터/초기화에 따라 진동이 커질 수 있습니다.
- 실무에서는 이 "안정성 vs 속도" 균형을 scheduler와 함께 맞춥니다.

## 자주 하는 실수 (FAQ)
**Q1. SGD는 Adam보다 항상 느리고 나쁜가요?**  
A. 아닙니다. 초기 수렴은 느릴 수 있지만, 최종 일반화는 SGD가 더 좋은 경우가 자주 있습니다.

**Q2. momentum을 크게 하면 무조건 좋은가요?**  
A. 아닙니다. 너무 크면 오히려 overshoot(지나침)가 심해질 수 있습니다.

**Q3. loss가 오르내리는데 학습이 실패한 건가요?**  
A. mini-batch SGD에서는 자연스러운 현상입니다. 추세(이동 평균)가 내려가는지를 보세요.

## 초보자용 빠른 시작 레시피 (Vision 분류/검출 공통)
"어디서부터 시작해야 할지"를 줄이기 위한 최소 스타터입니다.

- Optimizer: `SGD(momentum=0.9, nesterov=True)`
- 초기 lr: `0.01` (배치 64 기준의 출발점)
- weight decay: `1e-4`
- 스케줄러: 3~5 epoch warmup 후 cosine decay

배치 크기를 바꿀 때 간단 경험칙:
- batch 2배 증가 → lr 1.5~2배 범위에서 테스트
- batch 1/2 감소 → lr도 1/2~2/3 수준으로 먼저 낮춰 안정성 확인

## 실패 패턴별 1차 대응 (운영 체크)
- **첫 100~300 step에서 loss 급등** → lr 10배 낮추고 warmup 추가
- **train은 개선되는데 val 정체** → weight decay/augmentation 강화
- **NaN 또는 inf gradient 발생** → grad clipping(예: `max_norm=1.0`) 임시 적용 후 원인 추적
- **학습이 너무 느림** → lr scheduler 확인, `momentum`/배치 크기 동시 재점검

## 증상 → 원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 확인할 것 |
|---|---|---|
| 초반부터 loss가 폭발 | 학습률 과대, warmup 없음 | lr 10배 축소 + warmup 유무 |
| loss는 줄지만 검증 성능 정체 | weight decay/augmentation 부족 | 정규화 강도, 데이터 증강 설정 |
| step마다 loss 진동이 심함 | momentum 과대 또는 배치 너무 작음 | momentum 0.9→0.85, batch size |
| 학습이 매우 느림 | lr 과소 또는 scheduler 미사용 | base lr 상향, cosine/step scheduler |

## 관련 콘텐츠
- [Adam](/ko/docs/components/training/optimizer/adam)
- [LR Scheduler](/ko/docs/components/training/optimizer/lr-scheduler)
- [Weight Decay](/ko/docs/components/training/regularization/weight-decay)
