---
title: "Cross-Entropy Loss"
weight: 1
math: true
---

# Cross-Entropy Loss

{{% hint info %}}
**선수지식**: [엔트로피](/ko/docs/math/probability/entropy) | [확률분포](/ko/docs/math/probability/distribution) | [Softmax](/ko/docs/components/activation/softmax)
{{% /hint %}}

## 한 줄 요약
> **Cross-Entropy는 "정답 클래스에 높은 확률을 줄수록 보상이 커지는" 분류용 표준 손실 함수입니다.**

## 왜 필요한가?
분류 모델의 출력은 보통 클래스별 확률입니다. 
문제는 "얼마나 틀렸는지"를 숫자로 만들어야 역전파로 학습할 수 있다는 점입니다.

Cross-Entropy는
- 정답 확률이 낮으면 큰 패널티,
- 정답 확률이 높으면 작은 패널티를 주어,
모델이 정답 클래스 확률을 올리도록 유도합니다.

비유하면, 시험에서 정답 문항에 확신이 없을수록 더 크게 감점되는 채점 규칙과 비슷합니다.

## 수식

### 이진 분류
$$
L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

### 다중 분류
$$
L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

One-hot 라벨에서는 정답 클래스 $c^*$만 $y_{c^*}=1$ 이므로:
$$
L = -\log(\hat{y}_{c^*})
$$

**각 기호의 의미**
- $C$ : 클래스 개수
- $y_c$ : 클래스 $c$의 정답 라벨(보통 one-hot)
- $\hat{y}_c$ : 클래스 $c$로 예측한 확률
- $c^*$ : 정답 클래스 인덱스
- $L$ : 샘플 하나의 손실값

## 직관적 이해
- 정답 확률 $\hat{y}_{c^*}=1.0$ 이면 손실은 $0$
- 정답 확률 $0.1$ 이면 손실은 약 $2.30$
- 정답 확률 $0.01$ 이면 손실은 약 $4.61$

즉, **틀린 확신(confident wrong prediction)** 을 매우 강하게 벌점합니다.

## Softmax + Cross-Entropy (수치 안정성)
실무에서는 보통 logits(정규화 전 점수)를 바로 넣고, 프레임워크가 내부에서 안정적으로 계산하게 둡니다.

```python
import torch
import torch.nn.functional as F

B, C = 4, 3
logits = torch.tensor([
    [2.1, 0.3, -1.2],
    [0.1, 1.5, -0.2],
    [1.0, 0.2, 0.1],
    [-0.5, 0.4, 2.2],
], dtype=torch.float32)

targets = torch.tensor([0, 1, 0, 2])  # class index (one-hot 아님)

# 권장: 내부적으로 log-sum-exp trick 적용
loss = F.cross_entropy(logits, targets)
print(float(loss))
```

## Label Smoothing
정답 라벨을 100% 확신(one-hot)으로 두지 않고 약간 부드럽게 만드는 기법입니다.

$$
y'_c = (1 - \alpha) y_c + \frac{\alpha}{C}
$$

- $\alpha$ : smoothing 강도 (예: 0.1)
- 효과: 과적합 완화, calibration 개선

```python
import torch.nn as nn
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## 입력/출력 shape 빠른 기준표
초보자가 가장 자주 막히는 부분이 **"모델 출력 shape"와 "타깃 shape" 불일치**입니다.

| 상황 | 모델 출력(logits) | 타깃(target) | 비고 |
|---|---|---|---|
| 멀티클래스 분류(기본) | `(B, C)` | `(B,)` (`long`) | 클래스 인덱스 사용 |
| 시퀀스 분류(토큰 단위) | `(B, T, C)` | `(B, T)` (`long`) | 보통 `(B*T, C)`로 reshape 후 계산 |
| 이미지 분할(픽셀 단위) | `(B, C, H, W)` | `(B, H, W)` (`long`) | 채널 축 `C`는 logits 쪽에만 존재 |

> 핵심 기억: **CrossEntropyLoss는 one-hot 타깃이 아니라 클래스 인덱스 타깃을 기본으로 받습니다.**

## 미니 실습: 잘못된 사용 vs 올바른 사용
아래 코드는 같은 데이터를 두 방식으로 넣었을 때의 차이를 보여줍니다.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.1, -1.0]], dtype=torch.float32)  # (B=1, C=3)
target = torch.tensor([0], dtype=torch.long)                    # 정답 클래스 인덱스

# ✅ 올바른 사용: logits 그대로
loss_ok = F.cross_entropy(logits, target)

# ❌ 흔한 실수: softmax를 먼저 적용
probs = torch.softmax(logits, dim=1)
loss_wrong = F.cross_entropy(probs, target)

print("correct:", float(loss_ok))
print("wrong  :", float(loss_wrong))
```

왜 문제일까?
- `F.cross_entropy` 내부에 `log_softmax`가 이미 포함되어 있습니다.
- 바깥에서 softmax를 또 적용하면 gradient 스케일이 왜곡되어 학습이 느려지거나 불안정해질 수 있습니다.

## 디버깅 체크리스트
- [ ] `CrossEntropyLoss`에 **softmax를 미리 적용하지 않았는가?** (logits 그대로 넣기)
- [ ] 타깃 dtype이 `torch.long` 인가?
- [ ] 타깃 shape이 `(B,)` 인가? (`(B, C)` one-hot을 그대로 넣지 않기)
- [ ] 클래스 인덱스 범위가 `[0, C-1]` 인가?
- [ ] 손실이 NaN이면 logits 스케일 과대/학습률 과대 여부를 확인했는가?

## 자주 하는 실수 (FAQ)
**Q1. Cross-Entropy에 softmax를 먼저 적용해야 하나요?**  
A. 일반적으로 아닙니다. `F.cross_entropy`는 logits를 입력받아 내부에서 안정적으로 계산합니다.

**Q2. BCE와 Cross-Entropy는 어떻게 다른가요?**  
A. BCE는 주로 이진/멀티라벨, Cross-Entropy는 단일 정답 클래스 멀티클래스 분류에 주로 사용합니다.

**Q3. 정확도는 오르는데 손실이 잘 안 내려갑니다. 이상한가요?**  
A. 가능할 수 있습니다. 맞춘 샘플이 늘어도, 틀린 샘플에 매우 확신하면 손실은 크게 남을 수 있습니다.

## 증상 → 원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 확인할 것 |
|---|---|---|
| `Expected target size` 에러 | target shape 불일치 | `(B,)`인지, 불필요한 차원(`unsqueeze`)이 없는지 |
| `Target X is out of bounds` | 클래스 인덱스 범위 초과 | `num_classes=C`일 때 target이 `0~C-1`인지 |
| 손실이 초반부터 NaN | 과도한 LR, 잘못된 입력값 | learning rate, 입력에 NaN/Inf 존재 여부 |
| 정확도는 오르는데 calibration이 나쁨 | 과도한 확신 예측 | label smoothing, temperature scaling 검토 |

## 5분 점검 실험 (초보자용)
학습이 안 될 때 아래 순서로 **한 번에 하나만** 바꿔 보세요.

1. **입력만 점검**: logits/target shape, dtype 점검 후 1~2 step만 학습
2. **학습률 절반**: LR을 1/2로 줄여 NaN/진동이 사라지는지 확인
3. **Label Smoothing 0.1 적용**: 과확신이 줄어드는지 확인
4. **클래스 불균형 점검**: 특정 클래스만 계속 틀리면 class weight 또는 focal loss 검토

작게 바꿔야 원인을 분리할 수 있습니다. 여러 설정을 동시에 바꾸면 무엇이 개선 원인인지 알기 어렵습니다.

## 시각자료(나노바나나) 프롬프트
- **KO 다이어그램 1 (정답 확률 vs 손실 곡선)**  
  "다크 테마 배경(#1a1a2e), Cross-Entropy 손실 곡선 인포그래픽. x축: 정답 클래스 확률 p(y*), y축: loss=-log(p). p=1.0, 0.1, 0.01 지점 강조(각각 loss 0, 2.30, 4.61). 한국어 라벨: '확신 정답', '애매', '확신 오답'. 수식과 점선 가이드 포함, 깔끔한 벡터 스타일"

- **KO 다이어그램 2 (올바른 입력 파이프라인)**  
  "다크 테마 배경(#1a1a2e), 분류 학습 파이프라인 다이어그램. 순서: logits -> CrossEntropyLoss 내부(log_softmax + NLL) -> scalar loss -> backward. 잘못된 경로로 'softmax를 미리 적용'한 가지를 빨간 경고 아이콘으로 표시. 한국어 라벨과 화살표, 초보자용 비교표(올바름/잘못됨) 포함"

## 관련 콘텐츠
- [Focal Loss](/ko/docs/components/training/loss/focal-loss)
- [엔트로피](/ko/docs/math/probability/entropy)
- [확률분포](/ko/docs/math/probability/distribution)
- [Softmax](/ko/docs/components/activation/softmax)
