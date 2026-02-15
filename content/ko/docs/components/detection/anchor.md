---
title: "Anchor Box"
weight: 3
math: true
---

# Anchor Box

{{% hint info %}}
**선수지식**: [IoU](/ko/docs/components/detection/iou)
{{% /hint %}}

## 한 줄 요약
> **Anchor Box는 "기준 박스"를 먼저 깔아 두고, 모델이 그 기준에서 얼마나 이동/확대해야 하는지만 예측하게 만드는 방식입니다.**

## 왜 필요한가?
객체 검출에서 박스를 완전히 새로 예측하면 학습 초기에 불안정해지기 쉽습니다.

Anchor 방식은 먼저 여러 크기/비율의 "기준 박스"를 만들고,
모델은 기준 박스와 정답 박스의 **차이(delta)** 만 예측합니다.

- 장점 1: 학습이 더 안정적
- 장점 2: 작은 물체/큰 물체를 동시에 다루기 쉬움
- 사용 예: Faster R-CNN, RetinaNet, SSD 계열

## 수식/기호
Anchor $(x_a, y_a, w_a, h_a)$와 정답 박스 $(x, y, w, h)$가 있을 때, 회귀 타깃은 다음과 같습니다.

$$
\begin{aligned}
t_x &= \frac{x - x_a}{w_a}, \\
t_y &= \frac{y - y_a}{h_a}, \\
t_w &= \log\left(\frac{w}{w_a}\right), \\
t_h &= \log\left(\frac{h}{h_a}\right)
\end{aligned}
$$

**각 기호의 의미:**
- $(x, y)$ : 정답 박스 중심 좌표
- $(w, h)$ : 정답 박스 너비/높이
- $(x_a, y_a)$ : anchor 중심 좌표
- $(w_a, h_a)$ : anchor 너비/높이
- $(t_x, t_y, t_w, t_h)$ : 모델이 예측할 오프셋(변환량)

디코딩(역변환)은 다음처럼 합니다.

$$
\begin{aligned}
\hat{x} &= t_x w_a + x_a, \\
\hat{y} &= t_y h_a + y_a, \\
\hat{w} &= e^{t_w} w_a, \\
\hat{h} &= e^{t_h} h_a
\end{aligned}
$$

## 직관
사진 위에 서로 다른 모양의 스티커(anchor)를 미리 붙여 둔다고 생각하면 쉽습니다.

- 정답과 모양이 비슷한 스티커는 "조금만" 움직이면 맞음
- 정답과 많이 다른 스티커는 "많이" 변형해야 함

즉, "완전한 박스 생성" 대신 "기준 대비 보정" 문제로 바꾸는 것이 핵심입니다.

### Anchor 할당(매칭) 직관
실제 학습에서는 각 anchor를 정답 박스와 매칭해 **양성/음성** 라벨을 만듭니다.

- 보통 $\mathrm{IoU} \ge \tau_{pos}$ 이면 양성(positive)
- 보통 $\mathrm{IoU} < \tau_{neg}$ 이면 음성(negative)
- 그 사이 구간은 무시(ignore)해 학습 노이즈를 줄임

예를 들어 $(\tau_{pos}, \tau_{neg}) = (0.7, 0.3)$이면:

| Anchor IoU | 라벨 | 의미 |
|---|---|---|
| 0.82 | Positive | 이 anchor로 박스 회귀 + 분류를 함께 학습 |
| 0.55 | Ignore | 애매해서 손실 계산에서 제외 |
| 0.12 | Negative | 배경으로 분류 학습 |

## 구현 전에 알아둘 실무 포인트
1. **Anchor 스케일/비율 설계**
   - 보통 feature map 레벨마다 서로 다른 스케일을 둡니다. (예: P3는 작은 물체, P7은 큰 물체)
   - 종횡비(aspect ratio)는 데이터셋 통계(가로형/세로형 비중)에 맞춰야 합니다.

2. **회귀 타깃 정규화**
   - 구현에서는 $(t_x, t_y, t_w, t_h)$에 평균/표준편차를 적용해 학습을 안정화하기도 합니다.
   - 예: $t'_x = (t_x - \mu_x)/\sigma_x$ 형태로 스케일을 맞춘 뒤 loss를 계산합니다.

3. **경계 케이스 처리**
   - 매우 작은 anchor나 잘못된 박스가 들어오면 $\log(w/w_a)$에서 수치 불안정이 생길 수 있습니다.
   - 그래서 코드에서 `np.clip(..., 1e-6, None)`처럼 최소값을 고정합니다.

4. **양성/음성 불균형 완화**
   - 실제로는 negative anchor가 훨씬 많아 분류 loss가 배경에 치우치기 쉽습니다.
   - 그래서 mini-batch에서 positive:negative 비율(예: 1:3)을 맞추거나, hard negative mining/focal loss를 함께 사용합니다.

## 3분 미니 연습: 수식이 실제 숫자로 어떻게 움직이나?
아래처럼 anchor와 정답 박스를 하나만 두고 직접 계산해 보면, 회귀 타깃이 왜 필요한지 빠르게 감이 옵니다.

- Anchor(중심/크기): $(x_a, y_a, w_a, h_a) = (50, 50, 40, 20)$
- GT(중심/크기): $(x, y, w, h) = (58, 46, 50, 24)$

수식에 대입하면:
$$
\begin{aligned}
t_x &= \frac{58-50}{40}=0.20,\\
t_y &= \frac{46-50}{20}=-0.20,\\
t_w &= \log\left(\frac{50}{40}\right)\approx 0.223,\\
t_h &= \log\left(\frac{24}{20}\right)\approx 0.182
\end{aligned}
$$

해석:
- $t_x=0.20$ → anchor 너비의 20%만큼 **오른쪽으로 이동**
- $t_y=-0.20$ → anchor 높이의 20%만큼 **위로 이동**
- $t_w, t_h > 0$ → **가로/세로를 조금 키워야 함**

즉 모델은 "절대 좌표"를 외우는 게 아니라, "기준(anchor) 대비 얼마나 고칠지"를 학습합니다.

## 구현
```python
import numpy as np


def encode_boxes(anchors, gt_boxes):
    """
    anchors, gt_boxes: (N, 4) where each box is [x1, y1, x2, y2]
    return: (N, 4) [tx, ty, tw, th]
    """
    # Anchor center/size
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha

    # GT center/size
    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    x = gt_boxes[:, 0] + 0.5 * w
    y = gt_boxes[:, 1] + 0.5 * h

    # Delta targets
    tx = (x - xa) / np.clip(wa, 1e-6, None)
    ty = (y - ya) / np.clip(ha, 1e-6, None)
    tw = np.log(np.clip(w, 1e-6, None) / np.clip(wa, 1e-6, None))
    th = np.log(np.clip(h, 1e-6, None) / np.clip(ha, 1e-6, None))

    return np.stack([tx, ty, tw, th], axis=1)
```

## 미니 구현: IoU 기반 Anchor 라벨링
앞의 `encode_boxes`는 "회귀 타깃"만 만듭니다. 실제 학습 파이프라인에서는 먼저
각 anchor가 positive/negative/ignore 중 어디에 속하는지 라벨링해야 합니다.

```python
import numpy as np


def assign_anchor_labels(iou_max, pos_thr=0.7, neg_thr=0.3):
    """
    iou_max: (N,) 각 anchor의 최대 IoU
    return: labels (N,) where 1=positive, 0=negative, -1=ignore
    """
    labels = np.full_like(iou_max, fill_value=-1, dtype=np.int64)  # 기본 ignore
    labels[iou_max >= pos_thr] = 1
    labels[iou_max < neg_thr] = 0
    return labels


# 예시: anchor 8개의 최대 IoU
iou_max = np.array([0.82, 0.74, 0.55, 0.40, 0.29, 0.10, 0.68, 0.02])
labels = assign_anchor_labels(iou_max, pos_thr=0.7, neg_thr=0.3)

num_pos = (labels == 1).sum()
num_neg = (labels == 0).sum()
num_ign = (labels == -1).sum()

print(f"pos={num_pos}, neg={num_neg}, ignore={num_ign}")
# pos=2, neg=3, ignore=3
```

핵심 해석:
- positive가 너무 적으면(예: 배치당 거의 0개) 박스 회귀가 잘 안 배웁니다.
- 반대로 negative가 과도하면 분류가 "배경만 잘 맞추는" 방향으로 치우칠 수 있습니다.
- 그래서 임계값(`pos_thr`, `neg_thr`)과 샘플링 비율(예: pos:neg=1:3)을 함께 조정합니다.

## 실무 디버깅 체크리스트
- [ ] **좌표계 통일**: `xyxy` vs `cxcywh`가 섞이지 않았는가?
- [ ] **스케일 매칭**: 작은 물체가 많은데 큰 anchor 비중이 과도하지 않은가?
- [ ] **양성 비율 확인**: 배치당 positive anchor 수가 너무 적지 않은가? (예: 1% 미만이면 학습 정체 가능)
- [ ] **회귀 폭주 감시**: `tw`, `th`가 초반부터 큰 절댓값으로 튀지 않는가?
- [ ] **매칭 임계값 재점검**: $\tau_{pos}, \tau_{neg}$ 설정이 데이터 난이도에 맞는가?

## 자주 하는 실수 (FAQ)
**Q1. Anchor를 많이 깔면 무조건 성능이 좋아지나요?**  
A. 아닙니다. 너무 많으면 negative가 급증해 분류가 배경 위주로 학습될 수 있습니다. 먼저 데이터 분포에 맞는 스케일/비율부터 줄여서 시작하는 편이 안정적입니다.

**Q2. IoU 기준을 높이면 항상 더 좋은 모델인가요?**  
A. 반드시 그렇진 않습니다. 너무 높은 기준은 positive 수를 줄여 학습 신호가 약해질 수 있습니다. 특히 소형 객체가 많은 데이터셋에서는 중간 임계값부터 점진적으로 올리는 전략이 유효합니다.

**Q3. 회귀 loss가 NaN이 됩니다. 어디부터 봐야 하나요?**  
A. (1) 박스 폭/높이가 음수인지, (2) `log(w/w_a)` 입력이 0 이하인지, (3) clip/eps 적용 여부를 먼저 확인하세요.

## 증상→원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 볼 항목 |
|---|---|---|
| 리콜은 낮고 FP는 많은데 loss는 내려감 | anchor 비율/스케일이 데이터 분포와 불일치 | 클래스별 박스 종횡비 히스토그램, FPN 레벨별 positive 수 |
| 박스가 한쪽으로 계속 치우침 | 좌표 정규화/디코딩 스케일 불일치 | target 정규화(mean/std), decode 수식 구현 |
| 초반부터 box loss가 비정상적으로 큼 | 잘못된 매칭 임계값 또는 flipped box 유입 | $\tau_{pos}, \tau_{neg}$, `x1<x2, y1<y2` 검증 |
| 특정 크기 물체만 계속 놓침 | 대응 스케일 anchor 부족 | 누락 객체 크기 분포 vs anchor scale 커버리지 |

## 빠른 실험: 임계값을 바꾸면 positive 개수가 어떻게 달라질까?
아래 코드는 같은 IoU 분포에서 `pos_thr`, `neg_thr`를 바꿨을 때
positive/negative/ignore 수가 어떻게 변하는지 보여줍니다.

```python
import numpy as np


def count_labels(iou_max, pos_thr, neg_thr):
    labels = np.full_like(iou_max, -1, dtype=np.int64)
    labels[iou_max >= pos_thr] = 1
    labels[iou_max < neg_thr] = 0
    return {
        "pos": int((labels == 1).sum()),
        "neg": int((labels == 0).sum()),
        "ignore": int((labels == -1).sum()),
    }


iou_max = np.array([0.82, 0.74, 0.55, 0.40, 0.29, 0.10, 0.68, 0.02])

for pos_thr, neg_thr in [(0.7, 0.3), (0.6, 0.3), (0.5, 0.4)]:
    print((pos_thr, neg_thr), count_labels(iou_max, pos_thr, neg_thr))
```

해석 가이드:
- `pos_thr`를 낮추면 positive가 늘어 회귀 학습 신호는 강해지지만, 라벨 노이즈도 늘 수 있습니다.
- `neg_thr`를 올리면 negative가 줄고 ignore가 늘어 분류 난이도는 낮아질 수 있습니다.
- 초보자는 **positive가 너무 적은지(배치당 0~몇 개 수준)**부터 먼저 확인하세요.

## 1분 자기점검 (초보자용)
- [ ] 내 데이터셋의 박스 크기 분포(작음/중간/큼)를 한 번이라도 히스토그램으로 확인했다.
- [ ] 학습 로그에서 배치당 positive anchor 개수를 출력해 봤다.
- [ ] `pos_thr`, `neg_thr`를 한 번에 크게 바꾸지 않고 0.05~0.1 단위로 조정했다.
- [ ] NaN이 나면 `w,h>0`과 `log` 입력 안전장치(`clip/eps`)부터 확인한다.

## 관련 콘텐츠
- [IoU](/ko/docs/components/detection/iou)
- [NMS](/ko/docs/components/detection/nms)
- [YOLO](/ko/docs/architecture/detection/yolo)
