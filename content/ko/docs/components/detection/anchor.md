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

## 관련 콘텐츠
- [IoU](/ko/docs/components/detection/iou)
- [NMS](/ko/docs/components/detection/nms)
- [YOLO](/ko/docs/architecture/detection/yolo)
