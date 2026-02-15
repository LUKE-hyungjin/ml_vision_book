---
title: "IoU"
weight: 1
math: true
---

# IoU (Intersection over Union)

{{% hint info %}}
**선수지식**: 없음
{{% /hint %}}

## 한 줄 요약
> **IoU는 예측 박스와 정답 박스가 얼마나 겹치는지를 0~1로 수치화한 지표입니다.**

## 왜 필요한가?
객체 검출에서는 모델이 "대충 비슷한 위치"를 찾았는지, "정확히 맞췄는지"를 구분해야 합니다.
IoU는 이 겹침 정도를 공통 기준으로 만들어 주기 때문에:

- 평가(AP, mAP)의 기준이 되고
- 학습 중 양성/음성 샘플을 나누는 기준이 되며
- [NMS](/ko/docs/components/detection/nms)에서 중복 박스를 제거하는 기준이 됩니다.

비유하면, 정답 박스와 예측 박스를 종이에 겹쳐 놓았을 때 **겹친 면적 비율**을 보는 것입니다.

## 수식/기호
$$
\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

**각 기호의 의미:**
- $A$ : 예측 bounding box 영역
- $B$ : 정답(ground truth) bounding box 영역
- $|A \cap B|$ : 교집합(겹친 부분) 면적
- $|A \cup B|$ : 합집합(두 영역 전체) 면적

범위는 $0 \le \mathrm{IoU} \le 1$ 입니다.

- 0: 전혀 안 겹침
- 1: 완전히 일치

## 직관
- **교집합이 크고 합집합이 작을수록** IoU가 커집니다.
- 박스가 살짝만 어긋나도 교집합이 크게 줄어 IoU가 빠르게 떨어집니다.
- 그래서 IoU 임계값(예: 0.5, 0.75)에 따라 "맞았다/틀렸다" 판정이 달라집니다.

## 숫자로 보는 예시
두 박스의 면적이 각각 100이고, 겹친 면적이 60이라고 가정해봅시다.

- 교집합: $|A \cap B| = 60$
- 합집합: $|A \cup B| = 100 + 100 - 60 = 140$
- 따라서 $\mathrm{IoU} = 60/140 \approx 0.43$

이 경우 IoU 0.5 기준에서는 TP가 아니고, 0.4 기준에서는 TP가 될 수 있습니다.
즉, **같은 예측이라도 임계값에 따라 판정이 달라진다**는 점이 핵심입니다.

## 임계값에 따른 판정 변화
아래처럼 같은 IoU 값이라도 기준에 따라 결과가 달라집니다.

| IoU 값 | 임계값 0.5 | 임계값 0.75 |
|---:|---|---|
| 0.43 | FP (미탐지) | FP (미탐지) |
| 0.62 | TP | FP (엄격 기준에서는 실패) |
| 0.81 | TP | TP |

평가 기준이 엄격해질수록(예: 0.75) **정확한 위치 정렬**이 더 중요해집니다.

## 좌표 형식에서 자주 하는 실수
IoU 수식은 같아도 **박스 표현 방식**이 다르면 값이 틀어집니다.

- `xyxy`: `[x1, y1, x2, y2]` (좌상단/우하단)
- `cxcywh`: `[cx, cy, w, h]` (중심점/너비/높이)

모델 출력이 `cxcywh`인데 `xyxy`로 바로 계산하면 교집합이 잘못 계산됩니다.
실무에서는 **IoU 계산 전에 좌표계를 먼저 통일**하는 습관이 중요합니다.

간단 변환식:
$$
\begin{aligned}
x_1 &= c_x - \frac{w}{2}, \quad y_1 = c_y - \frac{h}{2},\\
x_2 &= c_x + \frac{w}{2}, \quad y_2 = c_y + \frac{h}{2}
\end{aligned}
$$

## 경계 케이스(실무 중요)
IoU 구현에서 성능보다 먼저 챙겨야 하는 건 **안전한 수치 처리**입니다.

- 박스가 뒤집힌 경우(`x2 < x1` 또는 `y2 < y1`)는 면적이 0이 되도록 `max(0, ...)` 처리
- 두 박스가 모두 비정상이라 합집합이 0이 되면, `0으로 나누기`를 피하려고 IoU를 0으로 반환
- 부동소수점 오차로 아주 작은 음수가 나올 수 있으므로, 필요하면 `eps`(예: `1e-9`)를 더해 안정화

이 세 가지만 지켜도 학습 중 NaN 전파를 크게 줄일 수 있습니다.

## IoU의 한계와 확장 지표
IoU는 직관적이지만, 박스가 **전혀 겹치지 않으면 항상 0**이라서
"얼마나 멀리 떨어졌는지" 정보를 주지 못합니다.

그래서 박스 회귀 손실에서는 다음 확장 지표를 자주 씁니다.

- **GIoU**: IoU에 "가장 작은 외접 박스" 정보를 추가
- **DIoU/CIoU**: 중심점 거리(그리고 종횡비)까지 반영

GIoU의 대표식은 다음과 같습니다.
$$
\mathrm{GIoU} = \mathrm{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
$$

**추가 기호:**
- $C$ : 박스 $A, B$를 모두 포함하는 가장 작은 축정렬 박스
- $|C \setminus (A \cup B)|$ : 외접 박스 안에서 두 박스가 차지하지 못한 빈 영역

겹침이 0이어도 GIoU는 음수가 될 수 있어, 학습 시 "어느 방향이 더 나쁜지"를 구분하기 쉬워집니다.

## 구현
```python
def box_iou(box1, box2):
    """
    box: [x1, y1, x2, y2] (좌상단, 우하단)
    """
    # 1) 교집합 좌표
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    # 2) 교집합 면적
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    # 3) 각 박스 면적
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    # 4) 합집합 면적
    union = area1 + area2 - inter

    # 5) IoU
    return inter / union if union > 0 else 0.0
```

## 미니 실습: 배치 IoU + 빠른 sanity check
아래 코드는 여러 박스 쌍의 IoU를 한 번에 계산하고,
초보자가 자주 놓치는 "좌표 뒤집힘/분모 0"를 즉시 확인할 수 있는 최소 예시입니다.

```python
import torch


def pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9):
    """
    boxes1, boxes2: [N, 4] in xyxy format
    returns: [N] IoU for pairwise rows
    """
    # 1) 좌표 순서 보정 (x1<=x2, y1<=y2)
    x11 = torch.minimum(boxes1[:, 0], boxes1[:, 2])
    y11 = torch.minimum(boxes1[:, 1], boxes1[:, 3])
    x12 = torch.maximum(boxes1[:, 0], boxes1[:, 2])
    y12 = torch.maximum(boxes1[:, 1], boxes1[:, 3])

    x21 = torch.minimum(boxes2[:, 0], boxes2[:, 2])
    y21 = torch.minimum(boxes2[:, 1], boxes2[:, 3])
    x22 = torch.maximum(boxes2[:, 0], boxes2[:, 2])
    y22 = torch.maximum(boxes2[:, 1], boxes2[:, 3])

    # 2) 교집합
    ix1 = torch.maximum(x11, x21)
    iy1 = torch.maximum(y11, y21)
    ix2 = torch.minimum(x12, x22)
    iy2 = torch.minimum(y12, y22)

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # 3) 합집합
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1 + area2 - inter

    # 4) 안전 나눗셈
    return inter / (union + eps)


pred = torch.tensor([
    [10., 10., 30., 30.],  # 꽤 잘 맞는 박스
    [10., 10., 20., 20.],  # 작은 박스
    [30., 30., 10., 10.],  # 뒤집힌 좌표(의도적)
])

gt = torch.tensor([
    [12., 12., 28., 28.],
    [40., 40., 60., 60.],
    [12., 12., 28., 28.],
])

iou = pairwise_iou_xyxy(pred, gt)
print(iou)  # tensor([...])

# 빠른 체크: IoU는 0~1 범위(작은 수치 오차 허용)
assert torch.all(iou >= -1e-6) and torch.all(iou <= 1.0 + 1e-6)
```

초보자용 체크 포인트:
- 1번째 쌍은 IoU가 상대적으로 큽니다.
- 2번째 쌍은 멀리 떨어져 IoU가 0에 가깝습니다.
- 3번째 쌍은 좌표가 뒤집혀 있었지만, 보정 로직 덕분에 NaN 없이 계산됩니다.

## 실무에서 자주 쓰는 임계값
| 용도 | IoU 기준 | 의미 |
|---|---:|---|
| Pascal VOC AP | 0.5 | 50% 이상 겹치면 TP |
| COCO AP | 0.5:0.95 | 여러 임계값 평균으로 더 엄격한 평가 |
| NMS | 0.5~0.7 | 너무 겹치는 예측 박스 제거 |
| Anchor 매칭(예시) | pos: ≥0.5, neg: <0.4 | 학습 샘플 라벨링 기준 |

## 디버깅 체크리스트 (현업용)
모델 성능이 갑자기 떨어지거나 AP가 비정상적으로 낮다면, 아래 순서로 확인하세요.

1. **좌표 포맷 통일**: 예측/정답이 둘 다 `xyxy`인지 확인
2. **좌표 범위 확인**: `x1 <= x2`, `y1 <= y2` 보장 (뒤집힌 박스 제거)
3. **스케일 일치**: 한쪽은 픽셀, 한쪽은 0~1 정규화 좌표가 아닌지 확인
4. **NMS 임계값 재점검**: 너무 낮으면 과억제, 너무 높으면 중복 검출 증가
5. **평가 임계값 확인**: VOC(0.5)와 COCO(0.5:0.95) 혼동 여부 점검

## 자주 하는 실수 (FAQ)
- **Q. IoU가 음수가 나올 수 있나요?**  
  A. 일반 IoU는 음수가 될 수 없습니다. 음수가 나오면 구현 버그일 가능성이 큽니다.

- **Q. 예측 박스가 정답을 완전히 포함하면 IoU=1인가요?**  
  A. 아닙니다. 두 박스가 완전히 동일할 때만 1입니다. 포함만으로는 1보다 작습니다.

- **Q. AP가 낮은데 분류 점수는 괜찮습니다. 왜 그럴까요?**  
  A. 분류는 맞았지만 박스 위치가 조금씩 틀려 IoU 임계값을 못 넘는 경우가 흔합니다.

## 30초 암기법: IoU 계산 4단계
처음 IoU를 구현할 때는 아래 4단계만 기억하면 실수를 크게 줄일 수 있습니다.

1. **겹치는 사각형 좌표 구하기**: `max(left)`, `max(top)`, `min(right)`, `min(bottom)`
2. **겹친 면적 계산**: `max(0, w) * max(0, h)`
3. **합집합 계산**: `area1 + area2 - inter`
4. **나누기 안전 처리**: `union == 0`이면 0 반환

실무에서 IoU 버그의 대부분은 2번(`max(0, ...)` 누락)과 4번(0 나누기)에서 발생합니다.

## 증상→원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 볼 항목 |
|---|---|---|
| AP50은 괜찮은데 AP75가 급락 | 박스 정렬 정밀도 부족 | NMS threshold, box regression 학습률 |
| 학습 초반 IoU가 계속 0 근처 | 좌표 포맷 혼합 | `xyxy`/`cxcywh` 변환 위치 |
| 간헐적 NaN 발생 | union=0 또는 음수 폭/높이 | `max(0, ...)`, `eps`, 입력 검증 |
| 클래스는 맞는데 FP가 많음 | NMS 과완화 | NMS IoU 기준 상향 검토 |

## 관련 콘텐츠
- [Anchor Box](/ko/docs/components/detection/anchor)
- [NMS](/ko/docs/components/detection/nms)
- [YOLO](/ko/docs/architecture/detection/yolo)
