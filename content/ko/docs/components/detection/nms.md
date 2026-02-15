---
title: "NMS"
weight: 2
math: true
---

# NMS (Non-Maximum Suppression)

{{% hint info %}}
**선수지식**: [IoU](/ko/docs/components/detection/iou)
{{% /hint %}}

## 한 줄 요약
> **NMS는 같은 물체를 가리키는 중복 박스 중에서 가장 신뢰도 높은 박스만 남기는 후처리 알고리즘입니다.**

## 왜 필요한가?
객체 검출 모델은 한 물체 주변에 비슷한 박스를 여러 개 예측하는 경향이 있습니다.
이 상태로 결과를 쓰면 같은 물체를 여러 번 검출한 것처럼 보이므로, 후처리로 중복을 제거해야 합니다.

NMS는 다음 모델들에서 거의 필수로 쓰입니다.
- 1-stage detector: [YOLO](/ko/docs/architecture/detection/yolo), SSD
- 2-stage detector: Faster R-CNN 계열

비유하면, 여러 사람이 같은 사람을 가리키며 "저기!"라고 말할 때 가장 확신 있는 한 사람의 지목만 채택하는 과정입니다.

## 수식/기호
NMS는 단일 수식보다는 절차 중심이지만, 판단 기준은 IoU 임계값으로 표현할 수 있습니다.

$$
\text{suppress box } j \quad \text{if} \quad \mathrm{IoU}(b_i, b_j) > t
$$

**각 기호의 의미:**
- $b_i$ : 현재 선택된(점수가 가장 높은) 박스
- $b_j$ : 비교 대상 박스
- $\mathrm{IoU}(b_i, b_j)$ : 두 박스의 겹침 비율
- $t$ : NMS 임계값 (보통 0.45~0.7)

## 직관
1. 점수가 가장 높은 박스를 하나 고릅니다.
2. 그 박스와 너무 많이 겹치는 박스(IoU > $t$)를 지웁니다.
3. 남은 박스에서 다시 최고 점수를 고르고 반복합니다.

핵심은 "**높은 점수 + 낮은 중복**" 조합만 남기는 것입니다.

## 구현
```python
import numpy as np


def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4) each box is [x1, y1, x2, y2]
    scores: (N,) confidence score
    """
    order = scores.argsort()[::-1]  # high -> low
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        ious = compute_iou(boxes[i], boxes[rest])

        # IoU가 임계값 이하인 박스만 유지
        order = rest[ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)
```

### 자주 쓰는 변형
- **Batched NMS**: 클래스가 다르면 서로 억제하지 않음
- **Soft-NMS**: 박스를 삭제하지 않고 점수만 감쇠
- **DIoU-NMS**: 중심점 거리까지 고려해 억제 강도 조절


## 실무에서 자주 놓치는 포인트
### 1) 클래스별 NMS인지 확인
대부분의 검출기는 **클래스별(class-wise) NMS**를 사용합니다.
즉, 클래스가 다르면 박스가 겹쳐도 서로 억제하지 않습니다.

예를 들어 사람(person) 박스와 배낭(backpack) 박스는 크게 겹칠 수 있으므로,
클래스 정보를 무시한 전역 NMS를 쓰면 정답을 잘못 지울 수 있습니다.

### 2) Top-K 사전 필터링
후보 박스가 너무 많으면 NMS 비용이 커집니다. (대략 $O(N^2)$ 비교)
실무에서는 보통 NMS 전에 score 상위 $K$개만 남겨 속도/메모리를 절약합니다.

- 예: 클래스별로 상위 1000개만 유지 후 NMS 수행
- 효과: 지연 시간 감소, 모바일/엣지 배포에서 특히 유리

### 3) 임계값 튜닝 순서
임계값은 보통 아래 순서로 맞추면 시행착오가 줄어듭니다.

1. score threshold를 먼저 정해 너무 약한 박스를 제거
2. 그다음 NMS IoU threshold를 조정해 중복 제거 강도 조절
3. 마지막으로 AP/재현율 변화를 보고 미세 조정

## 실무 팁
- 객체가 매우 밀집된 데이터셋이면 임계값을 낮춰(예: 0.45) 중복을 더 강하게 제거
- 작은 객체가 많으면 임계값을 너무 낮추지 않기(정답까지 지워질 수 있음)
- 보통 score threshold와 함께 튜닝해야 안정적

## 관련 콘텐츠
- [IoU](/ko/docs/components/detection/iou)
- [Anchor Box](/ko/docs/components/detection/anchor)
- [YOLO](/ko/docs/architecture/detection/yolo)
