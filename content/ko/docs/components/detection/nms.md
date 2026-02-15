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

## 작은 예제로 한 번에 이해하기
아래처럼 같은 물체를 가리키는 후보가 3개 있다고 가정해 봅시다. (IoU 기준 $t=0.5$)

| 박스 | 점수 | A와의 IoU | 결과 |
|---|---:|---:|---|
| A | 0.92 | - | 유지(최고 점수) |
| B | 0.88 | 0.76 | 제거 (A와 과도하게 겹침) |
| C | 0.61 | 0.31 | 유지 (겹침이 낮음) |

이 과정이 끝나면 A, C만 남습니다.  
핵심은 **"점수는 높지만 같은 물체면 하나만 남긴다"** 입니다.

## 수식 → 코드 매핑 (초보자 브릿지)
NMS를 처음 구현할 때는 아래 4줄을 수식과 1:1로 연결해서 보면 헷갈림이 줄어듭니다.

1. **최고 점수 박스 선택**
   - 수식 관점: 현재 단계의 기준 박스 $b_i$ 선택
   - 코드 관점: `i = order[0]`

2. **나머지와 IoU 계산**
   - 수식 관점: $\mathrm{IoU}(b_i, b_j)$ 계산
   - 코드 관점: `ious = compute_iou(boxes[i], boxes[rest])`

3. **임계값으로 억제 여부 결정**
   - 수식 관점: $\mathrm{IoU}(b_i, b_j) > t$ 이면 억제
   - 코드 관점: `order = rest[ious <= iou_threshold]`

4. **반복 종료 조건 확인**
   - 수식 관점: 비교할 박스가 없으면 종료
   - 코드 관점: `if len(order) == 1: break`

이 매핑을 익혀 두면 Soft-NMS나 class-wise NMS로 확장할 때도 구조를 그대로 재사용할 수 있습니다.

## 클래스별 NMS 최소 구현 스케치
실무에서 가장 흔한 버그는 "전역 NMS"를 돌려 다른 클래스를 서로 지워 버리는 문제입니다.
아래처럼 클래스 단위로 분리해서 NMS를 적용하면 안전합니다.

```python
def class_wise_nms(boxes, scores, labels, iou_threshold=0.5):
    keep_all = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep_c = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.append(idx[keep_c])  # 원래 인덱스로 복원
    return np.concatenate(keep_all)
```

## 디버깅 체크리스트 (현업용)
- [ ] **클래스별 NMS 적용 확인**: 다른 클래스끼리 억제되고 있지 않은가?
- [ ] **좌표 스케일 확인**: 박스가 픽셀 좌표인지, 0~1 정규화 좌표인지 섞이지 않았는가?
- [ ] **임계값 순서 점검**: score threshold를 먼저 정하고, 그다음 NMS IoU를 조절했는가?
- [ ] **Top-K 설정 확인**: 후보가 너무 많아 NMS 비용이 폭증하지 않는가?
- [ ] **군중 장면 오류 확인**: 사람 밀집 장면에서 TP가 과도하게 사라지지 않는가?

## 자주 하는 실수 (FAQ)
**Q1. NMS 임계값을 낮추면 항상 좋은가요?**  
A. 아닙니다. 중복은 줄지만 정답까지 지워 재현율(recall)이 떨어질 수 있습니다.

**Q2. score threshold와 NMS threshold 중 무엇을 먼저 조절하나요?**  
A. 보통 score threshold를 먼저 잡고, 그다음 NMS threshold를 맞춥니다. 순서를 바꾸면 원인 파악이 어려워집니다.

**Q3. 클래스가 다르면 IoU가 커도 지워야 하나요?**  
A. 일반적으로는 지우지 않습니다(class-wise NMS). 예: person과 backpack은 크게 겹쳐도 동시에 정답일 수 있습니다.

## NMS vs Soft-NMS 빠른 선택 가이드
초보자가 실무에서 가장 자주 막히는 지점은 "언제 Hard NMS를 쓰고, 언제 Soft-NMS를 쓰나?"입니다.

| 상황 | 추천 | 이유 |
|---|---|---|
| 실시간/엣지 환경, 지연 시간 최우선 | Hard NMS | 구현 단순, 속도 유리 |
| 군중 장면(사람 밀집), 박스가 많이 겹침 | Soft-NMS | 정답 박스를 과하게 지우는 현상 완화 |
| 베이스라인 재현/디버깅 시작 단계 | Hard NMS부터 | 문제 원인 분리가 쉬움 |

실무 팁: 먼저 Hard NMS로 기준 성능을 만든 뒤, recall 손실이 큰 데이터셋에서만 Soft-NMS를 추가 검토하면 시행착오가 줄어듭니다.

## 시각자료(나노바나나) 프롬프트
- **KO 다이어그램 1 (NMS 절차)**  
  "다크 테마 배경(#1a1a2e), Object Detection NMS 절차 인포그래픽. 입력: 동일 물체를 감싸는 5개 박스(서로 다른 confidence: 0.95, 0.88, 0.74, 0.60, 0.41). 단계 1: 최고 점수 박스 선택. 단계 2: IoU>0.5 박스 붉은색으로 suppress 표시. 단계 3: 남은 박스에서 반복. 최종 출력: 2개 박스만 유지. 화살표와 단계 번호(1,2,3), 한국어 라벨(선택/억제/유지), 깔끔한 벡터 스타일, 높은 가독성"

- **KO 다이어그램 2 (Threshold trade-off)**  
  "다크 테마 배경(#1a1a2e), NMS IoU threshold 변화(0.3/0.5/0.7)에 따른 결과 비교 3패널. 각 패널에 동일 예측 박스 집합, 유지된 박스 수와 TP/FP 경향 주석. 한국어 라벨: '과억제', '균형', '중복 잔존'. 테이블형 범례 포함, 미니멀 벡터 스타일"

## 관련 콘텐츠
- [IoU](/ko/docs/components/detection/iou)
- [Anchor Box](/ko/docs/components/detection/anchor)
- [YOLO](/ko/docs/architecture/detection/yolo)
