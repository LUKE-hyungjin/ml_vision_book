---
title: "레이블링"
weight: 4
---

# 레이블링 (Annotation)

## 개요

데이터 레이블링은 모델 성능의 상한선을 결정하는 중요한 작업입니다.

---

## 레이블링 도구

### 오픈소스

| 도구 | 태스크 | 특징 |
|------|--------|------|
| **LabelImg** | Detection | 간단, VOC/YOLO 포맷 |
| **CVAT** | Detection, Segmentation | 웹 기반, 협업 |
| **Label Studio** | 멀티태스크 | 유연한 설정 |
| **Labelme** | Segmentation | Polygon 지원 |
| **VoTT** | Detection | MS 개발, 간편 |

### 상용 서비스

| 서비스 | 특징 |
|--------|------|
| **Roboflow** | 자동 증강, 포맷 변환 |
| **Scale AI** | 대규모 레이블링 |
| **Supervisely** | 협업, 자동화 |
| **V7 (Darwin)** | AI 지원 레이블링 |

---

## CVAT 사용법

### 설치

```bash
# Docker로 설치
git clone https://github.com/opencv/cvat
cd cvat
docker compose up -d
```

### 프로젝트 설정

1. 프로젝트 생성
2. 레이블 정의 (클래스명, 속성)
3. Task 생성 및 이미지 업로드
4. 레이블링 작업
5. 데이터 내보내기 (COCO, YOLO 등)

---

## 품질 관리

### 일관성 유지

```
가이드라인 문서 필수:
- 클래스 정의 (예시 이미지 포함)
- 경계선 기준 (타이트하게? 여유있게?)
- 모호한 케이스 처리 방법
- 품질 기준 (IoU > 0.8 등)
```

### 교차 검증

```python
# 동일 이미지를 여러 작업자가 레이블링
# IoU로 일관성 측정

def calculate_agreement(boxes1, boxes2, threshold=0.5):
    """두 작업자 간 일치도 계산"""
    matched = 0
    for b1 in boxes1:
        for b2 in boxes2:
            if iou(b1, b2) > threshold:
                matched += 1
                break
    return matched / max(len(boxes1), len(boxes2))
```

### 리뷰 프로세스

```
1차 레이블링 → 리뷰 → 수정 → 최종 승인
     ↓           ↓
    80%         20%
   작업자      리뷰어
```

---

## 자동 레이블링

### 사전학습 모델 활용

```python
from ultralytics import YOLO

# 사전학습된 모델로 초기 레이블 생성
model = YOLO('yolov8x.pt')
results = model.predict('unlabeled_images/')

# 결과를 COCO 포맷으로 저장
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    # ... annotation 파일 생성
```

### SAM으로 Segmentation

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Detection 결과 → SAM으로 마스크 생성
predictor.set_image(image)
masks, _, _ = predictor.predict(box=detection_box)
```

### Human-in-the-loop

```
자동 레이블링 → 사람 검토/수정 → 모델 재학습 → 반복
     ↓              ↓              ↓
   빠름          정확도 보장      품질 향상
```

---

## Active Learning

효율적으로 레이블링할 샘플 선택:

```python
import numpy as np

def uncertainty_sampling(model, unlabeled_data, n_samples):
    """불확실성이 높은 샘플 선택"""
    predictions = model.predict(unlabeled_data)

    # 엔트로피 계산
    entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)

    # 불확실성 높은 순으로 선택
    indices = np.argsort(entropy)[-n_samples:]
    return indices
```

### 전략

| 전략 | 설명 |
|------|------|
| Uncertainty | 모델이 불확실한 샘플 |
| Diversity | 다양한 샘플 |
| Query-by-Committee | 여러 모델이 다르게 예측하는 샘플 |

---

## 비용 효율화

### 계층적 레이블링

```
1단계: 쉬운 샘플 (비전문가, 저비용)
        ↓
2단계: 어려운 샘플 (전문가, 고비용)
```

### 약한 레이블링 (Weak Supervision)

```python
# 이미지 레벨 레이블 → 객체 레벨 레이블
# CAM (Class Activation Map) 활용

from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[model.layer4])
grayscale_cam = cam(input_tensor=image)

# CAM에서 객체 위치 추정
threshold = 0.5
mask = grayscale_cam > threshold
bbox = mask_to_bbox(mask)
```

---

## 레이블링 체크리스트

### 시작 전
- [ ] 명확한 레이블링 가이드라인 작성
- [ ] 예시 이미지 준비
- [ ] 도구 선택 및 설정
- [ ] 작업자 교육

### 진행 중
- [ ] 정기적인 샘플 리뷰
- [ ] 작업자 간 일관성 체크
- [ ] 어려운 케이스 논의 및 가이드라인 업데이트

### 완료 후
- [ ] 전체 데이터 품질 검증
- [ ] 포맷 변환 및 검증
- [ ] 백업

---

## 관련 콘텐츠

- [데이터 포맷](/ko/docs/engineering/data/formats) - 어노테이션 포맷
- [데이터 증강](/ko/docs/engineering/data/augmentation) - 데이터 확장
- [SAM](/ko/docs/architecture/segmentation/sam) - 자동 세그멘테이션

