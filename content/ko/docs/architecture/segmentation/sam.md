---
title: "SAM"
weight: 3
math: true
---

# SAM (Segment Anything Model)

## 개요

- **논문**: Segment Anything (2023)
- **저자**: Alexander Kirillov et al. (Meta AI)
- **핵심 기여**: 프롬프트 기반의 범용 segmentation foundation model

## 핵심 아이디어

> "한 번 학습으로 모든 객체를 segmentation"

Point, box, text 등 다양한 프롬프트로 원하는 객체를 분할합니다.

---

## 프로젝트 구성

### 세 가지 요소

1. **Task**: Promptable segmentation
2. **Model**: SAM 아키텍처
3. **Data**: SA-1B 데이터셋 (11M 이미지, 1.1B 마스크)

---

## 구조

### 전체 아키텍처

```
┌─────────────────────────────────────────┐
│            Image Encoder                 │
│         (ViT-H, 실행 1회)                │
│    Image → Image Embeddings             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           Prompt Encoder                 │
│   Points/Boxes/Text → Prompt Tokens     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           Mask Decoder                   │
│   Image Emb + Prompt → Masks            │
└─────────────────────────────────────────┘
                    ↓
            Output Masks
```

### 1. Image Encoder

- **모델**: ViT-H (Vision Transformer Huge)
- **입력**: 1024×1024 이미지
- **출력**: 64×64 feature embedding
- **특징**: 이미지당 한 번만 실행 (0.15초)

### 2. Prompt Encoder

다양한 프롬프트 처리:

| 프롬프트 | 인코딩 방식 |
|----------|------------|
| Point | Positional encoding + learned embedding |
| Box | Corner points의 positional encoding |
| Mask | Convolution으로 downsampling |
| Text | CLIP text encoder (연구 중) |

### 3. Mask Decoder

- Transformer 기반 경량 디코더
- 양방향 attention (prompt ↔ image)
- 다중 마스크 출력 (모호성 해결)

---

## 프롬프트 유형

### Point Prompt

```python
# 클릭 한 번으로 객체 선택
point_coords = [[500, 375]]  # x, y
point_labels = [1]  # 1: foreground, 0: background
```

### Box Prompt

```python
# Bounding box로 객체 범위 지정
box = [100, 100, 400, 400]  # x1, y1, x2, y2
```

### Mask Prompt

```python
# 이전 마스크를 프롬프트로 사용 (iterative refinement)
mask_input = previous_mask
```

---

## 구현 예시

```python
from segment_anything import sam_model_registry, SamPredictor

# 모델 로드
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# 이미지 설정 (한 번만 실행)
predictor.set_image(image)

# Point prompt로 예측
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True,  # 3개 마스크 출력
)

# Box prompt로 예측
masks, scores, logits = predictor.predict(
    box=np.array([100, 100, 400, 400]),
    multimask_output=False,
)
```

### Automatic Mask Generation

```python
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(sam)

# 이미지의 모든 객체 자동 분할
masks = mask_generator.generate(image)

# 각 마스크 정보
for mask in masks:
    print(mask['segmentation'].shape)  # binary mask
    print(mask['area'])  # 마스크 면적
    print(mask['bbox'])  # bounding box
    print(mask['predicted_iou'])  # 예측 IoU
    print(mask['stability_score'])  # 안정성 점수
```

---

## 모델 변형

| 모델 | Encoder | 파라미터 | 속도 |
|------|---------|----------|------|
| SAM ViT-H | ViT-Huge | 636M | 0.15s/image |
| SAM ViT-L | ViT-Large | 308M | 0.1s/image |
| SAM ViT-B | ViT-Base | 91M | 0.06s/image |

### 경량화 버전

- **MobileSAM**: 모바일용 경량 모델
- **FastSAM**: YOLO 기반 실시간 버전
- **EfficientSAM**: 효율적인 학습 방법

---

## SA-1B 데이터셋

### 데이터 엔진

```
1. Manual Phase: 전문가가 마스크 생성
2. Semi-Automatic: SAM 제안 + 사람 수정
3. Fully Automatic: SAM 자동 생성 + 검증
```

### 규모

- 11M 이미지
- 1.1B 마스크
- 평균 이미지당 100개 마스크

---

## 활용

### Zero-shot Transfer

학습하지 않은 도메인에서도 동작:
- 의료 영상
- 위성 영상
- 현미경 이미지

### 응용 분야

- **이미지 편집**: 객체 선택 및 제거
- **Annotation 도구**: 라벨링 자동화
- **AR/VR**: 실시간 객체 분할
- **비디오 분석**: 프레임별 segmentation

---

## SAM 2 (2024)

SAM의 비디오 확장:

- 비디오 전체에서 객체 추적
- 시간적 일관성 유지
- 메모리 효율적 처리

---

## 한계점

- 큰 모델 크기 (ViT-H)
- 세밀한 경계에서 부정확
- Text prompt 지원 제한적
- Real-time 어려움

---

## 관련 콘텐츠

- [ViT](/ko/docs/architecture/transformer/vit) - Image encoder 기반
- [Mask R-CNN](/ko/docs/architecture/segmentation/mask-rcnn) - Instance segmentation
- [CLIP](/ko/docs/architecture/multimodal/clip) - Text-image 연결
- [Segmentation 태스크](/ko/docs/task/segmentation) - 평가 지표
