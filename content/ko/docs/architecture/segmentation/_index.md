---
title: "Segmentation"
weight: 4
bookCollapseSection: true
---

# Segmentation 모델

이미지의 각 픽셀에 레이블을 할당하는 모델들입니다.

## Segmentation 유형

### Semantic Segmentation

각 픽셀을 클래스로 분류 (객체 구분 X)

```
Input: 고양이 2마리 이미지
Output: 모든 고양이 픽셀 → "고양이" 클래스
```

### Instance Segmentation

각 객체 인스턴스를 구분

```
Input: 고양이 2마리 이미지
Output: 고양이1 픽셀, 고양이2 픽셀 (분리)
```

### Panoptic Segmentation

Semantic + Instance 통합

```
Output: 배경(semantic) + 각 객체(instance)
```

---

## 주요 모델

| 모델 | 유형 | 연도 | 특징 |
|------|------|------|------|
| [U-Net](/ko/docs/architecture/segmentation/unet) | Semantic | 2015 | Encoder-Decoder, 의료영상 |
| [Mask R-CNN](/ko/docs/architecture/segmentation/mask-rcnn) | Instance | 2017 | Faster R-CNN + Mask |
| DeepLab | Semantic | 2017 | Atrous Convolution |
| [SAM](/ko/docs/architecture/segmentation/sam) | Promptable | 2023 | Foundation Model |

---

## 관련 콘텐츠

- [Transposed Convolution](/ko/docs/math/transposed-conv) - Upsampling 기법
- [Segmentation 태스크](/ko/docs/task/segmentation) - 평가 지표, 데이터셋
