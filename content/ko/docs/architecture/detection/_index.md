---
title: "Detection"
weight: 3
bookCollapseSection: true
---

# Object Detection 모델

이미지에서 객체의 위치(bounding box)와 클래스를 동시에 예측하는 모델들입니다.

## 두 가지 접근 방식

### Two-Stage Detector

1. Region Proposal: 객체가 있을 법한 영역 제안
2. Classification: 각 영역을 분류

**특징**: 정확도 높음, 속도 느림

- R-CNN → Fast R-CNN → [Faster R-CNN](/ko/docs/architecture/detection/faster-rcnn)

### One-Stage Detector

한 번에 위치와 클래스를 예측

**특징**: 속도 빠름, 실시간 가능

- [YOLO](/ko/docs/architecture/detection/yolo)
- SSD
- RetinaNet

---

## 모델 비교

| 모델 | 유형 | FPS | mAP (COCO) | 특징 |
|------|------|-----|------------|------|
| Faster R-CNN | Two-stage | ~7 | 42.0 | 높은 정확도 |
| YOLOv3 | One-stage | ~30 | 33.0 | 실시간 |
| YOLOv8 | One-stage | ~100 | 53.9 | SOTA |
| DETR | Transformer | ~28 | 42.0 | End-to-end |

---

## 관련 콘텐츠

- [IoU](/ko/docs/components/detection/iou) & [NMS](/ko/docs/components/detection/nms) - Detection의 핵심 개념
- [Anchor Box](/ko/docs/components/detection/anchor) - 박스 예측 방식
- [Detection 태스크](/ko/docs/task/detection) - 평가 지표, 데이터셋
