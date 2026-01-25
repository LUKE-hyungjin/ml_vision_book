---
title: "데이터"
weight: 1
bookCollapseSection: true
---

# 데이터 엔지니어링

데이터 수집, 처리, 증강에 관한 실무 기술입니다.

## 주요 내용

| 주제 | 설명 |
|------|------|
| [데이터 증강](/ko/docs/engineering/data/augmentation) | 이미지 변환, Cutout, MixUp, AutoAugment |
| [데이터 포맷](/ko/docs/engineering/data/formats) | COCO, Pascal VOC, YOLO 포맷 |
| [데이터 파이프라인](/ko/docs/engineering/data/pipeline) | DataLoader, 전처리 최적화 |
| [레이블링](/ko/docs/engineering/data/labeling) | 도구, 품질 관리, 자동화 |

---

## 데이터의 중요성

> "Garbage in, garbage out"

아무리 좋은 모델도 데이터 품질이 나쁘면 성능이 제한됩니다.

### 데이터 vs 모델

| 상황 | 우선순위 |
|------|----------|
| 데이터 부족 | 증강, 전이학습 |
| 데이터 품질 낮음 | 정제, 레이블링 개선 |
| 데이터 불균형 | 샘플링, 가중치 조정 |
| 데이터 충분 | 모델 개선 |

