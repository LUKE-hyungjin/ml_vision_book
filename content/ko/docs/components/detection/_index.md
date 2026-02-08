---
title: "Detection 수학"
weight: 8
bookCollapseSection: true
math: true
---

# Detection 수학

객체 검출에서 사용되는 핵심 수학 개념들입니다.

## 핵심 개념

| 개념 | 설명 | 용도 |
|------|------|------|
| [IoU](/ko/docs/components/detection/iou) | 박스 간 겹침 측정 | 평가, 학습 |
| [NMS](/ko/docs/components/detection/nms) | 중복 박스 제거 | 후처리 |
| [Anchor Box](/ko/docs/components/detection/anchor) | 사전 정의된 박스 | 예측 기준점 |

## Detection 파이프라인

```
이미지 → Backbone → Feature Map → Head → 예측
                                    ↓
                              NMS → 최종 결과
```

## 관련 콘텐츠

- [Focal Loss](/ko/docs/components/training/loss/focal-loss)
- [Cross-Entropy](/ko/docs/components/training/loss/cross-entropy)
