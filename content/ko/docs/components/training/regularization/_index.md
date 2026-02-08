---
title: "정규화 기법"
weight: 3
bookCollapseSection: true
math: true
---

# 정규화 기법 (Regularization)

정규화는 모델의 과적합을 방지하여 일반화 성능을 높입니다.

## 왜 정규화인가?

- **과적합**: 학습 데이터에만 최적화 → 새 데이터 성능 저하
- **해결**: 모델 복잡도 제한, 불확실성 주입

## 핵심 개념

| 기법 | 방식 | 효과 |
|------|------|------|
| [Dropout](/ko/docs/components/training/regularization/dropout) | 뉴런 무작위 제거 | 앙상블 효과 |
| [Weight Decay](/ko/docs/components/training/regularization/weight-decay) | 가중치 크기 제한 | 간단한 모델 유도 |
| [Label Smoothing](/ko/docs/components/training/regularization/label-smoothing) | 정답 확률 부드럽게 | 과신뢰 방지 |

## 관련 콘텐츠

- [Cross-Entropy](/ko/docs/components/training/loss/cross-entropy) - Label Smoothing 적용
- [Adam](/ko/docs/components/training/optimizer/adam) - Weight Decay
