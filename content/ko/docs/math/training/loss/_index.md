---
title: "손실 함수"
weight: 1
bookCollapseSection: true
math: true
---

# 손실 함수 (Loss Functions)

손실 함수는 모델의 예측과 정답 간의 차이를 측정합니다. 학습의 목표는 이 값을 최소화하는 것입니다.

## 왜 손실 함수가 중요한가?

- **학습 방향 결정**: Gradient가 손실 함수에서 계산됨
- **문제 정의**: 어떤 손실을 쓰느냐 = 무엇을 최적화하느냐
- **수렴 속도**: 적절한 손실 선택이 학습 효율 결정

## 핵심 개념

| 손실 함수 | 용도 | 특징 |
|----------|------|------|
| [Cross-Entropy](/ko/docs/math/training/loss/cross-entropy) | 분류 | 확률분포 비교 |
| [Focal Loss](/ko/docs/math/training/loss/focal-loss) | 불균형 분류 | 어려운 샘플 집중 |
| [Contrastive Loss](/ko/docs/math/training/loss/contrastive-loss) | 표현 학습 | 유사도 기반 |

## 관련 콘텐츠

- [Backpropagation](/ko/docs/math/calculus/backpropagation) - 손실에서 gradient 계산
- [SGD](/ko/docs/math/training/optimizer/sgd) - 손실 최소화 알고리즘
