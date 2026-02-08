---
title: "최적화 알고리즘"
weight: 2
bookCollapseSection: true
math: true
---

# 최적화 알고리즘 (Optimizers)

Optimizer는 gradient를 사용해 파라미터를 업데이트하는 방법을 정의합니다.

## 기본 원리

모든 optimizer의 목표:
$$
\theta_{t+1} = \theta_t - \eta \cdot g(\nabla L, \text{history})
$$

- θ: 파라미터
- η: 학습률
- g: gradient 변환 함수

## 핵심 개념

| Optimizer | 특징 | 주 사용처 |
|-----------|------|----------|
| [SGD](/ko/docs/components/training/optimizer/sgd) | 단순, 일반화 좋음 | CNN, 대규모 학습 |
| [Adam](/ko/docs/components/training/optimizer/adam) | 적응적 학습률 | Transformer, 빠른 수렴 |
| [LR Scheduler](/ko/docs/components/training/optimizer/lr-scheduler) | 학습률 스케줄링 | 모든 모델 |

## 관련 콘텐츠

- [Gradient](/ko/docs/math/calculus/gradient) - Optimizer의 입력
- [Backpropagation](/ko/docs/math/calculus/backpropagation) - Gradient 계산
