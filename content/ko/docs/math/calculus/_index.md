---
title: "미적분"
weight: 2
bookCollapseSection: true
math: true
---

# 미적분 (Calculus)

딥러닝 학습의 핵심은 미분입니다. 손실 함수의 기울기를 계산하여 파라미터를 업데이트합니다.

## 왜 미적분인가?

딥러닝 학습 과정:
1. 순전파: 입력 → 예측
2. 손실 계산: 예측 vs 정답
3. **역전파**: 손실에 대한 각 파라미터의 기울기 계산
4. 파라미터 업데이트: 기울기 방향으로 이동

## 핵심 개념

| 개념 | 설명 | 딥러닝 적용 |
|------|------|------------|
| [Gradient](/ko/docs/math/calculus/gradient) | 다변수 함수의 미분 | 파라미터 업데이트 방향 |
| [Chain Rule](/ko/docs/math/calculus/chain-rule) | 합성 함수의 미분 | 역전파의 수학적 기반 |
| [Backpropagation](/ko/docs/math/calculus/backpropagation) | 효율적 기울기 계산 | 딥러닝 학습 알고리즘 |

## 관련 콘텐츠

- [SGD](/ko/docs/math/training/optimizer/sgd) - 기울기 기반 최적화
- [Adam](/ko/docs/math/training/optimizer/adam) - 적응적 학습률
- [Loss Functions](/ko/docs/math/training/loss) - 미분 가능한 목적 함수
