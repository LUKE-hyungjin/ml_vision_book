---
title: "확률/통계"
weight: 3
bookCollapseSection: true
math: true
---

# 확률과 통계 (Probability & Statistics)

딥러닝은 본질적으로 확률 모델입니다. 분류, 생성, 불확실성 추정 모두 확률에 기반합니다.

## 왜 확률인가?

- **분류**: P(class | image) 확률 출력
- **생성**: P(image) 분포에서 샘플링
- **Loss**: 확률분포 간 거리 (Cross-Entropy, KL Divergence)
- **Dropout**: 베르누이 분포로 랜덤 마스킹
- **Diffusion**: 노이즈 추가/제거의 확률 과정

## 핵심 개념

| 개념 | 설명 | 딥러닝 적용 |
|------|------|------------|
| [베이즈 정리](/ko/docs/math/probability/bayes) | 조건부 확률의 역산 | 불확실성 추정 |
| [확률분포](/ko/docs/math/probability/distribution) | 확률의 함수적 표현 | Softmax, VAE |
| [샘플링](/ko/docs/math/probability/sampling) | 분포에서 값 추출 | 생성 모델, Dropout |

## 관련 콘텐츠

- [Cross-Entropy Loss](/ko/docs/math/training/loss/cross-entropy) - 확률분포 기반 손실
- [Diffusion](/ko/docs/math/diffusion) - 확률 과정 기반 생성
- [Label Smoothing](/ko/docs/math/training/regularization/label-smoothing)
