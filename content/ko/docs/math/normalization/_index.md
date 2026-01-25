---
title: "정규화"
weight: 6
bookCollapseSection: true
math: true
---

# 정규화 (Normalization)

정규화는 활성화 값의 분포를 안정화하여 학습을 돕습니다.

## 왜 정규화인가?

**Internal Covariate Shift 문제**:
- 층을 통과할수록 활성화 분포가 변함
- 이전 층의 파라미터가 바뀌면 다음 층 입력 분포 변화
- 학습 불안정, 느린 수렴

**정규화의 효과**:
- 활성화 분포 안정화
- 더 큰 학습률 사용 가능
- 초기화에 덜 민감
- 일부 정규화 효과

## 핵심 개념

| 방법 | 정규화 축 | 주 사용처 |
|------|----------|----------|
| [Batch Norm](/ko/docs/math/normalization/batch-norm) | Batch | CNN |
| [Layer Norm](/ko/docs/math/normalization/layer-norm) | Feature | Transformer |
| [RMSNorm](/ko/docs/math/normalization/rms-norm) | Feature | LLM |

## 관련 콘텐츠

- [Dropout](/ko/docs/math/training/regularization/dropout) - 또 다른 정규화 기법
- [Weight Decay](/ko/docs/math/training/regularization/weight-decay)
