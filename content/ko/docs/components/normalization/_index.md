---
title: "정규화"
weight: 6
bookCollapseSection: true
math: true
---

# 정규화 (Normalization)

{{% hint info %}}
**선수지식**: [기댓값](/ko/docs/math/probability/expectation)
{{% /hint %}}

> **한 줄 요약**: 정규화는 활성화 값의 분포를 안정화하여, 깊은 네트워크의 **학습을 빠르고 안정적**으로 만드는 핵심 기법입니다.

## 왜 정규화인가?

레이어를 깊게 쌓으면 활성화 분포가 점점 변합니다 (Internal Covariate Shift). 정규화는 이를 막아줍니다:

- **활성화 분포 안정화** → 학습 안정성
- **더 큰 학습률 사용** → 빠른 수렴
- **초기화에 둔감** → 튜닝 부담 감소

## 핵심 질문: "어떤 축으로 정규화하나?"

모든 정규화 기법의 차이는 **(B, C, H, W) 텐서에서 어떤 축으로 평균/분산을 구하느냐**입니다:

![정규화 방법 비교](/images/components/normalization/ko/normalization-comparison.jpeg)

## 학습 순서

| 순서 | 개념 | 핵심 질문 | 주 사용처 |
|------|------|----------|----------|
| 1 | [Batch Norm](/ko/docs/components/normalization/batch-norm) | 배치에서 채널별 통계를 쓰면? | CNN (ResNet, VGG) |
| 2 | [Layer Norm](/ko/docs/components/normalization/layer-norm) | 샘플 내 feature 통계를 쓰면? | Transformer (ViT, GPT) |
| 3 | [RMSNorm](/ko/docs/components/normalization/rms-norm) | 평균 빼기를 생략하면? | 최신 LLM (LLaMA, Qwen) |
| 4 | [Group Norm](/ko/docs/components/normalization/group-norm) | 채널을 그룹으로 나누면? | Detection, Diffusion |
| 5 | [Instance Norm](/ko/docs/components/normalization/instance-norm) | 채널 각각을 독립적으로? | Style Transfer, GAN |

## 선택 가이드

| 상황 | 추천 | 이유 |
|------|------|------|
| CNN + 큰 배치 (≥16) | BatchNorm | 가장 검증된 방법 |
| CNN + 작은 배치 (1~4) | GroupNorm | 배치 무관 |
| Transformer / ViT | LayerNorm | 시퀀스 모델 표준 |
| 최신 LLM | RMSNorm | 속도 + 동등 성능 |
| Style Transfer / GAN | InstanceNorm | 스타일 정보 제거 |
| Diffusion (U-Net) | GroupNorm | 배치 무관 + CNN |

## 관련 콘텐츠

- [기댓값](/ko/docs/math/probability/expectation) — 평균, 분산의 수학적 정의
- [Dropout](/ko/docs/components/training/regularization/dropout) — 또 다른 정규화 기법
- [Weight Decay](/ko/docs/components/training/regularization/weight-decay) — 가중치 정규화
