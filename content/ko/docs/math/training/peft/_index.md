---
title: "PEFT"
weight: 4
bookCollapseSection: true
math: true
---

# PEFT (Parameter-Efficient Fine-Tuning)

PEFT는 대규모 모델을 효율적으로 미세조정하는 기법입니다.

## 왜 PEFT인가?

- **문제**: LLM/대형 모델 전체 파라미터 학습 = 메모리/시간 폭발
- **해결**: 소수의 파라미터만 학습, 나머지는 고정

## 핵심 기법

| 기법 | 방식 | 특징 |
|------|------|------|
| [LoRA](/ko/docs/math/training/peft/lora) | 저랭크 행렬 분해 | 가장 널리 사용 |
| [QLoRA](/ko/docs/math/training/peft/qlora) | LoRA + 양자화 | 메모리 극적 절감 |
| [Adapter](/ko/docs/math/training/peft/adapter) | 작은 모듈 삽입 | 원조 PEFT |
| [Prefix Tuning](/ko/docs/math/training/peft/prefix-tuning) | 가상 토큰 학습 | 프롬프트 기반 |

## 효율성 비교

| 방식 | 학습 파라미터 | GPU 메모리 |
|------|---------------|------------|
| Full Fine-tuning | 100% | 100% |
| LoRA | 0.1-1% | ~50% |
| QLoRA | 0.1-1% | ~25% |
| Adapter | 1-5% | ~60% |

## 관련 콘텐츠

- [Adam](/ko/docs/math/training/optimizer/adam) - 최적화
- [Weight Decay](/ko/docs/math/training/regularization/weight-decay)
- [SVD](/ko/docs/math/linear-algebra/svd) - 저랭크 분해
