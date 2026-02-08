---
title: "어텐션"
weight: 5
bookCollapseSection: true
math: true
---

# 어텐션 (Attention)

"어디에 집중할 것인가"를 학습하는 메커니즘으로, Transformer의 핵심입니다.

## 왜 Attention인가?

CNN의 한계:
- **고정된 receptive field**: 커널 크기에 의존
- **전역 관계 학습 어려움**: 깊은 층 필요

Attention의 장점:
- **동적 가중치**: 입력에 따라 달라지는 연결
- **전역 상호작용**: 한 번에 모든 위치 참조
- **해석 가능성**: 어디를 봤는지 시각화 가능

## 핵심 개념

| 개념 | 설명 | 적용 |
|------|------|------|
| [Self-Attention](/ko/docs/components/attention/self-attention) | 자기 자신 참조 | Transformer, ViT |
| [Cross-Attention](/ko/docs/components/attention/cross-attention) | 다른 시퀀스 참조 | VLM, Diffusion |
| [Positional Encoding](/ko/docs/components/attention/positional-encoding) | 위치 정보 주입 | 순서 인식 |

## 관련 콘텐츠

- [행렬](/ko/docs/math/linear-algebra/matrix) - QKV 연산
- [Softmax](/ko/docs/math/probability/distribution) - 어텐션 가중치
- [Layer Normalization](/ko/docs/components/normalization/layer-norm)
