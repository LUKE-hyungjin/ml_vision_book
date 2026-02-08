---
title: "어텐션"
weight: 5
bookCollapseSection: true
math: true
---

# 어텐션 (Attention)

> **핵심 아이디어: "모든 것을 보되, 중요한 것에 더 집중하라"**

## 왜 Attention인가?

책을 읽을 때를 생각해봅시다. 모든 문장을 같은 집중력으로 읽나요? 아닙니다. **중요한 부분에 더 집중**하고, 덜 중요한 부분은 빠르게 넘깁니다. 시험 전에 형광펜을 칠하는 것도 같은 원리입니다.

신경망도 마찬가지입니다. 이미지를 분석할 때, 배경보다는 **물체에 집중**해야 합니다. 문장을 이해할 때, 모든 단어가 똑같이 중요하지 않습니다. 이 "어디에 얼마나 집중할지"를 학습하는 것이 Attention입니다.

### CNN의 한계

CNN은 작은 창(커널)으로 이미지를 훑습니다. 이것은 **고정된 크기의 돋보기**와 같습니다:

| 문제 | 설명 |
|------|------|
| **고정된 시야** | 3x3 커널이면 항상 3x3만 봄 |
| **전역 관계 어려움** | 이미지 좌측과 우측을 한번에 볼 수 없음 → 깊은 층을 쌓아야 함 |
| **고정된 가중치** | 어떤 입력이 들어와도 같은 필터 적용 |

### Attention의 장점

Attention은 **줌인/줌아웃이 자유로운 카메라**와 같습니다:

| 장점 | 설명 |
|------|------|
| **동적 가중치** | 입력에 따라 연결이 달라짐 — "이건 중요하니 더 봐야지" |
| **전역 상호작용** | 한 번에 모든 위치를 참조 가능 |
| **해석 가능성** | 모델이 어디를 봤는지 시각화 가능 (Attention Map) |

## 핵심 개념

| 개념 | 한 줄 설명 | 대표 모델 |
|------|-----------|----------|
| [Self-Attention](/ko/docs/components/attention/self-attention) | 자기 자신 내에서 관계를 파악 | Transformer, ViT |
| [Cross-Attention](/ko/docs/components/attention/cross-attention) | 서로 다른 두 정보를 연결 | Stable Diffusion, VLM |
| [Positional Encoding](/ko/docs/components/attention/positional-encoding) | 순서 정보를 주입 | 모든 Transformer 기반 모델 |

## 읽는 순서

```
Self-Attention → Cross-Attention → Positional Encoding
```

Self-Attention이 기본입니다. 이것을 먼저 이해하면 나머지는 자연스럽게 따라옵니다.

## 관련 콘텐츠

**선수 수학:**
- [행렬](/ko/docs/math/linear-algebra/matrix) — Q, K, V 연산의 기초
- [확률분포 (Softmax)](/ko/docs/math/probability/distribution) — 어텐션 가중치 계산

**함께 사용되는 컴포넌트:**
- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — Transformer 블록 구성

**이 개념을 사용하는 아키텍처:**
- [ViT](/ko/docs/architecture/transformer/vit) — 이미지에 Self-Attention 적용
- [CLIP](/ko/docs/architecture/multimodal/clip) — 이미지-텍스트 연결
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) — Cross-Attention으로 텍스트 조건 반영
