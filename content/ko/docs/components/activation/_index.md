---
title: "활성화 함수"
weight: 7
bookCollapseSection: true
math: true
---

# 활성화 함수 (Activation Functions)

{{% hint info %}}
**선수지식**: [미분 기초](/ko/docs/math/calculus/basics)
{{% /hint %}}

> **한 줄 요약**: 활성화 함수는 뉴럴 네트워크에 **비선형성**을 부여하여, 단순 선형 변환 이상의 복잡한 패턴을 학습할 수 있게 만드는 핵심 함수입니다.

## 왜 활성화 함수인가?

활성화 함수가 없으면 아무리 레이어를 깊게 쌓아도 결과는 **하나의 선형 변환**과 같습니다:

```
Linear → Linear → Linear = 하나의 Linear
W₃(W₂(W₁x)) = (W₃W₂W₁)x = Wx  ← 행렬 곱은 합쳐짐!
```

활성화 함수가 비선형성을 끊어주기 때문에 깊은 네트워크가 의미를 가집니다:

```
Linear → ReLU → Linear → ReLU → Linear ≠ 하나의 Linear
```

## 핵심 질문: "어떤 활성화 함수를 쓸 것인가?"

활성화 함수마다 **출력 범위, 미분 특성, 계산 비용**이 다릅니다. 모델의 위치와 목적에 따라 적절한 함수를 선택해야 합니다.

![활성화 함수 그래프 비교](/images/components/activation/ko/activation-comparison.png)

## 학습 순서

| 순서 | 개념 | 핵심 질문 | 주 사용처 |
|------|------|----------|----------|
| 1 | [ReLU](/ko/docs/components/activation/relu) | 음수를 0으로 만들면? | CNN 은닉층 |
| 2 | [Sigmoid](/ko/docs/components/activation/sigmoid) | 출력을 0~1로 압축하면? | 이진 분류, 게이트 |
| 3 | [Softmax](/ko/docs/components/activation/softmax) | 여러 값을 확률 분포로 만들면? | 다중 분류 출력 |
| 4 | [GELU](/ko/docs/components/activation/gelu) | 확률적으로 뉴런을 살리면? | Transformer (ViT, GPT, BERT) |
| 5 | [Swish/SiLU](/ko/docs/components/activation/swish-silu) | x에 sigmoid를 곱하면? | EfficientNet, LLaMA |

## 선택 가이드

| 상황 | 추천 | 이유 |
|------|------|------|
| CNN 은닉층 | ReLU | 빠르고 검증됨 |
| Transformer 은닉층 | GELU | ViT, GPT, BERT 표준 |
| 최신 LLM (FFN) | Swish/SiLU | LLaMA, Qwen 등 채택 |
| 이진 분류 출력 | Sigmoid | 0~1 확률 |
| 다중 분류 출력 | Softmax | 확률 분포 |
| 게이트 메커니즘 | Sigmoid | LSTM, GRU, Attention |

## 관련 콘텐츠

- [미분 기초](/ko/docs/math/calculus/basics) — 활성화 함수의 미분
- [확률분포](/ko/docs/math/probability/distribution) — GELU의 수학적 배경
- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — 활성화 전/후 정규화
