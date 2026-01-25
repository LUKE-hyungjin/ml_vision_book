---
title: "Diffusion 수학"
weight: 9
bookCollapseSection: true
math: true
---

# Diffusion 수학

Diffusion 모델의 핵심 수학 개념들입니다.

## 핵심 아이디어

```
데이터 x₀ → 노이즈 추가 → 순수 노이즈 xT
         ← 노이즈 제거 ←
```

- **Forward Process**: 데이터에 점진적으로 노이즈 추가
- **Reverse Process**: 노이즈에서 데이터 복원 (학습 대상)

## 핵심 개념

| 개념 | 설명 |
|------|------|
| [DDPM](/ko/docs/math/diffusion/ddpm) | 기본 Diffusion 모델 |
| [Score Matching](/ko/docs/math/diffusion/score-matching) | Score 함수 학습 |
| [Sampling](/ko/docs/math/diffusion/sampling) | 생성 과정 (DDIM 등) |

## 관련 콘텐츠

- [확률 분포](/ko/docs/math/probability/distribution)
- [샘플링](/ko/docs/math/probability/sampling)
