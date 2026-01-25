---
title: "Generative"
weight: 6
bookCollapseSection: true
---

# 생성 모델 (Generative Models)

데이터의 분포를 학습하여 새로운 샘플을 생성하는 모델들입니다.

## 발전 과정

```
2013: VAE - 확률적 잠재 공간
  ↓
2014: GAN - 적대적 학습
  ↓
2020: DDPM - Diffusion 시작
  ↓
2022: Stable Diffusion - Latent Diffusion
  ↓
2023: ControlNet - 제어 가능한 생성
```

---

## 모델 비교

| 모델 | 방식 | 장점 | 단점 |
|------|------|------|------|
| [VAE](/ko/docs/architecture/generative/vae) | 변분 추론 | 안정적 학습 | 흐릿한 결과 |
| [GAN](/ko/docs/architecture/generative/gan) | 적대적 학습 | 선명한 결과 | 불안정, mode collapse |
| [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) | Diffusion | 고품질, 다양성 | 느린 샘플링 |
| [ControlNet](/ko/docs/architecture/generative/controlnet) | 조건부 Diffusion | 정밀 제어 | 추가 학습 필요 |

---

## 관련 콘텐츠

- [Diffusion Process](/ko/docs/math/diffusion-process) - Diffusion 수학
- [확률분포](/ko/docs/math/probability) - 생성 모델의 기초
- [Generation 태스크](/ko/docs/task/generation) - 평가 지표
