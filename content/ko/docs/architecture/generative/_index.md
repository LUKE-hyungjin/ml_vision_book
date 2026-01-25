---
title: "Generative"
weight: 6
bookCollapseSection: true
---

# 생성 모델 (Generative Models)

{{% hint info %}}
**선수지식**: [확률분포](/ko/docs/math/probability) | [Neural Network 기초](/ko/docs/architecture/cnn)
{{% /hint %}}

## 생성 모델이란?

> **비유**: 화가가 수천 장의 그림을 보고 "그림의 규칙"을 익히면, 새로운 그림을 그릴 수 있습니다. 생성 모델도 데이터의 "분포"를 학습하여 **새로운 샘플을 창조**합니다.

---

## 세 가지 접근법

{{< figure src="/images/generative/ko/generative-comparison.svg" caption="VAE vs GAN vs Diffusion 비교" >}}

| 모델 | 핵심 아이디어 | 비유 |
|------|--------------|------|
| **VAE** | 압축 → 복원 학습 | 그림을 요약하고 다시 그리기 |
| **GAN** | 진짜 vs 가짜 경쟁 | 위조지폐범 vs 감별사 |
| **Diffusion** | 노이즈 제거 학습 | 흐린 사진 복원하기 |

---

## 발전 과정

```
2013: VAE - "압축했다 복원하면 새로운 걸 만들 수 있지 않을까?"
  ↓
2014: GAN - "진짜와 구분 못하게 만들면 되지 않을까?"
  ↓
2020: DDPM - "노이즈를 조금씩 제거하면 어떨까?"
  ↓
2022: Stable Diffusion - "작은 공간에서 하면 더 빠르지 않을까?"
  ↓
2023: ControlNet - "포즈나 윤곽을 지정하면 어떨까?"
```

---

## 모델 상세

| 모델 | 방식 | 장점 | 단점 |
|------|------|------|------|
| [VAE](/ko/docs/architecture/generative/vae) | 변분 추론 | 안정적 학습 | 흐릿한 결과 |
| [GAN](/ko/docs/architecture/generative/gan) | 적대적 학습 | 선명한 결과 | 불안정, mode collapse |
| [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) | Latent Diffusion | 고품질, 다양성 | 느린 샘플링 |
| [ControlNet](/ko/docs/architecture/generative/controlnet) | 조건부 Diffusion | 정밀 제어 | 추가 학습 필요 |
| [Qwen Image Edit](/ko/docs/architecture/generative/qwen-image-edit) | VLM + Diffusion | 자연어 편집 | 무거움 |

---

## 왜 Diffusion이 승리했나?

1. **안정적 학습**: GAN처럼 불안정하지 않음
2. **고품질 결과**: VAE보다 선명함
3. **다양성**: mode collapse 없음
4. **조건부 생성**: 텍스트, 이미지 등 다양한 조건 가능

**단점**: 느린 샘플링 → DDIM, LCM 등으로 해결 중!

---

## 관련 콘텐츠

- [Diffusion 수학](/ko/docs/math/diffusion) - Forward/Reverse Process
- [확률 분포](/ko/docs/math/probability) - 생성 모델의 수학적 기초
- [Generation 태스크](/ko/docs/task/generation) - FID, IS 등 평가 지표
