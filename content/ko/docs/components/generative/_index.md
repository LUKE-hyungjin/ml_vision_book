---
title: "생성 모델 수학"
weight: 9
bookCollapseSection: true
math: true
---

# 생성 모델 수학

{{% hint info %}}
**선수지식**: [정규분포](/ko/docs/math/probability) | [기초 미적분 (gradient)](/ko/docs/math/calculus)
{{% /hint %}}

## 왜 Diffusion인가?

> **일상적 비유**: 잉크 한 방울을 물에 떨어뜨리면 천천히 퍼져서 균일해집니다. 이 "확산" 과정을 **거꾸로 돌릴 수 있다면**, 균일한 물에서 잉크 방울을 복원할 수 있겠죠?

Diffusion 모델은 이 아이디어를 수학적으로 구현합니다:
- **노이즈 추가**는 쉬움 (물리 법칙)
- **노이즈 제거**는 신경망이 학습

---

## 핵심 아이디어: Forward & Reverse

{{< figure src="/images/math/generative/ddpm/ko/diffusion-process.svg" caption="Diffusion의 Forward/Reverse 과정" >}}

| 과정 | 방향 | 설명 |
|------|------|------|
| **Forward** | 이미지 → 노이즈 | 조금씩 노이즈를 추가 (수학적으로 정의됨) |
| **Reverse** | 노이즈 → 이미지 | 조금씩 노이즈를 제거 (**신경망이 학습**) |

**핵심 통찰**: Forward는 단순한 수학이지만, Reverse를 배우면 **무에서 유를 창조**할 수 있습니다!

---

## Latent Diffusion: 더 빠르게

{{< figure src="/images/math/generative/ddpm/ko/latent-diffusion.svg" caption="Pixel Diffusion vs Latent Diffusion" >}}

**문제**: 512×512 이미지에서 직접 Diffusion → 너무 느림!

**해결**: 이미지를 **작은 잠재 공간(64×64)**으로 압축 후 Diffusion
- VAE Encoder: 512×512 → 64×64 (압축)
- Diffusion: 64×64에서 수행 (빠름!)
- VAE Decoder: 64×64 → 512×512 (복원)

→ Stable Diffusion이 이 방식 사용

---

## 핵심 개념

| 개념 | 설명 | 핵심 질문 |
|------|------|----------|
| [DDPM](/ko/docs/components/generative/ddpm) | Forward/Reverse 과정, Loss 유도 | "수학적으로 어떻게 정의되나?" |
| [Score Matching](/ko/docs/components/generative/score-matching) | Score 함수 학습 | "gradient 방향을 어떻게 아나?" |
| [Sampling](/ko/docs/components/generative/sampling) | 생성 과정 | "더 빨리 생성하려면?" |
| [Flow Matching](/ko/docs/components/generative/flow-matching) | 직선 경로 학습 | "더 빠르고 간단하게?" |

---

## 관련 콘텐츠

- [확률 분포](/ko/docs/math/probability/distribution) - 노이즈는 가우시안 분포
- [샘플링](/ko/docs/math/probability/sampling) - 분포에서 샘플 추출
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - Latent Diffusion 구현체
