---
title: "Generative"
weight: 6
bookCollapseSection: true
---

# 생성 모델 (Generative Models)

> **선수지식**: [확률 분포](/ko/docs/math/probability/distribution) | [VAE](/ko/docs/architecture/generative/vae) | [GAN](/ko/docs/architecture/generative/gan)

## 왜 필요한가?
생성 모델의 목표는 **데이터를 분류**하는 것이 아니라, 데이터의 패턴을 배워 **새로운 샘플을 만들어내는 것**입니다.
예를 들어 화가가 많은 그림을 보고 화풍의 규칙을 익힌 뒤 새 그림을 그리듯, 모델도 이미지 분포를 학습해 새로운 이미지를 생성합니다.

## 수식/기호 관점에서 보는 생성
생성 모델은 보통 데이터 분포 $p_{\text{data}}(x)$를 근사하는 모델 분포 $p_\theta(x)$를 학습합니다.

$$
\theta^* = \arg\min_\theta D\big(p_{\text{data}}(x) \;\|\; p_\theta(x)\big)
$$

- $x$: 이미지 같은 관측 데이터
- $\theta$: 모델 파라미터
- $p_{\text{data}}(x)$: 실제 데이터 분포
- $p_\theta(x)$: 모델이 학습한 분포
- $D(\cdot\|\cdot)$: 두 분포 차이(예: KL, JS, Wasserstein 등)

## 직관: 세 가지 큰 흐름
{{< figure src="/images/generative/ko/generative-comparison.svg" caption="VAE vs GAN vs Diffusion 비교" >}}

| 계열 | 핵심 아이디어 | 직관 |
|------|--------------|------|
| **VAE** | 잠재공간으로 압축 후 복원 | 요약본으로 다시 원문 복원 |
| **GAN** | 생성자-판별자 경쟁 학습 | 위조지폐범 vs 감별사 |
| **Diffusion** | 노이즈를 점진적으로 제거 | 흐린 사진을 단계적으로 복원 |

## 구현 관점: 어떤 모델을 먼저 볼까?
1. [VAE](/ko/docs/architecture/generative/vae): 확률 모델링과 잠재공간의 출발점
2. [GAN](/ko/docs/architecture/generative/gan): 고품질 생성의 고전
3. [DDPM](/ko/docs/architecture/generative/ddpm): 현대 생성 모델의 주류 기반
4. [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion): 실전 Text-to-Image의 표준
5. [Flux](/ko/docs/architecture/generative/flux): 최신 Flow Matching 계열

## 발전 과정 (요약)
- 2013: VAE
- 2014: GAN
- 2018: StyleGAN
- 2020: DDPM
- 2021: VQGAN, DALL-E
- 2022: Stable Diffusion
- 2023: ControlNet, DiT
- 2024: Flux

## 관련 콘텐츠
- [생성 모델 수학](/ko/docs/components/generative)
- [Flow Matching](/ko/docs/components/generative/flow-matching)
- [확률 분포](/ko/docs/math/probability/distribution)
- [Generation 태스크](/ko/docs/task/generation)
