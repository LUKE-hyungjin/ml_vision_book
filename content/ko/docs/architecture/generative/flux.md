---
title: "Flux"
weight: 10
math: true
---

# Flux

{{% hint info %}}
**선수지식**: [Flow Matching](/ko/docs/components/generative/flow-matching) | [DiT](/ko/docs/architecture/generative/dit) | [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion)
{{% /hint %}}

## 한 줄 요약

> **"Flow Matching + DiT = 더 빠르고 더 좋은 이미지 생성"**

---

## 왜 Flux인가?

Stable Diffusion 개발자들이 만든 **차세대 이미지 생성 모델**입니다.

```
Stable Diffusion (2022)
    ↓ 같은 팀이 개발
Flux (2024) - 더 빠르고, 더 좋고, 더 유연함
```

### Stable Diffusion vs Flux

| 특성 | Stable Diffusion | Flux |
|------|------------------|------|
| Backbone | U-Net | DiT (Transformer) |
| 수학 | DDPM | Flow Matching |
| 스텝 | 20-50 | 4-20 |
| 품질 | 좋음 | 더 좋음 |
| 텍스트 이해 | 보통 | 우수 |

> **비유**: SD가 "전통 유화"라면, Flux는 "디지털 아트 + AI"입니다. 더 빠르고 정확합니다!

---

## 핵심 구조

```
┌───────────────────────────────────────────────────┐
│                      Flux                          │
│                                                    │
│  텍스트 ──→ [T5 + CLIP] ──→ text_embed            │
│                                                    │
│  노이즈 ──→ [VAE Encoder] ──→ latent              │
│                                                    │
│  latent + text_embed ──→ [DiT] ──→ denoised       │
│                         Flow Matching              │
│                                                    │
│  denoised ──→ [VAE Decoder] ──→ 이미지            │
└───────────────────────────────────────────────────┘
```

### 구성 요소

| 컴포넌트 | 역할 | 특징 |
|----------|------|------|
| **T5-XXL** | 텍스트 인코딩 | 4.7B 파라미터, 강력한 언어 이해 |
| **CLIP** | 텍스트-이미지 정렬 | 보조 텍스트 인코더 |
| **VAE** | 이미지 압축 | 8배 압축 (SD와 유사) |
| **DiT** | 노이즈 제거 | Transformer 기반 |

---

## Flow Matching 적용

### DDPM vs Flow Matching

```
DDPM (Stable Diffusion):
- 1000 스텝 정의, 보통 20-50 스텝 사용
- 확률적 경로 (SDE)
- 노이즈 예측: ε_θ(x_t, t)

Flow Matching (Flux):
- 연속적 시간 [0, 1]
- 결정론적 경로 (ODE)
- 속도 예측: v_θ(x_t, t)
```

### 장점

```
Flow Matching으로 인한 이점:
✓ 더 적은 스텝 (4-20)
✓ 더 안정적 학습
✓ 더 나은 품질
✓ 더 빠른 샘플링
```

---

## DiT (Diffusion Transformer)

### U-Net → Transformer

```
기존 (SD):
┌─────────────────────┐
│       U-Net         │
│  Down → Middle → Up │
│  Skip Connections   │
└─────────────────────┘

Flux:
┌─────────────────────┐
│        DiT          │
│  Transformer Blocks │
│  + Adaptive LayerNorm│
└─────────────────────┘
```

### 왜 Transformer인가?

| 장점 | 설명 |
|------|------|
| **확장성** | 파라미터 늘리면 성능 향상 |
| **통합** | 텍스트와 이미지를 같은 방식으로 처리 |
| **효율성** | 최적화된 attention 사용 가능 |

---

## Flux 모델 라인업

| 모델 | 파라미터 | 특징 | 라이선스 |
|------|----------|------|----------|
| **Flux.1 [schnell]** | 12B | 4스텝, 초고속 | Apache 2.0 |
| **Flux.1 [dev]** | 12B | 개발/연구용 | 비상업적 |
| **Flux.1 [pro]** | 12B | 최고 품질 | API 전용 |

### schnell vs dev vs pro

```
schnell (빠름):
- 4스텝 생성
- 실시간 미리보기용
- 품질: ████████░░

dev (개발):
- 20스텝 생성
- 연구/실험용
- 품질: █████████░

pro (프로):
- 최적화된 스텝
- 상업용 API
- 품질: ██████████
```

---

## 듀얼 텍스트 인코더

### T5-XXL + CLIP

```
텍스트: "A cat wearing a top hat"
           ↓
    ┌──────┴──────┐
    ↓             ↓
 [T5-XXL]      [CLIP]
    ↓             ↓
 깊은 이해     이미지 정렬
    └──────┬──────┘
           ↓
    Combined Embedding
```

### 왜 두 개의 인코더?

| 인코더 | 역할 | 강점 |
|--------|------|------|
| **T5-XXL** | 언어 이해 | 복잡한 문장, 관계 이해 |
| **CLIP** | 시각-언어 정렬 | 이미지와의 연결 |

> **비유**: T5는 "문법 선생님", CLIP은 "미술 평론가". 둘이 협력!

---

## 학습 데이터

```
내부 데이터셋 (비공개):
- 수십억 개의 이미지-텍스트 쌍
- 고품질 캡션 (합성 포함)
- 다양한 스타일과 도메인
```

### 캡션 품질의 중요성

```
나쁜 캡션: "image"
좋은 캡션: "A photorealistic image of a golden retriever
          puppy playing in autumn leaves, warm sunlight,
          shallow depth of field, Canon EOS R5"
```

---

## 코드 예시

```python
# diffusers 라이브러리 사용
from diffusers import FluxPipeline
import torch

# 모델 로드
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# 이미지 생성
prompt = "A serene Japanese garden with cherry blossoms"
image = pipe(
    prompt,
    num_inference_steps=20,
    guidance_scale=3.5,
    height=1024,
    width=1024,
).images[0]

image.save("output.png")
```

### schnell (초고속) 버전

```python
# 4스텝만으로 생성!
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

image = pipe(
    prompt,
    num_inference_steps=4,  # 4스텝!
    guidance_scale=0.0,     # CFG 불필요
).images[0]
```

---

## Flux vs 경쟁 모델

| 특성 | Flux | SDXL | DALL-E 3 | Midjourney |
|------|------|------|----------|------------|
| 오픈소스 | O (일부) | O | X | X |
| 품질 | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ |
| 속도 | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★★☆ |
| 텍스트 이해 | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| 커스터마이징 | ★★★★☆ | ★★★★★ | ★☆☆☆☆ | ★★☆☆☆ |

---

## 확장 기능

### LoRA 지원

```python
# Flux용 LoRA 적용
pipe.load_lora_weights("path/to/flux_lora.safetensors")
```

### ControlNet (개발 중)

```python
# 조건부 생성
# - 포즈 제어
# - 깊이 맵
# - 엣지 검출
```

---

## 요약

| 질문 | 답변 |
|------|------|
| Flux가 뭔가요? | Flow Matching + DiT 기반 이미지 생성 모델 |
| SD와 뭐가 다른가요? | 더 빠름 (4스텝), 더 좋은 품질, Transformer 사용 |
| 왜 빠른가요? | Flow Matching (직선 경로) + 최적화된 DiT |
| 어떤 버전을 쓸까요? | 빠름: schnell, 품질: dev/pro |

---

## 관련 콘텐츠

- [Flow Matching](/ko/docs/components/generative/flow-matching) - 수학적 기반
- [DiT](/ko/docs/architecture/generative/dit) - 아키텍처 기반
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - 이전 세대
- [DDPM](/ko/docs/components/generative/ddpm) - 전통적 방식과 비교
