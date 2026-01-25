---
title: "Stable Diffusion"
weight: 3
math: true
---

# Stable Diffusion

## 개요

- **논문**: High-Resolution Image Synthesis with Latent Diffusion Models (2022)
- **저자**: Robin Rombach et al. (CompVis, Stability AI)
- **핵심 기여**: Latent space에서 Diffusion을 수행하여 효율적인 고해상도 생성

## 핵심 아이디어

> "픽셀 공간이 아닌 압축된 잠재 공간에서 Diffusion"

기존 DDPM은 픽셀 공간에서 직접 작동해 계산량이 많았습니다. Stable Diffusion은 VAE로 이미지를 압축한 후 latent space에서 diffusion을 수행합니다.

---

## Diffusion 기본 개념

### Forward Process (노이즈 추가)

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

점진적으로 노이즈를 추가하여 순수 노이즈로 변환

### Reverse Process (노이즈 제거)

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

학습된 모델로 노이즈를 제거하며 이미지 복원

---

## 구조

### 전체 아키텍처

```
Text Prompt
      ↓
┌─────────────────────────────┐
│      Text Encoder (CLIP)    │
└─────────────────────────────┘
      ↓
Text Embedding
      ↓                        Random Noise
      ↓                             ↓
┌─────────────────────────────────────────────┐
│                   U-Net                      │
│   (Latent Diffusion in compressed space)    │
│                                              │
│   Input: z_t + timestep + text_embedding    │
│   Output: predicted noise ε                 │
└─────────────────────────────────────────────┘
      ↓
Denoised Latent z_0
      ↓
┌─────────────────────────────┐
│      VAE Decoder            │
└─────────────────────────────┘
      ↓
Generated Image (512×512 or higher)
```

### 핵심 컴포넌트

| 컴포넌트 | 역할 |
|----------|------|
| **VAE** | 이미지 ↔ Latent 변환 (8× 압축) |
| **U-Net** | 조건부 노이즈 예측 |
| **Text Encoder** | 텍스트 → 임베딩 (CLIP) |
| **Scheduler** | 샘플링 전략 (DDPM, DDIM 등) |

---

## VAE (Variational Autoencoder)

512×512 이미지를 64×64×4 latent로 압축:

```
Image (512×512×3) → VAE Encoder → Latent (64×64×4)
Latent (64×64×4) → VAE Decoder → Image (512×512×3)
```

**장점:**
- 계산량 64배 감소 (512² vs 64²)
- 메모리 효율적
- 의미있는 latent space

---

## U-Net

조건부 노이즈 예측 네트워크:

```python
# U-Net 입력
z_t: 노이즈가 추가된 latent (64×64×4)
t: timestep embedding
c: text embedding (cross-attention으로 주입)

# U-Net 출력
epsilon: 예측된 노이즈 (64×64×4)
```

### Cross-Attention

텍스트 조건을 U-Net에 주입:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

- Q: 이미지 특징에서
- K, V: 텍스트 임베딩에서

---

## 학습

### 목적 함수

$$L = \mathbb{E}_{z, \epsilon, t, c} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

- $z$: 원본 이미지의 latent
- $\epsilon$: 추가된 노이즈
- $t$: timestep
- $c$: 텍스트 조건
- $\epsilon_\theta$: 예측된 노이즈

### 학습 데이터

- **LAION-5B**: 50억 이미지-텍스트 쌍
- 대규모 데이터로 일반화 능력 확보

---

## 샘플링

### DDPM Sampling

```python
# T=1000 스텝에서 시작
z_T = torch.randn_like(latent)

for t in reversed(range(T)):
    # 노이즈 예측
    eps = unet(z_t, t, text_embedding)
    # 한 스텝 denoise
    z_{t-1} = denoise_step(z_t, eps, t)

# VAE Decoder로 이미지 변환
image = vae.decode(z_0)
```

### DDIM Sampling (빠른 버전)

50 스텝으로도 좋은 품질:
- Deterministic sampling
- Skip steps 가능

---

## 구현 예시

### Diffusers 라이브러리

```python
from diffusers import StableDiffusionPipeline
import torch

# 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 이미지 생성
prompt = "a photo of an astronaut riding a horse"
image = pipe(prompt).images[0]
image.save("output.png")
```

### 세부 제어

```python
image = pipe(
    prompt="a beautiful landscape",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,  # CFG 강도
    height=512,
    width=512,
    generator=torch.Generator().manual_seed(42)
).images[0]
```

---

## Classifier-Free Guidance (CFG)

조건부/무조건부 예측을 결합:

$$\epsilon = \epsilon_\theta(z_t, t, \emptyset) + s \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))$$

- $s = 1$: 순수 조건부
- $s > 1$: 조건 강화 (보통 7.5 사용)
- 높을수록 프롬프트에 충실하지만 다양성 감소

---

## Stable Diffusion 버전

| 버전 | 특징 |
|------|------|
| **SD 1.x** | CLIP ViT-L/14, 512×512 |
| **SD 2.x** | OpenCLIP, 768×768, depth |
| **SDXL** | 1024×1024, 2개 text encoder |
| **SD 3** | DiT 아키텍처, Flow Matching |

---

## 활용

### Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)
new_image = pipe(
    prompt="oil painting style",
    image=init_image,
    strength=0.75  # 0: 원본 유지, 1: 완전 재생성
).images[0]
```

### Inpainting

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(...)
result = pipe(
    prompt="a cat sitting",
    image=image,
    mask_image=mask  # 수정할 영역
).images[0]
```

---

## 관련 콘텐츠

- [Diffusion Process](/ko/docs/math/diffusion-process) - Diffusion 수학
- [VAE](/ko/docs/architecture/generative/vae) - Latent 압축
- [ControlNet](/ko/docs/architecture/generative/controlnet) - 추가 조건 제어
- [DiT](/ko/docs/architecture/transformer/dit) - Transformer 기반 Diffusion
- [Generation 태스크](/ko/docs/task/generation) - 평가 지표
