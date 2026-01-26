---
title: "DALL-E"
weight: 9
math: true
---

# DALL-E

{{% hint info %}}
**선수지식**: [VQGAN](/ko/docs/architecture/generative/vqgan) | [Transformer](/ko/docs/architecture/transformer) | [CLIP](/ko/docs/architecture/multimodal/clip)
{{% /hint %}}

## 한 줄 요약

> **"텍스트를 이미지로 바꾸는 최초의 대규모 모델"**

---

## 왜 DALL-E인가?

2021년, OpenAI가 "텍스트만으로 이미지를 생성"하는 모델을 공개했습니다.

```
"아보카도 모양의 안락의자" → 🥑🪑
"우주복을 입은 강아지" → 🐕‍🦺🚀
```

> **비유**: DALL-E는 **"상상력 있는 화가"**입니다. 말로 설명하면 그려줍니다!

---

## DALL-E 버전별 비교

| 버전 | 연도 | 방식 | 해상도 |
|------|------|------|--------|
| **DALL-E 1** | 2021.01 | Transformer + dVAE | 256×256 |
| **DALL-E 2** | 2022.04 | CLIP + Diffusion | 1024×1024 |
| **DALL-E 3** | 2023.10 | 개선된 캡션 + Diffusion | 1024×1024 |

---

## DALL-E 1: Transformer 방식

### 핵심 아이디어

텍스트와 이미지를 **하나의 토큰 시퀀스**로 다룹니다.

```
[텍스트 토큰들] + [이미지 토큰들]
     256개           1024개
         ↓
    Transformer (GPT-3 스타일)
         ↓
    다음 토큰 예측
```

### 구조

```
┌─────────────────────────────────────────────────┐
│                   DALL-E 1                       │
│                                                  │
│  텍스트 ──→ [BPE Tokenizer] ──→ 256 토큰        │
│                                                  │
│  이미지 ──→ [dVAE Encoder] ──→ 32×32 = 1024 토큰│
│                                                  │
│  [텍스트 256 | 이미지 1024] ──→ Transformer      │
│                              12B 파라미터        │
│                                                  │
│  예측된 이미지 토큰 ──→ [dVAE Decoder] ──→ 이미지│
└─────────────────────────────────────────────────┘
```

### dVAE (discrete VAE)

이미지를 이산 토큰으로 변환:

$$
\text{이미지 (256×256×3)} \xrightarrow{\text{dVAE}} \text{토큰 (32×32)}
$$

- 코드북 크기: 8192
- 압축률: 64배 (256² → 32²)

---

## DALL-E 2: CLIP + Diffusion

### 왜 방식을 바꿨나?

DALL-E 1의 한계:
- 256×256 저해상도
- 텍스트 이해력 부족
- 생성 다양성 제한

### 새로운 구조

```
┌──────────────────────────────────────────────────┐
│                    DALL-E 2                       │
│                                                   │
│  텍스트 ──→ [CLIP Text Encoder] ──→ text_embed   │
│                                                   │
│  text_embed ──→ [Prior] ──→ image_embed (예측)   │
│                 (Diffusion)                       │
│                                                   │
│  image_embed ──→ [Decoder] ──→ 이미지            │
│                  (Diffusion)  64→256→1024        │
└──────────────────────────────────────────────────┘
```

### 구성 요소

| 컴포넌트 | 역할 | 설명 |
|----------|------|------|
| **CLIP Text Encoder** | 텍스트 이해 | 텍스트 → 임베딩 |
| **Prior** | 임베딩 변환 | 텍스트 임베딩 → 이미지 임베딩 |
| **Decoder (unCLIP)** | 이미지 생성 | 이미지 임베딩 → 실제 이미지 |

### Prior: 핵심 혁신

CLIP 공간에서 텍스트 임베딩을 이미지 임베딩으로 변환:

$$
P(\text{image\_embed} | \text{text\_embed})
$$

두 가지 방식:
1. **Autoregressive Prior**: 토큰 순차 생성
2. **Diffusion Prior**: Diffusion으로 임베딩 생성 (더 좋음)

---

## DALL-E 3: 캡션 품질 개선

### 핵심 개선점

**문제**: 학습 데이터의 캡션이 부정확함
- "사진" → 실제로는 "해변에서 노을을 바라보는 커플"

**해결**: 합성 캡션 생성
1. 이미지마다 상세한 캡션을 LLM으로 생성
2. 더 정확한 캡션으로 학습
3. 프롬프트 이해력 대폭 향상

```
기존 캡션: "two dogs"
새 캡션: "Two golden retriever puppies playing with a red ball
         in a sunny backyard with green grass"
```

### 프롬프트 재작성

사용자 프롬프트를 더 상세하게 자동 확장:

```
사용자: "고양이"
    ↓ GPT-4로 확장
내부: "A fluffy orange tabby cat sitting on a windowsill,
      soft natural lighting, photorealistic style"
```

---

## DALL-E 시리즈 비교

### 생성 품질

```
DALL-E 1:  ██░░░░░░░░  (256×256, 어색함)
DALL-E 2:  ██████░░░░  (1024×1024, 좋음)
DALL-E 3:  █████████░  (1024×1024, 매우 좋음)
```

### 텍스트 이해

```
프롬프트: "빨간 사과 위에 파란 컵"

DALL-E 1: 사과와 컵이 있지만 위치 관계 무시
DALL-E 2: 대체로 맞지만 가끔 실수
DALL-E 3: 정확하게 이해하고 생성
```

---

## DALL-E 2 상세 구조

### Prior (Diffusion 방식)

```python
def prior_forward(text_embed, timestep):
    """
    텍스트 임베딩에서 이미지 임베딩 예측
    """
    # 노이즈 추가된 이미지 임베딩
    noisy_image_embed = add_noise(image_embed, timestep)

    # 노이즈 예측
    predicted_noise = prior_network(
        noisy_image_embed,
        text_embed,  # 조건
        timestep
    )

    return predicted_noise
```

### Decoder (unCLIP)

```python
def decoder_forward(image_embed, timestep):
    """
    이미지 임베딩에서 실제 이미지 생성
    """
    # 64×64 → 256×256 → 1024×1024
    # 단계별 업샘플링 Diffusion

    image_64 = diffusion_64(image_embed)
    image_256 = upsample_256(image_64, image_embed)
    image_1024 = upsample_1024(image_256, image_embed)

    return image_1024
```

---

## 주요 기능

### 1. Text-to-Image

```
"A teddy bear on a skateboard in Times Square"
→ 타임스퀘어에서 스케이트보드 타는 테디베어
```

### 2. Variations

기존 이미지의 변형 생성:

```
원본 이미지 ──→ CLIP Image Encoder ──→ image_embed
                                           ↓
            ──→ Decoder (+ 노이즈) ──→ 변형 이미지들
```

### 3. Inpainting

이미지 일부 영역 수정:

```
원본 + 마스크 + "파란색 모자"
→ 마스크 영역에 파란색 모자 생성
```

---

## 한계점

| 한계 | 설명 |
|------|------|
| **텍스트 렌더링** | 글자 생성이 어려움 |
| **손/얼굴** | 세밀한 구조 오류 |
| **공간 관계** | "A 위에 B" 정확도 낮음 (DALL-E 3에서 개선) |
| **비공개** | 모델 가중치 비공개 |

---

## DALL-E vs Stable Diffusion

| 특성 | DALL-E | Stable Diffusion |
|------|--------|------------------|
| 개발사 | OpenAI | Stability AI |
| 접근성 | API 유료 | 오픈소스 |
| 품질 | 높음 | 높음 |
| 커스터마이징 | 제한적 | LoRA, ControlNet 등 |
| 안전 필터 | 강력함 | 선택적 |

---

## 요약

| 질문 | 답변 |
|------|------|
| DALL-E가 뭔가요? | OpenAI의 텍스트→이미지 생성 모델 |
| DALL-E 1과 2의 차이? | Transformer → CLIP + Diffusion |
| DALL-E 3의 개선점? | 상세 캡션으로 프롬프트 이해력 향상 |
| 왜 유명한가요? | Text-to-Image의 시작점 |

---

## 관련 콘텐츠

- [VQGAN](/ko/docs/architecture/generative/vqgan) - DALL-E 1의 이미지 토큰화
- [CLIP](/ko/docs/architecture/multimodal/clip) - DALL-E 2의 핵심 컴포넌트
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - 오픈소스 대안
- [Diffusion 수학](/ko/docs/math/generative/ddpm) - Diffusion 원리
