---
title: "VQGAN"
weight: 8
math: true
---

# VQGAN (Vector Quantized GAN)

{{% hint info %}}
**선수지식**: [VAE](/ko/docs/architecture/generative/vae) | [GAN](/ko/docs/architecture/generative/gan) | [Transformer](/ko/docs/architecture/transformer)
{{% /hint %}}

## 한 줄 요약

> **"이미지를 단어처럼 토큰화해서 Transformer로 생성한다"**

---

## 왜 VQGAN인가?

### 문제: Transformer로 이미지를 생성하고 싶다

Transformer는 텍스트에서 놀라운 성능을 보여줬습니다. 이미지도 가능할까?

```
텍스트: "나는 학교에 간다" → [나는, 학교에, 간다] (토큰)
이미지: 256×256×3 = 196,608 픽셀... 너무 많음!
```

### 해결: 이미지를 압축된 토큰으로!

```
이미지 (256×256)
    ↓ VQGAN Encoder
코드북 인덱스 (16×16 = 256개 토큰)
    ↓ Transformer
새로운 토큰 시퀀스
    ↓ VQGAN Decoder
새로운 이미지!
```

> **비유**: 이미지를 **"레고 블록 번호"**로 변환합니다. "빨간 2×4 블록, 파란 1×2 블록..."처럼 이미지를 토큰의 나열로 표현!

---

## 핵심 구조

```
┌──────────────────────────────────────────────────┐
│                     VQGAN                         │
│                                                   │
│  이미지 ──→ [Encoder] ──→ z ──→ [Quantize] ──→ z_q│
│  (256×256)    CNN      (16×16)   코드북 조회       │
│                                                   │
│  z_q ──→ [Decoder] ──→ 복원 이미지                │
│          CNN + GAN                                │
│                                                   │
│  + Discriminator (진짜/가짜 판별)                 │
└──────────────────────────────────────────────────┘
```

### 구성 요소

| 컴포넌트 | 역할 | 비유 |
|----------|------|------|
| **Encoder** | 이미지 → 연속 벡터 | "스케치 추출" |
| **Codebook** | 벡터 → 이산 토큰 | "레고 블록 카탈로그" |
| **Decoder** | 토큰 → 이미지 | "레고 조립" |
| **Discriminator** | 품질 향상 | "품질 검사관" |

---

## Vector Quantization (VQ)

### 핵심 아이디어

연속적인 벡터를 **가장 가까운 코드북 벡터**로 대체합니다.

```
Encoder 출력: z = [0.12, -0.34, 0.56, ...]
                    ↓ 가장 가까운 코드 찾기
코드북:        e_42 = [0.10, -0.35, 0.55, ...]
                    ↓
양자화 결과:   z_q = e_42, index = 42
```

### 수식

$$
z_q = e_k, \quad \text{where} \quad k = \arg\min_j \|z - e_j\|_2
$$

**기호 설명:**
- $z$: Encoder 출력 벡터
- $e_j$: 코드북의 j번째 벡터
- $k$: 가장 가까운 코드의 인덱스
- $z_q$: 양자화된 벡터

### 코드북 크기

```
코드북 크기: 8192개 (일반적)
각 코드 차원: 256

→ 16×16 = 256개 위치
→ 각 위치에서 8192개 중 1개 선택
→ 총 8192^256 가지 이미지 표현 가능!
```

---

## 손실 함수

### 전체 손실

$$
L = L_{\text{recon}} + L_{\text{VQ}} + \lambda L_{\text{GAN}}
$$

### 1. 재구성 손실 (Reconstruction Loss)

$$
L_{\text{recon}} = \|x - \hat{x}\|_1
$$

원본과 복원 이미지의 차이 (L1 loss)

### 2. VQ 손실 (Codebook Loss)

$$
L_{\text{VQ}} = \|sg[z] - e\|_2^2 + \beta \|z - sg[e]\|_2^2
$$

**기호 설명:**
- $sg[\cdot]$: stop gradient (역전파 차단)
- 첫 번째 항: 코드북을 encoder 출력에 맞추기
- 두 번째 항: encoder를 코드북에 맞추기 (commitment loss)

### 3. GAN 손실 (Adversarial Loss)

$$
L_{\text{GAN}} = \log D(x) + \log(1 - D(\hat{x}))
$$

Discriminator로 이미지 품질 향상

---

## VAE vs VQ-VAE vs VQGAN

| 모델 | 잠재 공간 | 품질 | 특징 |
|------|----------|------|------|
| **VAE** | 연속 (가우시안) | 흐릿함 | 간단 |
| **VQ-VAE** | 이산 (코드북) | 보통 | 토큰화 가능 |
| **VQGAN** | 이산 + GAN | 선명함 | 고품질 토큰화 |

```
VAE:    z ~ N(μ, σ²)     → 연속 → 흐릿함
VQ-VAE: z_q ∈ Codebook   → 이산 → 토큰화 가능
VQGAN:  VQ-VAE + GAN     → 이산 + 선명함!
```

---

## Transformer와 결합

### 학습 과정

```
1단계: VQGAN 학습 (이미지 ↔ 토큰)
2단계: Transformer 학습 (토큰 시퀀스 예측)
```

### 생성 과정

```python
# 1. 텍스트를 조건으로 토큰 생성
text = "a photo of a cat"
tokens = transformer.generate(text_embedding)  # [16×16 = 256 토큰]

# 2. 토큰을 이미지로 변환
image = vqgan.decode(tokens)
```

---

## 코드 예시

```python
import torch
import torch.nn as nn

class VQGAN(nn.Module):
    def __init__(self, num_codebook=8192, codebook_dim=256):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # 코드북: [8192, 256]
        self.codebook = nn.Embedding(num_codebook, codebook_dim)

    def quantize(self, z):
        """가장 가까운 코드북 벡터 찾기"""
        # z: [B, C, H, W] → [B, H, W, C]
        z = z.permute(0, 2, 3, 1)

        # 거리 계산
        d = torch.cdist(z, self.codebook.weight)

        # 가장 가까운 인덱스
        indices = d.argmin(dim=-1)

        # 코드북에서 가져오기
        z_q = self.codebook(indices)

        return z_q.permute(0, 3, 1, 2), indices

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices = self.quantize(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q, indices
```

---

## VQGAN의 영향

VQGAN은 많은 모델의 기반이 되었습니다:

| 모델 | VQGAN 활용 |
|------|-----------|
| **DALL-E** | 이미지 토큰화 |
| **Stable Diffusion** | VAE의 아이디어 차용 |
| **Parti** | 이미지 토큰 + Transformer |
| **Make-A-Scene** | 장면 생성 |

---

## 장단점

### 장점

| 장점 | 설명 |
|------|------|
| **고품질 압축** | 16배 압축에도 선명 |
| **Transformer 호환** | 언어 모델 기술 활용 |
| **다양한 응용** | 생성, 편집, 압축 |

### 단점

| 단점 | 설명 |
|------|------|
| **코드북 붕괴** | 일부 코드만 사용됨 |
| **2단계 학습** | VQGAN → Transformer |
| **Diffusion 대비** | 다양성이 낮음 |

---

## 요약

| 질문 | 답변 |
|------|------|
| VQGAN이 뭔가요? | 이미지를 이산 토큰으로 변환하는 모델 |
| VQ가 뭔가요? | 연속 벡터 → 코드북의 가장 가까운 벡터 |
| 왜 GAN을 추가하나요? | 이미지 품질 향상 (선명도) |
| 어디에 쓰이나요? | DALL-E, 이미지 Transformer 등 |

---

## 관련 콘텐츠

- [VAE](/ko/docs/architecture/generative/vae) - 기본 오토인코더
- [GAN](/ko/docs/architecture/generative/gan) - 적대적 학습
- [DALL-E](/ko/docs/architecture/generative/dall-e) - VQGAN 활용 모델
- [Transformer](/ko/docs/architecture/transformer) - 시퀀스 모델링
