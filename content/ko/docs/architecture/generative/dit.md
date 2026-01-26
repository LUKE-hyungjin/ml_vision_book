---
title: "DiT"
weight: 11
math: true
---

# DiT (Diffusion Transformer)

{{% hint info %}}
**선수지식**: [Transformer](/ko/docs/architecture/transformer) | [DDPM](/ko/docs/math/generative/ddpm) | [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion)
{{% /hint %}}

## 한 줄 요약

> **"U-Net 대신 Transformer로 Diffusion을 하면 어떨까?"**

---

## 왜 DiT인가?

### 문제: U-Net의 한계

Stable Diffusion은 U-Net을 사용합니다. 하지만...

```
U-Net의 한계:
- 구조가 복잡 (encoder-decoder + skip connections)
- 스케일링이 어려움
- Vision Transformer 발전을 활용 못함
```

### 해결: Transformer!

```
DiT의 접근:
- 단순한 Transformer 블록
- 파라미터 늘리면 성능 향상 (스케일링 법칙)
- ViT의 발전을 그대로 활용
```

> **비유**: U-Net은 "맞춤 제작 도구", Transformer는 "범용 공구". 범용 공구가 더 발전 속도가 빠릅니다!

---

## U-Net vs DiT

| 특성 | U-Net | DiT |
|------|-------|-----|
| 구조 | Encoder-Decoder | Transformer 블록 |
| Skip Connection | 필수 | 불필요 |
| 스케일링 | 어려움 | 쉬움 |
| 구현 복잡도 | 높음 | 낮음 |
| 최신 기술 적용 | 제한적 | ViT 기술 활용 |

```
U-Net:
  ┌───┐     ┌───┐
  │ E │─────│ D │
  │ n │ skip│ e │
  │ c │─────│ c │
  └───┘     └───┘

DiT:
  ┌─────────────┐
  │ Transformer │
  │   Block     │
  │  × N 반복   │
  └─────────────┘
```

---

## DiT 구조

### 전체 아키텍처

```
┌────────────────────────────────────────────────┐
│                      DiT                        │
│                                                 │
│  노이즈 이미지 (256×256)                        │
│       ↓                                         │
│  [Patchify] - 이미지를 패치로 분할              │
│       ↓                                         │
│  패치 토큰 (16×16 = 256개)                      │
│       ↓                                         │
│  [Linear Embedding]                             │
│       ↓                                         │
│  + Positional Embedding                         │
│       ↓                                         │
│  [DiT Block] × N                                │
│    - Self-Attention                             │
│    - MLP                                        │
│    - AdaLN (조건 주입)                          │
│       ↓                                         │
│  [Unpatchify] - 토큰을 이미지로 복원            │
│       ↓                                         │
│  예측된 노이즈                                  │
└────────────────────────────────────────────────┘
```

### Patchify

이미지를 패치로 나눕니다 (ViT와 동일):

$$
\text{이미지 (256×256)} \rightarrow \text{패치 (16×16개, 각 16×16 크기)}
$$

```
┌──┬──┬──┬──┐
│ 1│ 2│ 3│ 4│   256×256 이미지
├──┼──┼──┼──┤        ↓
│ 5│ 6│ 7│ 8│   16×16 = 256개 패치
├──┼──┼──┼──┤   각 패치: 16×16×3
│ 9│10│11│12│
├──┼──┼──┼──┤
│13│14│15│16│
└──┴──┴──┴──┘
```

---

## DiT Block

### 핵심: AdaLN (Adaptive Layer Normalization)

조건(timestep, class)을 주입하는 방법:

$$
\text{AdaLN}(x, c) = \gamma(c) \cdot \text{LayerNorm}(x) + \beta(c)
$$

**기호 설명:**
- $x$: 입력 feature
- $c$: 조건 (timestep + class embedding)
- $\gamma(c), \beta(c)$: 조건에서 유도된 스케일, 시프트

### DiT Block 구조

```python
class DiTBlock(nn.Module):
    def forward(self, x, c):
        # c: timestep + class 조건

        # 1. AdaLN으로 조건 주입
        gamma1, beta1, alpha1 = self.adaln1(c)

        # 2. Self-Attention
        x = x + alpha1 * self.attn(
            gamma1 * self.norm1(x) + beta1
        )

        # 3. AdaLN + MLP
        gamma2, beta2, alpha2 = self.adaln2(c)
        x = x + alpha2 * self.mlp(
            gamma2 * self.norm2(x) + beta2
        )

        return x
```

---

## 조건 주입 방법 비교

DiT 논문에서 여러 방법을 실험:

| 방법 | 설명 | 성능 |
|------|------|------|
| **In-context** | 조건을 토큰으로 추가 | 보통 |
| **Cross-attention** | 별도 attention으로 주입 | 좋음 |
| **AdaLN** | LayerNorm 파라미터 조절 | 좋음 |
| **AdaLN-Zero** | AdaLN + zero 초기화 | **최고** |

### AdaLN-Zero

```
일반 AdaLN:
  α 초기화 = 1 (바로 영향)

AdaLN-Zero:
  α 초기화 = 0 (처음엔 skip, 점점 학습)
```

> **비유**: 처음엔 "조용히 관찰"하다가 점점 "적극 개입"

---

## 스케일링 법칙

DiT의 핵심 발견: **크기를 키우면 성능이 좋아진다!**

| 모델 | 파라미터 | Depth | Hidden | FID ↓ |
|------|----------|-------|--------|-------|
| DiT-S | 33M | 12 | 384 | 68.4 |
| DiT-B | 130M | 12 | 768 | 43.5 |
| DiT-L | 458M | 24 | 1024 | 23.3 |
| DiT-XL | 675M | 28 | 1152 | **9.62** |

```
파라미터 ↑  →  FID ↓  →  품질 ↑

DiT-S: ██░░░░░░░░ (33M)
DiT-XL: ████████░░ (675M)
        └─── 20배 더 크지만 7배 더 좋음!
```

---

## DiT의 영향

DiT 아키텍처를 사용하는 모델들:

| 모델 | 개발사 | 용도 |
|------|--------|------|
| **Stable Diffusion 3** | Stability AI | 이미지 생성 |
| **Flux** | Black Forest Labs | 이미지 생성 |
| **SORA** | OpenAI | 비디오 생성 |
| **Hunyuan-DiT** | Tencent | 이미지 생성 |

```
2023: DiT 논문 발표
  ↓
2024: SD3, Flux, SORA 모두 DiT 기반!
```

---

## 코드 예시

```python
import torch
import torch.nn as nn

class DiT(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=4,      # latent space
        hidden_size=1152,   # DiT-XL
        depth=28,
        num_heads=16,
        num_classes=1000,
    ):
        super().__init__()

        # 패치 임베딩
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

        # 위치 임베딩
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size)
        )

        # 조건 임베딩
        self.time_embed = TimestepEmbedding(hidden_size)
        self.class_embed = nn.Embedding(num_classes, hidden_size)

        # DiT 블록들
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads)
            for _ in range(depth)
        ])

        # 출력층
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

    def forward(self, x, t, y):
        """
        x: 노이즈 latent [B, C, H, W]
        t: timestep [B]
        y: class label [B]
        """
        # 패치화
        x = self.patch_embed(x)  # [B, hidden, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]

        # 위치 임베딩
        x = x + self.pos_embed

        # 조건 임베딩
        c = self.time_embed(t) + self.class_embed(y)

        # DiT 블록 통과
        for block in self.blocks:
            x = block(x, c)

        # 출력
        x = self.final_layer(x, c)

        return x  # 예측된 노이즈
```

---

## Latent DiT

실제로는 픽셀이 아닌 **latent space**에서 동작:

```
이미지 (512×512×3)
    ↓ VAE Encoder
Latent (64×64×4)
    ↓ DiT
Denoised Latent (64×64×4)
    ↓ VAE Decoder
이미지 (512×512×3)
```

**장점:**
- 계산량 64배 감소
- 의미있는 공간에서 작업
- Stable Diffusion과 동일한 VAE 사용 가능

---

## DiT vs U-Net 실험 결과

ImageNet 256×256 생성 (FID, 낮을수록 좋음):

| 모델 | 파라미터 | FID |
|------|----------|-----|
| U-Net (ADM) | 554M | 10.94 |
| **DiT-XL/2** | 675M | **9.62** |

> 비슷한 크기에서 DiT가 더 좋음!

---

## 요약

| 질문 | 답변 |
|------|------|
| DiT가 뭔가요? | Transformer 기반 Diffusion 모델 |
| U-Net과 뭐가 다른가요? | 더 단순하고 스케일링이 쉬움 |
| 왜 중요한가요? | SD3, Flux, SORA의 핵심 아키텍처 |
| AdaLN이 뭔가요? | 조건을 LayerNorm으로 주입하는 방법 |

---

## 관련 콘텐츠

- [Transformer](/ko/docs/architecture/transformer) - 기본 Transformer
- [DDPM](/ko/docs/math/generative/ddpm) - Diffusion 수학
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - U-Net 기반
- [Flux](/ko/docs/architecture/generative/flux) - DiT 기반 최신 모델
- [Flow Matching](/ko/docs/math/generative/flow-matching) - DiT와 자주 결합
