---
title: "DiT"
weight: 2
math: true
---

# DiT (Diffusion Transformer)

## 개요

- **논문**: Scalable Diffusion Models with Transformers (2023)
- **저자**: William Peebles, Saining Xie (Meta AI, NYU)
- **핵심 기여**: Diffusion 모델의 backbone을 U-Net에서 Transformer로 교체

## 핵심 아이디어

> "U-Net 대신 Transformer를 쓰면 scaling이 더 잘 된다"

기존 Diffusion 모델(Stable Diffusion 등)은 U-Net을 사용하지만, DiT는 Transformer를 사용하여 더 나은 scaling 특성을 보여줍니다.

---

## 배경

### 기존 Diffusion 모델의 backbone

```
Stable Diffusion: Latent → U-Net → Denoised Latent
```

- U-Net은 CNN 기반
- Skip connection으로 세부 정보 보존
- 하지만 모델 크기 증가 시 효율 저하

### DiT의 접근

```
Latent → Patchify → Transformer → Unpatchify → Denoised Latent
```

- ViT 스타일로 latent를 패치화
- Transformer로 처리
- 모델 크기에 따라 일관된 성능 향상

---

## 구조

### 전체 아키텍처

```
Input: Noisy Latent (z_t) + Timestep (t) + Class (c)
            ↓
    Patchify (latent → patches)
            ↓
    + Position Embedding
            ↓
┌─────────────────────────────────────┐
│         DiT Blocks × N             │
│  ┌─────────────────────────────┐   │
│  │     Layer Norm              │   │
│  │           ↓                 │   │
│  │    Multi-Head Self-Attn    │   │
│  │           ↓                 │   │
│  │     Layer Norm              │   │
│  │           ↓                 │   │
│  │    Pointwise FFN           │   │
│  │           ↓                 │   │
│  │    + AdaLN (t, c embed)    │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
            ↓
    Final Layer Norm
            ↓
    Linear (predict noise & variance)
            ↓
    Unpatchify
            ↓
Output: Predicted Noise (ε) + Variance (Σ)
```

### AdaLN (Adaptive Layer Norm)

조건 정보(timestep, class)를 Layer Norm에 주입:

$$\text{AdaLN}(h, y) = y_s \odot \text{LayerNorm}(h) + y_b$$

여기서 $y_s$, $y_b$는 timestep과 class embedding에서 예측된 scale, shift 값

---

## 조건 주입 방식

### 세 가지 변형

| 방식 | 설명 | 성능 |
|------|------|------|
| In-context | 조건을 추가 토큰으로 | 중간 |
| Cross-attention | 조건에 cross-attention | 중간 |
| **AdaLN-Zero** | Adaptive LN + zero 초기화 | **최고** |

### AdaLN-Zero

각 블록의 출력을 0으로 초기화:
- 학습 초기에 identity mapping
- 안정적인 학습

---

## 모델 변형

| 모델 | Layers | Hidden | Heads | Params | FID-50K |
|------|--------|--------|-------|--------|---------|
| DiT-S/2 | 12 | 384 | 6 | 33M | 68.4 |
| DiT-B/2 | 12 | 768 | 12 | 130M | 43.5 |
| DiT-L/2 | 24 | 1024 | 16 | 458M | 23.3 |
| DiT-XL/2 | 28 | 1152 | 16 | 675M | **2.27** |

- `/2`는 patch size 2 의미 (latent space에서)
- 모델 크기 증가에 따른 일관된 성능 향상

---

## Scaling 특성

### Gflops vs FID

DiT는 compute가 증가할수록 일관되게 FID가 개선:

```
Gflops:  10 → 100 → 1000
FID:     68 →  43 →  2.27
```

이는 LLM에서 관찰되는 scaling law와 유사합니다.

### 왜 Transformer가 잘 scaling 되는가?

1. **Attention의 유연성**: 전역적 패턴 학습
2. **표준화된 구조**: 최적화 기법 적용 용이
3. **병렬화**: 효율적인 학습

---

## 구현 예시

```python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        # AdaLN parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)  # scale, shift for both norms + gate
        )

    def forward(self, x, c):
        # c: condition embedding (timestep + class)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * h

        # FFN with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, hidden_size=1152,
                 depth=28, num_heads=16, num_classes=1000):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2

        # Patch embedding
        self.x_embedder = nn.Conv2d(4, hidden_size, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Condition embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes, hidden_size)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # Output
        self.final_layer = FinalLayer(hidden_size, patch_size, 4)

    def forward(self, x, t, y):
        # Patchify
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Condition
        c = self.t_embedder(t) + self.y_embedder(y)

        # DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # Unpatchify
        x = self.final_layer(x, c)
        return x
```

---

## DiT의 영향

### 후속 모델들

- **Sora (OpenAI)**: DiT 기반 비디오 생성
- **Stable Diffusion 3**: DiT 아키텍처 채택
- **PixArt-α**: 효율적인 DiT 학습

### 왜 중요한가?

1. Diffusion 모델의 새로운 방향 제시
2. Scaling law 확인
3. Transformer의 범용성 증명

---

## U-Net vs DiT

| 측면 | U-Net | DiT |
|------|-------|-----|
| 구조 | CNN + Skip | Transformer |
| Scaling | 제한적 | 일관적 향상 |
| 계산량 | 상대적 효율 | 더 많은 compute |
| 구현 복잡도 | 복잡 | 단순 |

---

## 관련 콘텐츠

- [Transformer](/ko/docs/architecture/transformer) - 기반 아키텍처
- [ViT](/ko/docs/architecture/transformer/vit) - 이미지 패치화 아이디어
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - U-Net 기반 Diffusion
- [Diffusion 수학](/ko/docs/math/generative/ddpm) - Diffusion 수식
