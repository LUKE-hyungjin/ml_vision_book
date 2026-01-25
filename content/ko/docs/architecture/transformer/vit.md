---
title: "ViT"
weight: 1
math: true
---

# ViT (Vision Transformer)

## 개요

- **논문**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)
- **저자**: Alexey Dosovitskiy et al. (Google)
- **핵심 기여**: 이미지를 패치로 나누어 순수 Transformer로 처리

## 핵심 아이디어

> "이미지를 단어처럼 취급하자"

이미지를 16×16 패치로 나누고, 각 패치를 토큰처럼 Transformer에 입력합니다.

---

## 구조

### 전체 아키텍처

```
Input Image (224×224×3)
        ↓
Split into patches (14×14 = 196 patches of 16×16)
        ↓
Linear projection (flatten + linear → D dim)
        ↓
+ Positional Embedding + [CLS] token
        ↓
┌─────────────────────────────┐
│    Transformer Encoder      │
│  [Multi-Head Attn + FFN]×L │
└─────────────────────────────┘
        ↓
[CLS] token output
        ↓
MLP Head → Classification
```

### 핵심 컴포넌트

| 컴포넌트 | 설명 |
|----------|------|
| Patch Embedding | 16×16×3 → D 차원 벡터 |
| [CLS] Token | 전체 이미지 표현 학습 |
| Position Embedding | 학습 가능한 위치 인코딩 |
| Transformer Encoder | L개 레이어 |

---

## Patch Embedding

```python
# 이미지를 패치로 분할 후 선형 변환
# (B, 3, 224, 224) → (B, 196, 768)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

---

## 모델 변형

| 모델 | Layers | Hidden | Heads | Params |
|------|--------|--------|-------|--------|
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

---

## 학습

### 사전학습의 중요성

ViT는 CNN과 달리 inductive bias가 적어서 **대규모 데이터**가 필수:

| 학습 데이터 | ImageNet Accuracy |
|------------|-------------------|
| ImageNet-1k (1.2M) | 79.7% (ResNet보다 낮음) |
| ImageNet-21k (14M) | 84.2% |
| JFT-300M | **88.5%** |

### 학습 설정

- Optimizer: Adam (β₁=0.9, β₂=0.999)
- Learning rate: warmup + cosine decay
- Augmentation: RandAugment, Mixup, CutMix
- Regularization: Dropout, Stochastic Depth

---

## 구현 예시

```python
import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                      kernel_size=patch_size, stride=patch_size)

        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Classification (CLS token만 사용)
        return self.head(x[:, 0])
```

### HuggingFace 사용

```python
from transformers import ViTForImageClassification, ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
```

---

## ViT의 특징

### Attention 시각화

ViT는 이미지의 어떤 부분에 주목하는지 시각화 가능:

```python
# 마지막 레이어의 attention map
attentions = model.get_last_selfattention(x)
# CLS token이 어디를 보는지 확인
cls_attention = attentions[:, :, 0, 1:]  # (B, heads, num_patches)
```

### 장점

- 전역적 context 이해
- 확장성 우수 (모델/데이터 scale up 가능)
- 다양한 태스크 transfer 용이

### 단점

- 대규모 데이터 필요
- 계산량 O(n²)
- 작은 객체/고해상도 처리 어려움

---

## ViT 변형

| 모델 | 핵심 개선 |
|------|----------|
| **DeiT** | Knowledge distillation으로 적은 데이터 학습 |
| **Swin** | Shifted window로 효율적 계산 |
| **CvT** | Conv 추가로 inductive bias 강화 |
| **BEiT** | BERT 스타일 사전학습 |

---

## 관련 콘텐츠

- [Transformer](/ko/docs/architecture/transformer) - 기반 아키텍처
- [ResNet](/ko/docs/architecture/cnn/resnet) - 대조되는 CNN 접근
- [CLIP](/ko/docs/architecture/multimodal/clip) - ViT를 image encoder로 사용
- [DiT](/ko/docs/architecture/transformer/dit) - Diffusion + Transformer
