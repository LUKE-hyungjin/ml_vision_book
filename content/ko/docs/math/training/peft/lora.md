---
title: "LoRA"
weight: 1
math: true
---

# LoRA (Low-Rank Adaptation)

## 개요

LoRA는 가중치 업데이트를 저랭크 행렬로 분해하여 학습 파라미터 수를 극적으로 줄입니다.

## 핵심 아이디어

기존 가중치 W를 수정하는 대신, 저랭크 업데이트를 더함:

$$
W' = W + \Delta W = W + BA
$$

- **W**: 원본 가중치 (고정, d × k)
- **B**: 저랭크 행렬 (d × r)
- **A**: 저랭크 행렬 (r × k)
- **r**: 랭크 (보통 4~64, r << min(d, k))

## 파라미터 절감

| | 파라미터 수 |
|---|------------|
| Full Fine-tuning | d × k |
| LoRA | d × r + r × k = r(d + k) |

예: d=4096, k=4096, r=8
- Full: 16,777,216
- LoRA: 65,536 (0.4%)

## 수식

순전파:
$$
h = Wx + BAx = Wx + B(Ax)
$$

LoRA 스케일링:
$$
h = Wx + \frac{\alpha}{r} BAx
$$

- α: 스케일링 팩터 (보통 r과 같은 값)

## 구현

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=8, alpha=8):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False  # 고정

        d, k = original_layer.weight.shape
        self.lora_A = nn.Parameter(torch.randn(r, k) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d, r))
        self.scale = alpha / r

    def forward(self, x):
        # 원본 + LoRA
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
```

## 어디에 적용?

Transformer에서 주로 적용하는 층:

```python
# 일반적인 설정
target_modules = [
    "q_proj", "v_proj",           # Attention (필수)
    "k_proj", "o_proj",           # Attention (선택)
    "gate_proj", "up_proj", "down_proj"  # FFN (선택)
]
```

## PEFT 라이브러리 사용

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || 0.06%
```

## 추론 시 병합

학습 후 원본 가중치에 병합하여 추론 속도 유지:

```python
# 병합
merged_weight = W + (alpha/r) * B @ A

# PEFT에서
model = model.merge_and_unload()
```

## 하이퍼파라미터 가이드

| 파라미터 | 일반적인 값 | 설명 |
|----------|-------------|------|
| r | 8~64 | 높을수록 표현력↑, 파라미터↑ |
| alpha | r 또는 2r | 학습률 스케일링 |
| dropout | 0.05~0.1 | LoRA 층에 적용 |
| target | q, v | 최소한 이 두 개 |

## 관련 콘텐츠

- [QLoRA](/ko/docs/math/training/peft/qlora) - LoRA + 양자화
- [SVD](/ko/docs/math/linear-algebra/svd) - 저랭크 분해 수학
- [Transformer](/ko/docs/architecture/transformer)
