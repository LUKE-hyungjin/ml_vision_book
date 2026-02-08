---
title: "Adapter"
weight: 3
math: true
---

# Adapter

## 개요

Adapter는 Transformer 블록 사이에 작은 모듈을 삽입하여 학습합니다. PEFT의 원조 기법입니다.

## 구조

```
        ┌─────────────┐
   x ───┤  Attention  ├─── + ───┬───
        └─────────────┘         │
                                ▼
                          ┌──────────┐
                          │ Adapter  │  ← 새로 추가
                          └──────────┘
                                │
        ┌─────────────┐         ▼
   x ───┤     FFN     ├─── + ───┬───
        └─────────────┘         │
                                ▼
                          ┌──────────┐
                          │ Adapter  │  ← 새로 추가
                          └──────────┘
```

## Adapter 모듈

Bottleneck 구조:

$$
\text{Adapter}(x) = x + f(xW_{down})W_{up}
$$

```python
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)

        # 초기화: 작은 값으로 시작
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))
```

## Transformer에 삽입

```python
class TransformerBlockWithAdapter(nn.Module):
    def __init__(self, dim, num_heads, bottleneck_dim=64):
        super().__init__()
        # 원본 층 (고정)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Adapter (학습)
        self.adapter1 = Adapter(dim, bottleneck_dim)
        self.adapter2 = Adapter(dim, bottleneck_dim)

    def forward(self, x):
        # Attention + Adapter
        attn_out = self.attention(x, x, x)[0]
        x = self.norm1(x + attn_out)
        x = self.adapter1(x)

        # FFN + Adapter
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        x = self.adapter2(x)

        return x
```

## 변형들

### AdapterFusion

여러 태스크의 어댑터를 동시에 사용:

```python
class AdapterFusion(nn.Module):
    def __init__(self, adapters, dim):
        super().__init__()
        self.adapters = nn.ModuleList(adapters)
        self.attention = nn.Linear(dim, len(adapters))

    def forward(self, x):
        # 각 어댑터 출력
        outputs = [adapter(x) for adapter in self.adapters]
        outputs = torch.stack(outputs, dim=-1)  # (B, L, D, num_adapters)

        # 어텐션으로 결합
        weights = F.softmax(self.attention(x), dim=-1)  # (B, L, num_adapters)
        return (outputs * weights.unsqueeze(-2)).sum(-1)
```

### Parallel Adapter

순차가 아닌 병렬 적용:

```python
def forward(self, x):
    # 병렬 적용
    return self.main_layer(x) + self.adapter(x)
```

## LoRA vs Adapter

| | LoRA | Adapter |
|---|------|---------|
| 위치 | 기존 가중치에 더함 | 층 사이에 삽입 |
| 추론 | 병합 가능 (속도 동일) | 추가 연산 필요 |
| 파라미터 | 더 적음 | 더 많음 |
| 유연성 | 제한적 | 다양한 변형 가능 |

## PEFT 라이브러리

```python
from peft import AdapterConfig, get_peft_model

config = AdapterConfig(
    adapter_size=64,
    adapter_act="gelu",
    adapter_dropout=0.1
)

model = get_peft_model(base_model, config)
```

## 관련 콘텐츠

- [LoRA](/ko/docs/components/training/peft/lora)
- [Transformer](/ko/docs/architecture/transformer)
- [Layer Normalization](/ko/docs/components/normalization/layer-norm)
