---
title: "RMSNorm"
weight: 3
math: true
---

# RMSNorm

## 개요

RMSNorm (Root Mean Square Normalization)은 Layer Norm의 간소화 버전으로, mean centering을 제거하여 계산 효율을 높입니다.

## Layer Norm vs RMSNorm

Layer Norm:
$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

RMSNorm:
$$
\hat{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

여기서:
$$
\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}
$$

## 차이점

- **Mean centering 제거**: μ 계산 불필요
- **Bias 제거**: β 파라미터 없음
- **15-20% 속도 향상**: 계산량 감소

## 구현

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS 계산
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 정규화
        x_norm = x / rms
        return self.weight * x_norm

# 사용
rms_norm = RMSNorm(dim=4096)
x = torch.randn(32, 2048, 4096)
y = rms_norm(x)
```

## 왜 효과적인가?

연구에 따르면:
- Layer Norm의 성공은 주로 **re-scaling**에서 옴
- Mean centering은 성능에 큰 영향 없음
- 제거해도 동등한 성능, 더 빠른 속도

## 사용 사례

최신 대형 언어 모델에서 표준:
- **LLaMA** (Meta)
- **Qwen** (Alibaba)
- **Mistral**
- **Gemma** (Google)

## 관련 콘텐츠

- [Layer Norm](/ko/docs/math/normalization/layer-norm)
- [Batch Norm](/ko/docs/math/normalization/batch-norm)
