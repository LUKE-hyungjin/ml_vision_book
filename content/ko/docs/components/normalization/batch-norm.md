---
title: "Batch Normalization"
weight: 1
math: true
---

# Batch Normalization

## 개요

Batch Normalization은 미니배치 단위로 활성화를 정규화하여 학습을 안정화합니다.

## 수식

입력 x에 대해 (B, C, H, W 텐서):

**1. 배치 통계 계산** (채널별):
$$
\mu_c = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} x_{b,c,h,w}
$$

$$
\sigma_c^2 = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} (x_{b,c,h,w} - \mu_c)^2
$$

**2. 정규화**:
$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

**3. 스케일/시프트** (학습 가능):
$$
y = \gamma \hat{x} + \beta
$$

## 구현

```python
import torch
import torch.nn as nn

# PyTorch BatchNorm
bn = nn.BatchNorm2d(num_features=64)  # 채널 수

x = torch.randn(32, 64, 224, 224)  # (B, C, H, W)
y = bn(x)

# 수동 구현
class ManualBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            # Running statistics 업데이트
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]
```

## Training vs Inference

**Training**:
- 현재 배치의 mean/var 사용
- Running mean/var 업데이트 (EMA)

**Inference**:
- 저장된 running mean/var 사용
- 배치 크기에 무관한 일관된 출력

```python
model.train()   # training mode
model.eval()    # inference mode - BatchNorm 동작 변경!
```

## 장점과 한계

**장점**:
- 학습 안정화
- 더 큰 학습률 사용 가능
- 약한 정규화 효과

**한계**:
- 배치 크기 의존: 작은 배치에서 불안정
- Training/Inference 불일치
- RNN/Transformer에 부적합

## BatchNorm 대안

| 상황 | 대안 |
|------|------|
| 작은 배치 | Group Norm |
| Transformer | Layer Norm |
| 온라인 학습 | Layer Norm |
| Style Transfer | Instance Norm |

## 관련 콘텐츠

- [Layer Norm](/ko/docs/components/normalization/layer-norm)
- [RMSNorm](/ko/docs/components/normalization/rms-norm)
- [Dropout](/ko/docs/components/training/regularization/dropout)
