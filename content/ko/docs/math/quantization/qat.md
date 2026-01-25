---
title: "QAT"
weight: 2
math: true
---

# QAT (Quantization-Aware Training)

## 개요

QAT는 학습 중에 양자화 효과를 시뮬레이션하여 양자화에 강건한 모델을 만듭니다.

## 핵심 아이디어

```
순전파: 양자화 시뮬레이션 (fake quantization)
역전파: 실수 그래디언트 (STE)
결과: 양자화에 적응된 가중치
```

## Fake Quantization

양자화/역양자화를 한 번에:

$$
\text{FakeQuant}(x) = s \cdot \text{clip}\left(\text{round}\left(\frac{x}{s}\right), q_{min}, q_{max}\right)
$$

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        # 양자화 후 역양자화
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, qmin, qmax)
        x_dq = (x_q - zero_point) * scale

        # 역전파를 위해 마스크 저장
        ctx.save_for_backward(x, scale)
        ctx.qmin, ctx.qmax = qmin, qmax

        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors

        # Straight-Through Estimator
        # 범위 내 값은 그래디언트 통과, 범위 밖은 0
        x_q = x / scale
        mask = (x_q >= ctx.qmin) & (x_q <= ctx.qmax)

        grad_input = grad_output * mask.float()

        return grad_input, None, None, None, None
```

## PyTorch QAT

```python
import torch.quantization as quant

# 1. 모델에 qconfig 설정
model.qconfig = quant.get_default_qat_qconfig('fbgemm')

# 2. QAT 준비 (fake quant 삽입)
model_prepared = quant.prepare_qat(model, inplace=False)

# 3. 학습
optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-4)

for epoch in range(epochs):
    model_prepared.train()
    for batch in train_loader:
        output = model_prepared(batch)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Freeze BN 통계 (마지막 몇 에폭)
    if epoch >= epochs - 3:
        model_prepared.apply(quant.disable_observer)
        model_prepared.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

# 4. 최종 양자화
model_quantized = quant.convert(model_prepared.eval())
```

## 커스텀 QAT 모듈

```python
class QATLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits

        # 학습 가능한 스케일
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.input_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 가중치 fake quantize
        qmax = 2 ** (self.bits - 1) - 1
        w_scale = self.linear.weight.abs().max() / qmax * self.weight_scale

        w_q = fake_quantize(self.linear.weight, w_scale, 0, -qmax, qmax)

        # 입력 fake quantize (선택적)
        x_scale = x.abs().max() / qmax * self.input_scale
        x_q = fake_quantize(x, x_scale, 0, -qmax, qmax)

        return F.linear(x_q, w_q, self.linear.bias)
```

## LSQ (Learned Step Size Quantization)

스케일도 학습:

$$
\frac{\partial L}{\partial s} = \frac{\partial L}{\partial \bar{x}} \cdot \frac{\partial \bar{x}}{\partial s}
$$

```python
class LSQ(nn.Module):
    def __init__(self, bits=8, all_positive=False):
        super().__init__()
        self.bits = bits
        self.all_positive = all_positive

        # 학습 가능한 step size
        self.step_size = nn.Parameter(torch.ones(1))

        if all_positive:
            self.qmin, self.qmax = 0, 2**bits - 1
        else:
            self.qmin, self.qmax = -(2**(bits-1)), 2**(bits-1) - 1

    def forward(self, x):
        # 그래디언트 스케일링
        grad_scale = 1.0 / np.sqrt(x.numel() * self.qmax)

        step = self.step_size * grad_scale + self.step_size.detach() * (1 - grad_scale)

        x_q = torch.round(x / step).clamp(self.qmin, self.qmax)
        return x_q * step
```

## Mixed Precision QAT

층마다 다른 비트:

```python
class MixedPrecisionQAT(nn.Module):
    def __init__(self, model, bit_config):
        """
        bit_config: {layer_name: bits}
        예: {'conv1': 8, 'fc': 4}
        """
        super().__init__()
        self.model = model

        for name, module in model.named_modules():
            if name in bit_config:
                bits = bit_config[name]
                self.wrap_with_qat(module, bits)

    def wrap_with_qat(self, module, bits):
        # 모듈에 fake quantization 추가
        module.register_forward_hook(
            lambda m, i, o: fake_quantize(o, compute_scale(o), 0, -(2**(bits-1)), 2**(bits-1)-1)
        )
```

## 학습 팁

1. **Learning Rate**: PTQ 모델로 시작하면 작은 LR (1e-5)
2. **Epochs**: 원래 학습의 10-20% 정도
3. **BN Freeze**: 마지막 에폭에서 BN 통계 고정
4. **Warmup**: 처음에는 FP32로 몇 스텝 학습

```python
# 점진적 양자화
for epoch in range(epochs):
    if epoch < warmup_epochs:
        # FP32 학습
        model.apply(quant.disable_fake_quant)
    else:
        # QAT 활성화
        model.apply(quant.enable_fake_quant)

    train_one_epoch(model, train_loader, optimizer)
```

## PTQ vs QAT

| | PTQ | QAT |
|---|-----|-----|
| 학습 필요 | X | O |
| 정확도 | 낮음~중간 | 높음 |
| 시간 | 빠름 | 느림 |
| 4-bit | 어려움 | 가능 |

## 관련 콘텐츠

- [PTQ](/ko/docs/math/quantization/ptq)
- [Data Types](/ko/docs/math/quantization/data-types)
- [Backpropagation](/ko/docs/math/calculus/backpropagation)
