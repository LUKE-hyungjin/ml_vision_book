---
title: "PTQ"
weight: 1
math: true
---

# PTQ (Post-Training Quantization)

## 개요

PTQ는 이미 학습된 모델을 추가 학습 없이 양자화합니다. 빠르고 간단하지만 정확도 손실이 있을 수 있습니다.

## 양자화 수식

### 비대칭 양자화 (Asymmetric)

$$
Q(x) = \text{round}\left(\frac{x - x_{min}}{s}\right), \quad s = \frac{x_{max} - x_{min}}{2^b - 1}
$$

역양자화:
$$
\hat{x} = s \cdot Q(x) + x_{min}
$$

### 대칭 양자화 (Symmetric)

$$
Q(x) = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}
$$

```python
def symmetric_quantize(x, bits=8):
    """대칭 양자화"""
    qmax = 2 ** (bits - 1) - 1

    scale = x.abs().max() / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)

    return q.to(torch.int8), scale


def asymmetric_quantize(x, bits=8):
    """비대칭 양자화"""
    qmin, qmax = 0, 2 ** bits - 1

    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(-x_min / scale)

    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)

    return q.to(torch.uint8), scale, zero_point
```

## 캘리브레이션

최적의 스케일/영점 찾기:

```python
def calibrate(model, calibration_data, method='minmax'):
    """
    캘리브레이션 데이터로 양자화 파라미터 결정
    """
    stats = {}

    def hook(name):
        def fn(module, input, output):
            if name not in stats:
                stats[name] = {'min': [], 'max': []}
            stats[name]['min'].append(output.min().item())
            stats[name]['max'].append(output.max().item())
        return fn

    # Hook 등록
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook(name)))

    # 캘리브레이션 실행
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)

    # Hook 제거
    for h in handles:
        h.remove()

    # 스케일 계산
    scales = {}
    for name, s in stats.items():
        if method == 'minmax':
            scales[name] = {
                'min': min(s['min']),
                'max': max(s['max'])
            }
        elif method == 'percentile':
            # 이상치 제외
            scales[name] = {
                'min': np.percentile(s['min'], 1),
                'max': np.percentile(s['max'], 99)
            }

    return scales
```

## Granularity (세분화 수준)

| 수준 | 스케일 개수 | 정확도 | 효율성 |
|------|-------------|--------|--------|
| Per-tensor | 1 per tensor | 낮음 | 높음 |
| Per-channel | 1 per channel | 높음 | 중간 |
| Per-group | 1 per N elements | 매우 높음 | 낮음 |

```python
def per_channel_quantize(weight, bits=8):
    """채널별 양자화 (Conv, Linear)"""
    qmax = 2 ** (bits - 1) - 1

    # 각 출력 채널별 스케일
    if weight.dim() == 4:  # Conv: (out, in, h, w)
        scales = weight.abs().amax(dim=(1, 2, 3)) / qmax
        q = torch.round(weight / scales.view(-1, 1, 1, 1))
    else:  # Linear: (out, in)
        scales = weight.abs().amax(dim=1) / qmax
        q = torch.round(weight / scales.view(-1, 1))

    return q.clamp(-qmax, qmax).to(torch.int8), scales
```

## Weight-Only Quantization

가중치만 양자화 (LLM에서 인기):

```python
class WeightOnlyLinear(nn.Module):
    def __init__(self, weight, scale, bits=4):
        super().__init__()
        self.register_buffer('weight_q', weight)  # INT4
        self.register_buffer('scale', scale)
        self.bits = bits

    def forward(self, x):
        # 실시간 역양자화
        weight = self.weight_q.float() * self.scale
        return F.linear(x, weight)
```

## PyTorch 동적 양자화

```python
import torch.quantization as quant

# 동적 양자화 (추론 시 활성화 양자화)
model_dynamic = quant.quantize_dynamic(
    model,
    {nn.Linear},  # 양자화할 층
    dtype=torch.qint8
)

# 정적 양자화 (캘리브레이션 필요)
model.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model)

# 캘리브레이션
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

model_static = quant.convert(model_prepared)
```

## GPTQ (4-bit LLM)

레이어별 최적화:

```python
# AutoGPTQ 사용
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config={
        'bits': 4,
        'group_size': 128,
        'damp_percent': 0.1
    }
)

model.quantize(calibration_data)
```

## AWQ (Activation-aware Weight Quantization)

중요한 가중치 보호:

```python
# AWQ는 활성화 크기에 따라 스케일 조정
# 큰 활성화 = 중요한 가중치 = 더 정밀하게 양자화

from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("model_name")
model.quantize(tokenizer, quant_config={'w_bit': 4, 'q_group_size': 128})
```

## 관련 콘텐츠

- [QAT](/ko/docs/components/quantization/qat)
- [Data Types](/ko/docs/components/quantization/data-types)
- [QLoRA](/ko/docs/components/training/peft/qlora)
