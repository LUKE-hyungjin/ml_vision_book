---
title: "PTQ"
weight: 1
math: true
---

# PTQ (Post-Training Quantization)

{{% hint info %}}
**선수지식**: [Data Types](/ko/docs/components/quantization/data-types)
{{% /hint %}}

## 한 줄 요약
> **PTQ는 추가 학습 없이 이미 학습된 모델을 INT8/INT4로 변환해, 추론 속도와 메모리를 개선하는 가장 빠른 양자화 방법입니다.**

## 왜 필요한가?
배포 단계에서는 "정확도 1~2%"보다 "지연 시간/메모리/비용"이 더 급한 경우가 많습니다.
PTQ는 재학습 없이 적용 가능해서 다음 상황에서 특히 유용합니다.

- 학습 데이터를 다시 쓰기 어렵거나 시간이 부족할 때
- GPU/엣지 메모리가 제한적일 때
- 먼저 빠른 베이스라인을 만들고, 필요하면 [QAT](/ko/docs/components/quantization/qat)로 넘어가고 싶을 때

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

**기호 설명(초보자 필수):**
- $x$: 원래 실수 텐서 값 (가중치/활성화)
- $Q(x)$: 정수 격자(INT8 등)로 옮긴 값
- $\hat{x}$: 다시 실수로 복원한 근사값
- $x_{min}, x_{max}$: 관측된 최소/최대값 (캘리브레이션에서 추정)
- $s$: scale (실수 1칸이 정수 몇 칸에 대응되는지)
- $b$: 비트 수 (예: INT8이면 $b=8$)

**언제 대칭/비대칭을 쓰나?**
- 대칭: 0 중심 분포(가중치)에 자주 사용, 구현 단순
- 비대칭: 0이 치우친 분포(활성화)에 유리한 경우가 많음

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

## 10분 PTQ 빠른 절차 (처음 하는 사람용)
1. **FP32 기준선 저장**: 정확도/지연시간/메모리 3가지를 먼저 기록
2. **캘리브레이션 샘플 준비**: 실제 배포 분포와 유사한 100~1000개 샘플
3. **INT8 PTQ 1차 적용**: per-tensor + minmax로 가장 단순하게 시작
4. **오차 큰 레이어 확인**: 초기 Conv/마지막 Head 중심으로 민감도 점검
5. **저비용 개선 적용**: per-channel 또는 percentile로 재측정

초보자 핵심 원칙: **한 번에 옵션 하나만 바꾸고, 항상 FP32 기준선과 비교**하세요.

## 미니 실험: 양자화 오차를 숫자로 확인하기 (MSE)
아래 코드는 같은 텐서를 FP32/INT8로 왕복시켜 보고, 양자화 오차가 어느 정도인지 빠르게 확인합니다.

```python
import torch


def fake_ptq_roundtrip(x: torch.Tensor, bits: int = 8):
    # 대칭 per-tensor 양자화
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max().clamp(min=1e-8) / qmax

    x_int = torch.round(x / scale).clamp(-qmax, qmax)
    x_deq = x_int * scale

    mse = torch.mean((x - x_deq) ** 2).item()
    max_abs_err = torch.max(torch.abs(x - x_deq)).item()
    return mse, max_abs_err


x = torch.randn(4096) * 0.5  # 예시 activation 분포
mse, max_abs_err = fake_ptq_roundtrip(x, bits=8)
print(f"MSE={mse:.6e}, max_abs_err={max_abs_err:.6e}")
```

해석 가이드:
- MSE가 작고(`~1e-6`~`1e-4` 수준), max error가 허용 범위면 PTQ 후보로 괜찮습니다.
- 레이어별로 이 값을 비교하면 **어떤 블록이 양자화에 민감한지** 빠르게 찾을 수 있습니다.
- 민감 레이어는 per-channel 적용 또는 FP16 유지(부분 양자화)로 완화합니다.

## 증상 → 원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 할 조치 |
|---|---|---|
| INT8 변환 직후 정확도 급락 | 캘리브레이션 분포 불일치 | 캘리브레이션 샘플 재구성(조명/해상도/클래스 비율 반영) |
| 특정 클래스만 유독 성능 저하 | 클래스 편향된 캘리브레이션 | 클래스 균형 샘플로 재캘리브레이션 |
| 속도 개선이 기대보다 작음 | FP32 fallback 연산 다수 | 백엔드 연산자 지원표 확인 + unsupported op 추적 |
| 박스 좌표/로짓이 불안정 | activation 양자화 과도 | 해당 블록만 FP16/FP32 유지(부분 양자화) |

## 실무 디버깅 체크리스트
- [ ] 캘리브레이션 데이터가 실제 배포 입력 분포(밝기, 해상도, 클래스 비율)와 유사한가?
- [ ] 성능 급락 레이어가 특정 블록(초기 Conv, 마지막 Head)에 몰려 있는가?
- [ ] Per-tensor에서 성능이 낮다면 Per-channel로 바꿔 비교했는가?
- [ ] 연산자별 지원 여부(TensorRT/ONNX Runtime) 때문에 FP32 fallback이 발생하지 않는가?
- [ ] 속도 측정 시 warm-up 이후 지연 시간을 비교했는가?

## 자주 하는 실수 (FAQ)
**Q1. PTQ를 하면 정확도가 항상 조금만 떨어지나요?**  
A. 아닙니다. 모델 구조와 데이터 분포에 따라 큰 하락이 날 수 있습니다. 특히 activation 분포가 긴 꼬리를 가지면 min-max만으로는 손실이 커질 수 있습니다.

**Q2. 데이터가 없어도 PTQ 가능한가요?**  
A. 기술적으로 가능하지만 권장되지 않습니다. 최소한 소량의 대표 캘리브레이션 데이터는 있어야 scale/zero-point가 현실적인 값으로 잡힙니다.

**Q3. PTQ와 QAT는 어떤 순서로 적용하나요?**  
A. 보통 PTQ로 빠르게 baseline을 본 뒤, 정확도 손실이 허용 범위를 넘으면 QAT로 넘어갑니다.

## 다음 단계 가이드
1. PTQ baseline 측정 (정확도/지연시간/메모리)
2. Per-channel, percentile calibration 등 저비용 개선 적용
3. 여전히 손실이 크면 [QAT](/ko/docs/components/quantization/qat)로 전환

## 관련 콘텐츠

- [QAT](/ko/docs/components/quantization/qat)
- [Data Types](/ko/docs/components/quantization/data-types)
- [QLoRA](/ko/docs/components/training/peft/qlora)
