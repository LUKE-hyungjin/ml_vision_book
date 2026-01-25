---
title: "Data Types"
weight: 3
math: true
---

# 데이터 타입

딥러닝에서 사용되는 다양한 숫자 표현 형식입니다.

## 부동소수점 (Floating Point)

### FP32 (Single Precision)

```
| 1 bit | 8 bits  | 23 bits    |
| sign  | exponent| mantissa   |
```

- 범위: ±3.4 × 10³⁸
- 정밀도: ~7자리
- 기본 학습 정밀도

### FP16 (Half Precision)

```
| 1 bit | 5 bits  | 10 bits    |
| sign  | exponent| mantissa   |
```

- 범위: ±65504
- 정밀도: ~3자리
- 메모리 절반, 속도 2배

### BF16 (Brain Float)

```
| 1 bit | 8 bits  | 7 bits     |
| sign  | exponent| mantissa   |
```

- 범위: FP32와 동일
- 정밀도: ~2자리
- 학습에 더 안정적 (넓은 범위)

```python
import torch

# 타입 변환
x_fp32 = torch.randn(100, 100)
x_fp16 = x_fp32.half()       # FP16
x_bf16 = x_fp32.bfloat16()   # BF16

# 메모리 비교
print(f"FP32: {x_fp32.element_size()} bytes")  # 4
print(f"FP16: {x_fp16.element_size()} bytes")  # 2
print(f"BF16: {x_bf16.element_size()} bytes")  # 2
```

## 정수 (Integer)

### INT8

```
| 1 bit | 7 bits  |
| sign  | value   |
```

- 범위: -128 ~ 127 (signed) 또는 0 ~ 255 (unsigned)
- PTQ의 기본 형식

### INT4

```
| 4 bits |
| value  |
```

- 범위: -8 ~ 7 (signed) 또는 0 ~ 15 (unsigned)
- LLM 양자화에서 인기

## 특수 포맷

### FP8 (E4M3, E5M2)

```
E4M3: | 1 | 4 | 3 |  # sign, exp, mantissa
E5M2: | 1 | 5 | 2 |
```

- 최신 GPU 지원 (H100)
- E4M3: 더 높은 정밀도
- E5M2: 더 넓은 범위

### NF4 (4-bit NormalFloat)

QLoRA에서 사용:

```python
# 정규분포에 최적화된 비균등 양자값
nf4_quantiles = [
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0
]
```

## Mixed Precision Training

FP16/BF16과 FP32 혼합:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    # FP16으로 순전파
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # 스케일된 역전파
    scaler.scale(loss).backward()

    # 언스케일 후 업데이트
    scaler.step(optimizer)
    scaler.update()
```

### Loss Scaling

FP16에서 작은 그래디언트 보호:

```python
# 문제: FP16 최소값 = 6e-5
# 작은 그래디언트가 0이 됨

# 해결: Loss를 크게 스케일 → 그래디언트도 크게
scaled_loss = loss * 1024  # 스케일 업
scaled_loss.backward()

for param in model.parameters():
    param.grad /= 1024  # 스케일 다운
```

## 비교

| 타입 | 비트 | 범위 | 정밀도 | 용도 |
|------|------|------|--------|------|
| FP32 | 32 | ±3.4e38 | 높음 | 기본 |
| FP16 | 16 | ±65504 | 중간 | 추론 |
| BF16 | 16 | ±3.4e38 | 낮음 | 학습 |
| INT8 | 8 | -128~127 | - | PTQ |
| INT4 | 4 | -8~7 | - | LLM |
| FP8 | 8 | 다양 | 낮음 | 최신 |

## PyTorch 타입 선택

```python
# 학습
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# 추론
model = model.half()  # FP16

# 양자화 추론
model_int8 = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

## 하드웨어 지원

| 타입 | NVIDIA | AMD | Apple |
|------|--------|-----|-------|
| FP16 | Tensor Core | MI | ANE |
| BF16 | A100+ | MI200+ | M1+ |
| INT8 | Tensor Core | MI | ANE |
| FP8 | H100+ | MI300+ | - |

## 관련 콘텐츠

- [PTQ](/ko/docs/math/quantization/ptq)
- [QAT](/ko/docs/math/quantization/qat)
- [QLoRA](/ko/docs/math/training/peft/qlora)
