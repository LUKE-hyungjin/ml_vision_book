---
title: "모델 최적화"
weight: 4
---

# 모델 최적화

## 개요

모델 크기와 추론 속도를 개선하는 기법들입니다.

---

## 최적화 기법 분류

| 기법 | 설명 | 정확도 영향 |
|------|------|-------------|
| **양자화** | 비트 수 줄임 (FP32→INT8) | 약간 감소 |
| **프루닝** | 불필요한 가중치 제거 | 약간 감소 |
| **Knowledge Distillation** | 작은 모델에 지식 전달 | 가능한 한 유지 |
| **아키텍처 최적화** | 효율적 구조 설계 | 설계에 따라 다름 |

---

## 양자화 (Quantization)

### 종류

| 방식 | 설명 | 특징 |
|------|------|------|
| **PTQ** | Post-Training Quantization | 빠름, 약간의 정확도 손실 |
| **QAT** | Quantization-Aware Training | 느림, 정확도 유지 |
| **Dynamic** | 추론 시 동적 양자화 | 간단, 제한적 성능 향상 |

### PyTorch 동적 양자화

```python
import torch.quantization

# 동적 양자화 (주로 Linear, LSTM에 효과적)
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### PyTorch 정적 양자화 (PTQ)

```python
import torch.quantization

model.eval()

# 양자화 설정
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 퓨전 (Conv + BN + ReLU)
model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

# 준비
model_prepared = torch.quantization.prepare(model_fused)

# 캘리브레이션 (대표 데이터로)
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 변환
model_quantized = torch.quantization.convert(model_prepared)
```

### QAT (Quantization-Aware Training)

```python
# 준비
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# 학습 (fake quantization 포함)
for epoch in range(epochs):
    for data, target in train_loader:
        output = model_prepared(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 변환
model_prepared.eval()
model_quantized = torch.quantization.convert(model_prepared)
```

---

## 프루닝 (Pruning)

### 비구조적 프루닝

개별 가중치 제거:

```python
import torch.nn.utils.prune as prune

# L1 프루닝 (30% 제거)
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# 영구 적용
prune.remove(model.conv1, 'weight')
```

### 구조적 프루닝

전체 채널/필터 제거:

```python
# 채널 프루닝
prune.ln_structured(model.conv1, name='weight', amount=0.3, n=2, dim=0)
```

### 글로벌 프루닝

```python
parameters_to_prune = [
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc, 'weight'),
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
```

---

## Knowledge Distillation

### 기본 구조

```
Teacher (큰 모델) ──→ Soft Labels
                          ↓
                    Student (작은 모델)
                          ↑
Ground Truth ────────→ Hard Labels
```

### 구현

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    T: temperature (높을수록 soft한 분포)
    alpha: soft loss 비중
    """
    # Soft loss (teacher의 지식)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    # Hard loss (실제 레이블)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss

# 학습
teacher.eval()
student.train()

for data, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(data)

    student_logits = student(data)
    loss = distillation_loss(student_logits, teacher_logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 효율적인 아키텍처

### MobileNet

Depthwise Separable Convolution:

```python
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise: 채널별 독립 연산
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch)
        # Pointwise: 1x1 conv
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

### EfficientNet

Compound Scaling:

```
depth = α^φ
width = β^φ
resolution = γ^φ

여기서 α · β² · γ² ≈ 2
```

---

## 혼합 정밀도 학습

학습 시 FP16 사용으로 메모리와 속도 개선:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast():  # FP16 연산
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 최적화 파이프라인

```
1. 베이스라인 측정
   ↓
2. 프루닝 (30-50%)
   ↓
3. 재학습 (정확도 복구)
   ↓
4. 양자화 (INT8)
   ↓
5. ONNX 변환
   ↓
6. TensorRT 최적화
   ↓
7. 최종 벤치마크
```

---

## 성능 측정

```python
import torch
import time

def benchmark(model, input_shape, n_runs=100, warmup=10):
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)

    # 워밍업
    for _ in range(warmup):
        with torch.no_grad():
            model(x)

    # 동기화 (GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 측정
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    elapsed = time.time() - start

    latency = elapsed / n_runs * 1000
    throughput = n_runs / elapsed

    print(f"Latency: {latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")

    return latency, throughput
```

### 모델 크기 측정

```python
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
```

---

## 관련 콘텐츠

- [TensorRT](/ko/docs/engineering/deployment/tensorrt) - GPU 최적화
- [ONNX](/ko/docs/engineering/deployment/onnx) - 모델 변환
- [모델 서빙](/ko/docs/engineering/deployment/serving) - 배포

