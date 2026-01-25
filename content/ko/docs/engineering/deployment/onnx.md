---
title: "ONNX"
weight: 1
---

# ONNX (Open Neural Network Exchange)

## 개요

ONNX는 딥러닝 모델의 표준 교환 포맷입니다.

```
PyTorch    ─┐
TensorFlow ─┼──→ ONNX ──→ TensorRT / OpenVINO / CoreML / ...
Keras      ─┘
```

---

## PyTorch → ONNX 변환

### 기본 변환

```python
import torch
import torch.onnx

model = MyModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# 더미 입력 (모델 구조 추론용)
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX 변환
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=17,
)
```

### 주요 파라미터

| 파라미터 | 설명 |
|----------|------|
| `input_names` | 입력 텐서 이름 |
| `output_names` | 출력 텐서 이름 |
| `dynamic_axes` | 동적 차원 (배치 크기 등) |
| `opset_version` | ONNX 연산자 버전 |

---

## ONNX 검증

### 구조 확인

```python
import onnx

# 모델 로드 및 검증
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# 그래프 정보
print(onnx.helper.printable_graph(model.graph))
```

### Netron으로 시각화

```bash
pip install netron
netron model.onnx
```

---

## ONNX Runtime 추론

### 기본 추론

```python
import onnxruntime as ort
import numpy as np

# 세션 생성
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 입력 정보
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 추론
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
```

### 성능 최적화

```python
# 세션 옵션
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

# GPU 옵션
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
    }),
    'CPUExecutionProvider'
]

session = ort.InferenceSession("model.onnx", sess_options, providers=providers)
```

---

## 동적 입력 처리

### 가변 배치 크기

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    dynamic_axes={
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }
)
```

### 가변 이미지 크기

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch'}
    }
)
```

---

## 일반적인 문제 해결

### 지원되지 않는 연산자

```python
# 커스텀 연산자 등록
from torch.onnx import register_custom_op_symbolic

def my_custom_op(g, input):
    return g.op("CustomOp", input)

register_custom_op_symbolic('my_custom_op', my_custom_op, opset_version=11)
```

### 동적 제어 흐름

```python
# torch.jit.script 사용
@torch.jit.script
def my_function(x):
    if x.sum() > 0:
        return x * 2
    return x

# 또는 조건부 로직 제거
```

### 값 불일치

```python
# PyTorch와 ONNX Runtime 출력 비교
import numpy as np

torch_output = model(input_tensor).detach().numpy()
onnx_output = session.run([output_name], {input_name: input_data})[0]

np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-5)
```

---

## ONNX 모델 최적화

### onnxoptimizer

```python
import onnx
from onnxoptimizer import optimize

model = onnx.load("model.onnx")
optimized_model = optimize(model, [
    'eliminate_deadend',
    'eliminate_identity',
    'fuse_consecutive_transposes',
    'fuse_bn_into_conv',
])
onnx.save(optimized_model, "model_optimized.onnx")
```

### onnx-simplifier

```bash
pip install onnx-simplifier
onnxsim model.onnx model_simplified.onnx
```

```python
import onnx
from onnxsim import simplify

model = onnx.load("model.onnx")
simplified_model, check = simplify(model)
onnx.save(simplified_model, "model_simplified.onnx")
```

---

## 벤치마크

```python
import time
import numpy as np

# 워밍업
for _ in range(10):
    session.run([output_name], {input_name: input_data})

# 측정
n_runs = 100
start = time.time()
for _ in range(n_runs):
    session.run([output_name], {input_name: input_data})
elapsed = time.time() - start

print(f"Average latency: {elapsed / n_runs * 1000:.2f} ms")
print(f"Throughput: {n_runs / elapsed:.2f} FPS")
```

---

## 관련 콘텐츠

- [TensorRT](/ko/docs/engineering/deployment/tensorrt) - GPU 최적화
- [모델 서빙](/ko/docs/engineering/deployment/serving) - 추론 서버
- [최적화](/ko/docs/engineering/deployment/optimization) - 양자화

