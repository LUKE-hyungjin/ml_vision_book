---
title: "TensorRT"
weight: 2
---

# TensorRT

## 개요

TensorRT는 NVIDIA GPU에서 딥러닝 추론을 최적화하는 라이브러리입니다.

```
ONNX 모델 → TensorRT 최적화 → 최대 10x 빠른 추론
```

---

## 주요 최적화

| 기법 | 설명 |
|------|------|
| Layer Fusion | 여러 레이어를 하나로 병합 |
| Precision Calibration | FP32 → FP16/INT8 변환 |
| Kernel Auto-tuning | 최적의 CUDA 커널 선택 |
| Dynamic Tensor Memory | 메모리 재사용 |
| Multi-stream Execution | 병렬 처리 |

---

## ONNX → TensorRT 변환

### trtexec (CLI)

```bash
# FP16 엔진 생성
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16

# INT8 엔진 생성 (캘리브레이션 필요)
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --int8 \
        --calib=calibration_cache.bin

# 동적 배치
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:8x3x224x224 \
        --maxShapes=input:16x3x224x224
```

### Python API

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# ONNX 파싱
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 빌더 설정
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # FP16 활성화

# 엔진 빌드
serialized_engine = builder.build_serialized_network(network, config)

# 저장
with open("model.engine", "wb") as f:
    f.write(serialized_engine)
```

---

## TensorRT 추론

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# 엔진 로드
with open("model.engine", "rb") as f:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 메모리 할당
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

d_input = cuda.mem_alloc(np.prod(input_shape) * 4)  # float32
d_output = cuda.mem_alloc(np.prod(output_shape) * 4)

# 추론
def infer(input_data):
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2([int(d_input), int(d_output)])
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    return output

# 사용
input_data = np.random.randn(*input_shape).astype(np.float32)
result = infer(input_data)
```

---

## INT8 양자화

### 캘리브레이션

```python
import tensorrt as trt
import numpy as np

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file):
        super().__init__()
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.batch_idx = 0

        # 디바이스 메모리 할당
        self.d_input = cuda.mem_alloc(self.data_loader.batch_size * 3 * 224 * 224 * 4)

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.data_loader)
            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# 빌더 설정
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = Calibrator(calibration_loader, "calibration.cache")
```

---

## 동적 Shape

```python
# 프로파일 설정
profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 3, 224, 224),   # 최소
    opt=(8, 3, 224, 224),   # 최적
    max=(32, 3, 224, 224)   # 최대
)
config.add_optimization_profile(profile)

# 추론 시 shape 설정
context.set_input_shape("input", (batch_size, 3, 224, 224))
```

---

## torch2trt (간편한 변환)

```python
from torch2trt import torch2trt
import torch

model = MyModel().cuda().eval()
x = torch.randn(1, 3, 224, 224).cuda()

# TensorRT 변환
model_trt = torch2trt(
    model,
    [x],
    fp16_mode=True,
    max_batch_size=16,
)

# 추론
y_trt = model_trt(x)

# 저장/로드
torch.save(model_trt.state_dict(), "model_trt.pth")
```

---

## Triton과 통합

### 모델 저장소 구조

```
model_repository/
└── my_model/
    ├── config.pbtxt
    └── 1/
        └── model.plan  # TensorRT 엔진
```

### config.pbtxt

```protobuf
name: "my_model"
platform: "tensorrt_plan"
max_batch_size: 16

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```

---

## 성능 비교

### 벤치마크 예시 (ResNet-50, RTX 3090)

| Precision | Latency | Throughput |
|-----------|---------|------------|
| PyTorch FP32 | 5.2 ms | 192 FPS |
| TensorRT FP32 | 2.1 ms | 476 FPS |
| TensorRT FP16 | 1.0 ms | 1000 FPS |
| TensorRT INT8 | 0.6 ms | 1667 FPS |

---

## 문제 해결

### 지원되지 않는 연산

```python
# 플러그인 사용
# 또는 ONNX 레벨에서 수정

# 예: 지원되지 않는 레이어를 여러 지원 레이어로 분해
```

### 정확도 저하 (INT8)

```python
# 1. 캘리브레이션 데이터 확인 (대표성 있는 데이터 사용)
# 2. 민감한 레이어는 FP16 유지
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
```

### 메모리 부족

```python
# 워크스페이스 크기 조정
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB
```

---

## 관련 콘텐츠

- [ONNX](/ko/docs/engineering/deployment/onnx) - 모델 변환
- [모델 서빙](/ko/docs/engineering/deployment/serving) - Triton 서버
- [최적화](/ko/docs/engineering/deployment/optimization) - 양자화 이론

