---
title: "엣지 디바이스"
weight: 3
---

# 엣지 디바이스

## 개요

엣지에서 AI 추론을 수행하는 하드웨어입니다.

---

## 왜 엣지인가?

| 클라우드 | 엣지 |
|----------|------|
| 높은 지연시간 | 낮은 지연시간 |
| 네트워크 의존 | 오프라인 가능 |
| 대역폭 비용 | 로컬 처리 |
| 보안/프라이버시 우려 | 데이터 로컬 유지 |
| 무제한 컴퓨팅 | 제한된 리소스 |

---

## NVIDIA Jetson 시리즈

### 라인업

| 모델 | GPU | 메모리 | AI 성능 | 용도 |
|------|-----|--------|---------|------|
| **Orin NX 16GB** | Ampere 1024 CUDA | 16GB | 100 TOPS | 고성능 엣지 |
| **Orin NX 8GB** | Ampere 918 CUDA | 8GB | 70 TOPS | 상용 |
| **Orin Nano 8GB** | Ampere 1024 CUDA | 8GB | 40 TOPS | 입문/개발 |
| **Orin Nano 4GB** | Ampere 512 CUDA | 4GB | 20 TOPS | 저비용 |
| **Xavier NX** | Volta 384 CUDA | 8GB | 21 TOPS | 이전 세대 |
| **Nano (구형)** | Maxwell 128 CUDA | 4GB | 0.5 TFLOPS | 교육용 |

### JetPack 설치

```bash
# NVIDIA SDK Manager 사용 (호스트 PC에서)
# 또는 SD 카드 이미지 다운로드 후 플래싱
```

### 환경 설정

```bash
# 전력 모드 설정
sudo nvpmodel -m 0  # MAXN (최대 성능)
sudo nvpmodel -m 1  # 저전력

# 팬 속도
sudo jetson_clocks --fan

# 상태 모니터링
jtop  # tegrastats의 GUI 버전
```

---

## Jetson에서 모델 실행

### TensorRT 변환

```python
# ONNX → TensorRT
import tensorrt as trt

# trtexec 사용
# trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

### PyTorch 직접 실행

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval().cuda()

# FP16 추론
model.half()
input = torch.randn(1, 3, 224, 224).half().cuda()

with torch.no_grad():
    output = model(input)
```

### DeepStream

NVIDIA의 스트리밍 분석 SDK:

```
Camera → Decode → Inference → Tracking → Display/Save
         (NVDEC)  (TensorRT)  (NvTracker)
```

```bash
# 파이프라인 실행
deepstream-app -c config.txt
```

---

## 기타 엣지 디바이스

### Google Coral

| 모델 | AI 성능 | 인터페이스 |
|------|---------|------------|
| USB Accelerator | 4 TOPS | USB |
| Dev Board | 4 TOPS | 독립형 |
| Dev Board Mini | 4 TOPS | 소형 |

```python
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

interpreter = make_interpreter('model_edgetpu.tflite')
interpreter.allocate_tensors()

common.set_input(interpreter, image)
interpreter.invoke()
output = common.output_tensor(interpreter, 0)
```

### Intel Neural Compute Stick

```python
from openvino.runtime import Core

core = Core()
model = core.read_model("model.xml")
compiled = core.compile_model(model, "MYRIAD")

result = compiled([input_data])
```

### Raspberry Pi + Hailo

```
Raspberry Pi 5 + Hailo-8L M.2
→ 13 TOPS AI 성능
```

---

## 성능 최적화

### 배치 처리

```python
# 단일 추론보다 배치 처리가 효율적
batch_size = 4
inputs = torch.stack([preprocess(img) for img in images])
outputs = model(inputs)
```

### 파이프라인 병렬화

```python
from concurrent.futures import ThreadPoolExecutor

def pipeline():
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 병렬: 이미지 획득, 전처리, 추론
        futures = {
            'capture': executor.submit(capture_image),
            'preprocess': executor.submit(preprocess, prev_image),
            'infer': executor.submit(model, preprocessed),
        }
```

### 메모리 관리

```python
import gc
import torch

# 명시적 메모리 해제
del model
gc.collect()
torch.cuda.empty_cache()

# 메모리 모니터링
print(torch.cuda.memory_allocated() / 1024**2, "MB")
```

---

## 전력 관리

### Jetson 전력 모드

```bash
# 사용 가능한 모드 확인
sudo nvpmodel -q --verbose

# 예: Orin NX
# Mode 0: MAXN (최대 성능)
# Mode 1: 15W (저전력)
# Mode 2: 10W (초저전력)
```

### 동적 조절

```python
import subprocess

def set_power_mode(mode):
    subprocess.run(['sudo', 'nvpmodel', '-m', str(mode)])

# 작업량에 따라 전력 모드 조절
if workload_high:
    set_power_mode(0)  # MAXN
else:
    set_power_mode(1)  # 저전력
```

---

## 실제 배포

### Docker 사용

```dockerfile
# Jetson용 Dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

COPY model.engine /app/
COPY main.py /app/

WORKDIR /app
CMD ["python3", "main.py"]
```

```bash
docker run --runtime nvidia -it my-app
```

### 시스템 서비스

```bash
# /etc/systemd/system/vision-app.service
[Unit]
Description=Vision Application
After=network.target

[Service]
ExecStart=/usr/bin/python3 /app/main.py
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vision-app
sudo systemctl start vision-app
```

---

## 벤치마크

### Jetson Orin NX (FP16)

| 모델 | 입력 크기 | 지연시간 | FPS |
|------|-----------|----------|-----|
| ResNet-50 | 224×224 | 2.1 ms | 476 |
| YOLOv8n | 640×640 | 8.5 ms | 118 |
| YOLOv8s | 640×640 | 15.2 ms | 66 |
| YOLOv8m | 640×640 | 32.1 ms | 31 |

---

## 디바이스 선택 가이드

| 요구사항 | 추천 |
|----------|------|
| 최고 성능 | Jetson AGX Orin |
| 비용 효율 | Jetson Orin NX/Nano |
| 초저전력 | Coral, Hailo |
| 개발/프로토타입 | Jetson Orin Nano |
| 대량 생산 | 가격/지원 협상 필요 |

---

## 관련 콘텐츠

- [TensorRT](/ko/docs/engineering/deployment/tensorrt) - GPU 최적화
- [최적화](/ko/docs/engineering/deployment/optimization) - 모델 경량화
- [카메라](/ko/docs/engineering/hardware/camera) - 이미지 획득

