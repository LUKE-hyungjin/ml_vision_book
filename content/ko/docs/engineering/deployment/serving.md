---
title: "모델 서빙"
weight: 3
---

# 모델 서빙

## 개요

학습된 모델을 API로 제공하는 인프라 구축입니다.

---

## 서빙 옵션

| 도구 | 특징 | 적합한 경우 |
|------|------|-------------|
| **FastAPI** | 간단, Python 친화적 | 프로토타입, 소규모 |
| **Triton** | 고성능, 다중 모델 | 대규모 프로덕션 |
| **TorchServe** | PyTorch 통합 | PyTorch 중심 |
| **TensorFlow Serving** | TF 최적화 | TensorFlow 중심 |
| **BentoML** | ML 특화 | 빠른 배포 |

---

## FastAPI 서빙

### 기본 구조

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io

app = FastAPI()

# 모델 로드 (시작 시 한 번)
model = torch.load("model.pt")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 로드
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 전처리
    input_tensor = preprocess(image)

    # 추론
    with torch.no_grad():
        output = model(input_tensor)

    # 후처리
    result = postprocess(output)

    return {"prediction": result}

# 실행: uvicorn main:app --host 0.0.0.0 --port 8000
```

### 배치 처리

```python
from fastapi import BackgroundTasks
import asyncio
from collections import deque

batch_queue = deque()
batch_size = 32
batch_timeout = 0.1  # 100ms

async def batch_processor():
    while True:
        if len(batch_queue) >= batch_size or \
           (len(batch_queue) > 0 and time_since_first > batch_timeout):
            # 배치 처리
            batch = [batch_queue.popleft() for _ in range(min(len(batch_queue), batch_size))]
            inputs = torch.stack([item['input'] for item in batch])
            outputs = model(inputs)
            for item, output in zip(batch, outputs):
                item['future'].set_result(output)
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_processor())
```

---

## Triton Inference Server

### 모델 저장소

```
model_repository/
├── resnet/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── yolo/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

### config.pbtxt

```protobuf
name: "resnet"
platform: "onnxruntime_onnx"
max_batch_size: 32

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

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100000
}

instance_group [
  { count: 2, kind: KIND_GPU }
]
```

### 서버 실행

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### 클라이언트

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# 입력 준비
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# 추론
outputs = [httpclient.InferRequestedOutput("output")]
result = client.infer("resnet", inputs, outputs=outputs)

output_data = result.as_numpy("output")
```

---

## TorchServe

### 모델 아카이브 생성

```bash
# handler.py 작성
torch-model-archiver --model-name resnet \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pt \
    --handler image_classifier \
    --export-path model_store
```

### handler.py

```python
from ts.torch_handler.vision_handler import VisionHandler

class MyHandler(VisionHandler):
    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image))
            image = self.transform(image)
            images.append(image)
        return torch.stack(images)

    def postprocess(self, output):
        probs = torch.softmax(output, dim=1)
        return probs.tolist()
```

### 서버 실행

```bash
torchserve --start --model-store model_store --models resnet=resnet.mar
```

---

## BentoML

### 모델 저장

```python
import bentoml

# 모델 저장
bentoml.pytorch.save_model("resnet", model)
```

### 서비스 정의

```python
# service.py
import bentoml
from bentoml.io import Image, JSON

runner = bentoml.pytorch.get("resnet:latest").to_runner()
svc = bentoml.Service("image_classifier", runners=[runner])

@svc.api(input=Image(), output=JSON())
async def predict(image):
    input_tensor = preprocess(image)
    result = await runner.async_run(input_tensor)
    return postprocess(result)
```

### 빌드 및 배포

```bash
bentoml build
bentoml serve service:svc
```

---

## 성능 최적화

### 동시성

```python
# FastAPI with uvicorn workers
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

### GPU 메모리 관리

```python
# 모델별 GPU 할당
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 또는 Triton에서 instance_group 설정
```

### 모니터링

```python
from prometheus_client import Counter, Histogram, start_http_server

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')

@app.middleware("http")
async def add_metrics(request, call_next):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        response = await call_next(request)
    return response
```

---

## Docker 배포

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Kubernetes 배포

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    spec:
      containers:
      - name: model-server
        image: my-model-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: model-server
spec:
  selector:
    app: model-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 관련 콘텐츠

- [TensorRT](/ko/docs/engineering/deployment/tensorrt) - GPU 최적화
- [ONNX](/ko/docs/engineering/deployment/onnx) - 모델 변환
- [최적화](/ko/docs/engineering/deployment/optimization) - 모델 경량화

