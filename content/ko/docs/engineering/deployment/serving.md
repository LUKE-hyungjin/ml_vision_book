---
title: "모델 서빙"
weight: 3
---

# 모델 서빙

{{% hint info %}}
**선수지식**: [데이터 파이프라인](/ko/docs/engineering/data/pipeline) | [ONNX](/ko/docs/engineering/deployment/onnx)
{{% /hint %}}

## 한 줄 요약
> **모델 서빙은 학습된 모델을 안정적인 API로 배포해, 실제 사용자 요청을 빠르게 처리하는 운영 기술입니다.**

## 왜 필요한가?
모델을 잘 학습해도, 서비스에서 느리거나 자주 실패하면 실사용 가치가 떨어집니다.
서빙은 "좋은 모델"을 "쓸 수 있는 제품"으로 바꾸는 마지막 단계입니다.

초보자 관점에서 핵심 질문은 3가지입니다.
- 어떤 서버를 선택해야 하나? (FastAPI/Triton/TorchServe/BentoML)
- 속도와 안정성을 어떻게 같이 맞출까?
- 장애가 날 때 어디부터 점검할까?

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

## 실무 디버깅 체크리스트

- [ ] **입력 스키마 고정**: 클라이언트 요청 포맷(이미지 크기, 채널 순서, dtype)이 문서화되어 있는가?
- [ ] **전처리 일치**: 학습 때 쓴 normalize/resize가 서빙 코드와 동일한가?
- [ ] **배치 정책 확인**: 동적 배치가 지연시간 SLA를 깨지 않는가?
- [ ] **타임아웃/재시도 설정**: 업스트림 호출 실패 시 무한 대기하지 않는가?
- [ ] **모니터링 지표**: p50/p95 latency, error rate, GPU 메모리 사용량을 수집 중인가?

## 자주 하는 실수 (FAQ)

**Q1. FastAPI로 시작하면 나중에 무조건 다시 갈아타야 하나요?**  
A. 아닙니다. 초기에는 FastAPI로 빠르게 검증하고, 트래픽/모델 수가 커질 때 Triton/BentoML로 단계적 이전하는 방식이 일반적입니다.

**Q2. 모델 정확도가 좋은데 운영에서 성능이 나쁩니다. 왜 그럴까요?**  
A. 전처리 불일치, 배치 정책 미스매치, I/O 병목(디코딩/네트워크) 때문에 추론 성능이 크게 떨어질 수 있습니다.

**Q3. GPU가 있는데도 응답이 느립니다. 어디부터 봐야 하나요?**  
A. 먼저 (1) GPU utilization, (2) batch size, (3) CPU 전처리 병목, (4) worker 수를 순서대로 확인하세요.

## 초보자용 선택 가이드 (언제 무엇을 고를까?)

처음 서빙을 설계할 때는 "최고 성능"보다 **운영 가능한 복잡도**를 먼저 맞추는 편이 안전합니다.

1. **트래픽이 작고 모델 1~2개**
   - FastAPI로 시작
   - 목표: 배포 자동화 + 모니터링 지표(p95, error rate)부터 안정화

2. **모델 수가 늘거나 GPU 효율이 중요**
   - Triton/BentoML 검토
   - 목표: 동적 배치, 멀티 모델 운영, 버전 관리 체계화

3. **장애 대응 난이도가 급증**
   - 서버 교체보다 먼저 runbook(장애 대응 문서)과 알림 체계를 정비
   - 목표: "원인 미상 장애"를 줄이는 관측 가능성(observability) 확보

핵심 원칙:
- **작게 시작해서 계측하고, 병목이 확인된 뒤 확장**합니다.
- 도구 교체보다 "전처리 일치 + SLA + 모니터링"이 성능 개선에 더 큰 영향을 주는 경우가 많습니다.

## 지연시간(Latency) vs 처리량(Throughput) 빠른 감각

서빙 튜닝은 대부분 이 두 가지 균형 문제입니다.

| 선택 | 장점 | 리스크 |
|---|---|---|
| 배치 크기 증가 | 처리량(초당 요청 수) 상승 | 개별 요청 대기시간 증가 가능 |
| score/후처리 단순화 | 지연시간 감소 | 정확도/정밀한 후처리 성능 저하 가능 |
| worker 수 증가 | 동시성 향상 | 메모리 사용량 증가, 컨텍스트 스위칭 비용 |

초보자 운영 팁:
- 먼저 서비스 목표를 숫자로 고정하세요. 예: `p95 < 200ms`, `error rate < 1%`
- 목표 없이 튜닝하면, 체감 개선은 있는데 장애율이 오르는 역효과가 자주 발생합니다.

## 증상 → 원인 → 1차 조치

| 관측 증상 | 흔한 원인 | 1차 조치 |
|---|---|---|
| 응답이 간헐적으로 2~3초 이상 튐 | 동적 배치 timeout 과대, 입력 크기 편차 | batch timeout 축소, 입력 해상도 상한 도입 |
| GPU 사용률은 낮은데 CPU 100% | 이미지 decode/전처리 병목 | 전처리 워커 분리, resize/normalize 최적화 |
| 배포 후 정확도 급락 | 학습-서빙 전처리 불일치 | transform 파이프라인 diff 점검 |
| 트래픽 증가 시 5xx 급증 | timeout/retry/backpressure 부재 | 서버 타임아웃, 큐 제한, 재시도 정책 정비 |

## 5분 스모크 테스트 (배포 직후 필수)

운영 사고를 줄이려면 배포 직후 최소 검증을 자동화해야 합니다.
아래 스크립트는 **헬스체크 + 예측 API 지연시간**을 빠르게 확인하는 예시입니다.

```python
import time
import requests

BASE = "http://localhost:8000"

# 1) 헬스체크
r = requests.get(f"{BASE}/health", timeout=3)
print("health:", r.status_code, r.text[:80])

# 2) 예측 API 20회 지연시간 측정 (간단 p95 근사)
latencies = []
for _ in range(20):
    t0 = time.time()
    with open("sample.jpg", "rb") as f:
        resp = requests.post(f"{BASE}/predict", files={"file": f}, timeout=10)
    latencies.append((time.time() - t0) * 1000)
    if resp.status_code != 200:
        print("predict failed:", resp.status_code, resp.text[:120])

latencies.sort()
p50 = latencies[len(latencies)//2]
p95 = latencies[int(len(latencies)*0.95)-1]
print(f"p50={p50:.1f}ms, p95={p95:.1f}ms")
```

실무 기준 예시:
- `health` 200 응답
- `predict` 실패율 0%
- p95가 목표 SLA(예: 200ms) 이하

## 배포 전/후 운영 체크 (초보자용)

### 배포 전
- [ ] 모델 아티팩트 버전(파일 해시/태그) 고정
- [ ] 전처리 파이프라인 버전 고정 (`resize`, `normalize`, 색상 채널 순서)
- [ ] 롤백 대상(직전 안정 버전) 준비

### 배포 직후
- [ ] 스모크 테스트 실행 결과 기록 (p50/p95/error rate)
- [ ] 이전 버전 대비 정확도 샘플 비교(최소 20건)
- [ ] 알림(에러율/지연시간) 임계치가 실제로 동작하는지 확인

### 장애 발생 시
- [ ] 최근 변경점 1개씩 되돌리며 원인 분리 (모델/전처리/인프라)
- [ ] GPU/CPU 사용률과 큐 길이 동시 확인
- [ ] 원인 확정 전까지는 보수적 설정(작은 배치, 짧은 timeout)으로 임시 안정화

## 관련 콘텐츠

- [TensorRT](/ko/docs/engineering/deployment/tensorrt) - GPU 최적화
- [ONNX](/ko/docs/engineering/deployment/onnx) - 모델 변환
- [최적화](/ko/docs/engineering/deployment/optimization) - 모델 경량화

