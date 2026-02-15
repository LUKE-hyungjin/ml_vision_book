---
title: "Model Serving"
weight: 3
---

# Model Serving

{{% hint info %}}
**Prerequisites**: [Data Pipeline](/en/docs/engineering/data/pipeline) | [ONNX](/en/docs/engineering/deployment/onnx)
{{% /hint %}}

## One-line Summary
> **Model serving is the engineering layer that turns a trained model into a reliable API for real user traffic.**

## Why is this needed?
Even a highly accurate model is not useful if inference is slow, unstable, or hard to operate.
Serving is the final step that converts a "good model" into a "usable product."

For beginners, three practical questions matter most:
- Which server should I start with? (FastAPI / Triton / TorchServe / BentoML)
- How do I balance latency and throughput?
- When production fails, where should I debug first?

## Overview
Model serving means packaging inference as an API with preprocessing, model execution, postprocessing, and monitoring.

---

## Serving options

| Tool | Strength | Best fit |
|------|----------|----------|
| **FastAPI** | Simple, Python-friendly | Prototype, small-scale |
| **Triton** | High-performance, multi-model | Large production |
| **TorchServe** | Tight PyTorch integration | PyTorch-focused teams |
| **TensorFlow Serving** | Optimized for TF | TensorFlow stacks |
| **BentoML** | ML-oriented deployment workflow | Fast productization |

---

## FastAPI serving

### Minimal structure

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io

app = FastAPI()

# Load model once at startup
model = torch.load("model.pt")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor)

    result = postprocess(output)
    return {"prediction": result}

# run: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Batching sketch

```python
from collections import deque
import asyncio

batch_queue = deque()
batch_size = 32
batch_timeout = 0.1  # 100 ms

async def batch_processor():
    while True:
        # pseudo logic: flush when enough requests or timeout is reached
        await asyncio.sleep(0.01)
```

---

## Triton Inference Server

### Model repository layout

```text
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

### Server run command

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

---

## Production debugging checklist

- [ ] **Fix request schema**: input format (image size, channel order, dtype) is explicitly defined.
- [ ] **Match preprocessing**: train-time normalize/resize exactly matches serving pipeline.
- [ ] **Validate batching policy**: dynamic batching does not violate latency SLA.
- [ ] **Set timeout/retry**: avoid indefinite waiting on upstream dependencies.
- [ ] **Track core metrics**: p50/p95 latency, error rate, GPU memory/utilization.

## Common mistakes (FAQ)

**Q1. If I start with FastAPI, do I always need a full migration later?**  
A. Not always. A common path is FastAPI for early validation, then gradual migration to Triton/BentoML when traffic or model count grows.

**Q2. My model is accurate offline, but production quality is poor. Why?**  
A. Typical causes are preprocessing mismatch, batching policy mismatch, and I/O bottlenecks (decode/network).

**Q3. I have GPUs but latency is still high. What should I inspect first?**  
A. Check in order: (1) GPU utilization, (2) batch size, (3) CPU preprocessing bottleneck, (4) worker/process count.

## Beginner decision guide (what to choose first)

When designing your first serving stack, optimize for **operable complexity** before chasing maximum throughput.

1. **Small traffic, 1-2 models**
   - Start with FastAPI
   - Goal: stabilize deployment automation + core metrics (p95, error rate)

2. **More models or stronger GPU efficiency needs**
   - Evaluate Triton or BentoML
   - Goal: dynamic batching, multi-model operation, cleaner versioning

3. **Operational incidents become frequent**
   - Before replacing the stack, improve runbooks and alerting
   - Goal: better observability and fewer "unknown-cause" incidents

Core principle:
- **Start small, instrument first, scale after bottlenecks are measured.**
- In many teams, preprocessing consistency + SLA control + monitoring improve outcomes more than tool migration.

## Quick intuition: Latency vs Throughput

Most serving tuning is a balance between these two.

| Choice | Benefit | Risk |
|---|---|---|
| Increase batch size | higher throughput (req/s) | higher per-request waiting time |
| Simplify score/postprocess | lower latency | possible drop in precision/postprocess quality |
| Increase workers | better concurrency | higher memory use, more context-switch overhead |

Beginner ops tip:
- Fix explicit service targets first, e.g. `p95 < 200ms`, `error rate < 1%`.
- Without numeric targets, local tuning often improves one metric while silently hurting reliability.

## Symptom → likely cause → first action

| Observed symptom | Common cause | First action |
|---|---|---|
| tail latency spikes (2-3s+) | oversized dynamic-batch timeout, variable input size | lower batch timeout, cap input resolution |
| low GPU utilization but CPU at 100% | decode/preprocess bottleneck | separate preprocess workers, optimize resize/normalize path |
| accuracy drops after deployment | train-serving preprocessing mismatch | diff the transform pipeline step by step |
| 5xx increases under traffic burst | missing timeout/retry/backpressure | add server timeout, bounded queue, retry policy |

## 5-minute smoke test (required right after deployment)

To reduce production incidents, automate a minimal post-deploy verification.
The script below quickly checks **health endpoint + prediction latency**.

```python
import time
import requests

BASE = "http://localhost:8000"

# 1) Health check
r = requests.get(f"{BASE}/health", timeout=3)
print("health:", r.status_code, r.text[:80])

# 2) Measure prediction latency for 20 requests (simple p95 approximation)
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

Example acceptance criteria:
- `health` returns 200
- `predict` error rate is 0%
- p95 is below your SLA target (e.g., 200ms)

## Pre/Post deployment ops checklist (beginner-friendly)

### Before deployment
- [ ] Pin model artifact version (hash/tag)
- [ ] Pin preprocessing version (`resize`, `normalize`, channel order)
- [ ] Prepare rollback target (last known good version)

### Right after deployment
- [ ] Run smoke test and record p50/p95/error rate
- [ ] Compare sample predictions with previous version (at least 20 samples)
- [ ] Verify alert thresholds (error rate/latency) are actually firing

### During incident
- [ ] Roll back one change at a time to isolate cause (model / preprocess / infra)
- [ ] Check GPU/CPU utilization together with queue length
- [ ] Until root cause is confirmed, stabilize with conservative settings (smaller batch, shorter timeout)

## Related Content

- [TensorRT](/en/docs/engineering/deployment/tensorrt) - GPU optimization
- [ONNX](/en/docs/engineering/deployment/onnx) - model conversion
- [Optimization](/en/docs/engineering/deployment/optimization) - deployment optimization
