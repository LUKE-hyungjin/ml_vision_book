---
title: "배포"
weight: 2
bookCollapseSection: true
---

# 모델 배포 (Deployment)

학습된 모델을 프로덕션 환경에 배포하는 기술입니다.

## 주요 내용

| 주제 | 설명 |
|------|------|
| [ONNX](/ko/docs/engineering/deployment/onnx) | 모델 포맷 변환 |
| [TensorRT](/ko/docs/engineering/deployment/tensorrt) | NVIDIA GPU 최적화 |
| [모델 서빙](/ko/docs/engineering/deployment/serving) | API 서버 구축 |
| [최적화](/ko/docs/engineering/deployment/optimization) | 양자화, 프루닝 |

---

## 배포 파이프라인

```
학습 (PyTorch) → 변환 (ONNX) → 최적화 (TensorRT) → 서빙 (Triton)
       ↓              ↓              ↓              ↓
    .pt 파일     .onnx 파일    .engine 파일    REST/gRPC API
```

---

## 고려사항

| 요소 | 질문 |
|------|------|
| **성능** | 필요한 처리량(TPS)? 지연시간(latency)? |
| **하드웨어** | GPU? CPU? 엣지? |
| **비용** | 클라우드 비용? 하드웨어 투자? |
| **확장성** | 트래픽 증가에 대응? |
| **운영** | 모니터링? 업데이트? A/B 테스트? |

