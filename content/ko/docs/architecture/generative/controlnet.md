---
title: "ControlNet"
weight: 4
math: true
---

# ControlNet

## 개요

- **논문**: Adding Conditional Control to Text-to-Image Diffusion Models (2023)
- **저자**: Lvmin Zhang, Maneesh Agrawala (Stanford)
- **핵심 기여**: 사전학습된 Diffusion 모델에 공간적 조건 추가

## 핵심 아이디어

> "원본 모델을 고정하고, 복사본에서 조건을 학습"

Stable Diffusion의 가중치를 건드리지 않고, 에지/포즈/깊이 등의 조건을 추가할 수 있습니다.

---

## 왜 필요한가?

### 텍스트만으로는 부족한 제어

```
Prompt: "a person raising their hand"
→ 손 위치, 포즈를 정확히 지정 불가
```

### ControlNet의 해결

```
Prompt: "a person" + Pose Image (조건)
→ 원하는 포즈 그대로 생성
```

---

## 구조

### 전체 아키텍처

```
Input Image
      ↓
┌──────────────────┐
│ Condition Extractor │ (Canny, Pose, Depth 등)
└──────────────────┘
      ↓
Condition (edge/pose/depth map)
      ↓
┌───────────────────────────────────────────────┐
│                ControlNet                      │
│                                                │
│  ┌─────────────────┐    ┌─────────────────┐  │
│  │ Trainable Copy  │───→│   Zero Conv    │  │
│  │  of SD Encoder  │    │  (initialized 0)│  │
│  └─────────────────┘    └────────┬────────┘  │
│                                   ↓           │
│  ┌─────────────────────────────────────────┐ │
│  │        Locked Stable Diffusion          │ │
│  │     (original weights frozen)           │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
      ↓
Generated Image (following the condition)
```

### 핵심 컴포넌트

| 컴포넌트 | 설명 |
|----------|------|
| **Locked Copy** | 원본 SD 가중치 고정 |
| **Trainable Copy** | SD encoder의 학습 가능한 복사본 |
| **Zero Convolution** | 0으로 초기화된 1×1 Conv |

---

## Zero Convolution

학습 시작 시 ControlNet의 영향이 0:

$$y = x + \text{ZeroConv}(\text{ControlNet}(x, c))$$

**초기 상태:**
- ZeroConv 가중치 = 0
- ControlNet 출력이 0
- 원본 SD와 동일하게 동작

**학습 후:**
- 조건에 맞게 출력 조정

이렇게 하면 학습 초기에 원본 SD의 품질을 해치지 않습니다.

---

## 지원하는 조건들

| 조건 | 설명 | 용도 |
|------|------|------|
| **Canny Edge** | 윤곽선 | 구조 유지 |
| **OpenPose** | 인체 키포인트 | 포즈 제어 |
| **Depth** | 깊이 맵 | 3D 구조 |
| **Normal** | 노멀 맵 | 표면 방향 |
| **Scribble** | 낙서/스케치 | 대략적 형태 |
| **Segmentation** | 세그멘테이션 | 영역별 제어 |
| **M-LSD** | 직선 검출 | 건축물 등 |
| **SoftEdge** | 부드러운 에지 | HED, PiDi |

---

## 학습

### 데이터셋 구성

```
{
    "source": "condition_image.png",  # 조건 이미지
    "target": "ground_truth.png",      # 원본 이미지
    "prompt": "text description"
}
```

### 손실 함수

표준 Diffusion loss와 동일:

$$L = \mathbb{E}_{z, \epsilon, t, c, c_{control}} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c, c_{control}) \|^2 \right]$$

### 학습 전략

- 원본 SD 가중치 고정
- ControlNet 부분만 학습
- GPU 1개로도 학습 가능 (효율적)

---

## 구현 예시

### 기본 사용

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import cv2
import numpy as np

# ControlNet 로드 (Canny edge)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# 파이프라인 생성
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 조건 이미지 생성 (Canny edge)
image = Image.open("input.png")
image_np = np.array(image)
canny = cv2.Canny(image_np, 100, 200)
canny_image = Image.fromarray(canny)

# 생성
output = pipe(
    prompt="a beautiful garden",
    image=canny_image,
    num_inference_steps=30
).images[0]
```

### OpenPose 사용

```python
from controlnet_aux import OpenposeDetector

# 포즈 추출
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
pose = openpose(image)

# ControlNet으로 생성
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
# ... 이후 동일
```

### 다중 ControlNet

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# 여러 ControlNet 로드
controlnets = [
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny"),
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth"),
]

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnets,  # 리스트로 전달
    torch_dtype=torch.float16
)

# 다중 조건으로 생성
output = pipe(
    prompt="a house",
    image=[canny_image, depth_image],  # 각 조건 이미지
    controlnet_conditioning_scale=[1.0, 0.5]  # 각 조건의 강도
).images[0]
```

---

## 조건 강도 조절

```python
output = pipe(
    prompt="a landscape",
    image=canny_image,
    controlnet_conditioning_scale=0.5  # 0: 무시, 1: 완전 준수
).images[0]
```

낮은 값: 더 자유로운 생성
높은 값: 조건에 더 충실

---

## IP-Adapter와 결합

이미지 스타일 + 구조 제어:

```python
# IP-Adapter: 참조 이미지의 스타일
# ControlNet: 구조 조건

pipe.load_ip_adapter("h94/IP-Adapter", ...)
output = pipe(
    prompt="a portrait",
    image=canny_image,           # 구조
    ip_adapter_image=style_image  # 스타일
).images[0]
```

---

## ControlNet 변형

| 변형 | 특징 |
|------|------|
| **T2I-Adapter** | 더 가벼운 구조 |
| **ControlNet-XS** | 경량화 버전 |
| **ControlNet 1.1** | 개선된 학습 전략 |
| **ControlNet++** | SDXL 지원 |

---

## 관련 콘텐츠

- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - 기반 모델
- [Diffusion Process](/ko/docs/math/diffusion-process) - Diffusion 수학
- [U-Net](/ko/docs/architecture/segmentation/unet) - ControlNet 구조와 유사
- [Generation 태스크](/ko/docs/task/generation) - 이미지 생성 평가
