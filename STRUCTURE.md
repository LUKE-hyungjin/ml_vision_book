# Vision Engineer 지식 가이드 - 콘텐츠 구조

## 개요

3가지 입구는 **서로 연결되지 않고**, 오직 공유 콘텐츠로만 연결됩니다.
입구는 "순서/방식"만 제공하고, 실제 내용은 공유 콘텐츠에만 존재합니다.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  타임라인    │     │  Top-Down   │     │  Bottom-Up  │
│  (순서 제공)  │     │  (순서 제공)  │     │  (순서 제공)  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│                    공유 콘텐츠                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │  수학   │ │아키텍처│ │ 태스크  │ │  기타   │        │
│  └────────┘ └────────┘ └────────┘ └────────┘        │
└──────────────────────────────────────────────────────┘
```

**핵심 원칙:**
- 입구 → 콘텐츠: O (링크 허용)
- 콘텐츠 → 콘텐츠: O (링크 허용)
- 입구 → 입구: X (링크 금지)
- 콘텐츠 → 입구: X (링크 금지)

---

## 공유 콘텐츠 분류 기준

| 카테고리 | 기준 | 예시 |
|----------|------|------|
| **math/** | 수식이 핵심인 개념 | 선형대수, Convolution 연산, Loss 함수, Attention 수식 |
| **architecture/** | 특정 모델/네트워크 구조 | ResNet, YOLO, ViT, Diffusion |
| **task/** | 문제 정의 + 평가지표 + 데이터셋 | Classification, Detection, Segmentation |
| **engineering/** | 실무 기술 | 데이터 증강, 모델 배포, 하드웨어 |
| **etc/** | 외부 링크/자료 모음 | 논문, 강의, 도서, 도구 |

### 애매한 경우 판단 기준

- "Convolution이 뭐야?" → **math/** (수식/연산 설명)
- "ResNet 구조가 뭐야?" → **architecture/** (모델 구조)
- "Detection은 어떻게 평가해?" → **task/** (mAP, IoU 등 평가)
- "모델 배포는 어떻게 해?" → **engineering/** (TensorRT, ONNX 등)
- "Detection 관련 좋은 논문?" → **etc/** (외부 자료)

---

## 1. 타임라인 페이지

> 연대순 "순서"만 제공. 각 시대별로 어떤 콘텐츠를 봐야 하는지 링크.

```markdown
# 타임라인

연대순으로 Vision 기술의 발전을 따라갑니다.

## ~2012: Classical Computer Vision
- [선형대수](/docs/math/linear-algebra)
- [기하학](/docs/math/geometry)
- [SIFT & HOG](/docs/architecture/classical/sift-hog)

## 2012-2015: CNN 시대의 시작
- [Convolution](/docs/math/convolution)
- [Backpropagation](/docs/math/backpropagation)
- [CNN 기초](/docs/architecture/cnn)
- [AlexNet](/docs/architecture/cnn/alexnet)
- [VGG](/docs/architecture/cnn/vgg)
- [ResNet](/docs/architecture/cnn/resnet)
- [Classification](/docs/task/classification)

## 2015-2017: Detection & Segmentation
- [IoU & NMS](/docs/math/iou-nms)
- [Faster R-CNN](/docs/architecture/detection/faster-rcnn)
- [YOLO](/docs/architecture/detection/yolo)
- [U-Net](/docs/architecture/segmentation/unet)
- [Detection](/docs/task/detection)
- [Segmentation](/docs/task/segmentation)

## 2017-2019: Attention의 등장
- [Attention](/docs/math/attention)
- [Transformer](/docs/architecture/transformer)

## 2020-2021: Vision Transformer & CLIP
- [ViT](/docs/architecture/transformer/vit)
- [CLIP](/docs/architecture/multimodal/clip)
- [Contrastive Learning](/docs/math/contrastive)
- [Self-supervised Learning](/docs/task/self-supervised)

## 2021-2022: Diffusion 시대
- [Diffusion Process](/docs/math/diffusion-process)
- [Stable Diffusion](/docs/architecture/generative/stable-diffusion)
- [Generation](/docs/task/generation)

## 2023: Controllable Generation & SAM
- [ControlNet](/docs/architecture/generative/controlnet)
- [SAM](/docs/architecture/segmentation/sam)

## 2023-2024: VLM & DiT
- [VLM](/docs/architecture/multimodal/vlm)
- [DiT](/docs/architecture/transformer/dit)
- [Vision-Language](/docs/task/vision-language)

## 2024-현재: 3D & Video Generation
- [NeRF](/docs/architecture/3d/nerf)
- [3D Gaussian Splatting](/docs/architecture/3d/3dgs)
- [3D Vision](/docs/task/3d-vision)
```

---

## 2. Top-Down 페이지

> 문제/태스크별 "순서"만 제공. 해당 문제를 풀기 위해 어떤 콘텐츠를 봐야 하는지 링크.

```markdown
# Top-Down

문제를 정하고, 필요한 지식을 찾아갑니다.

## Image Classification
1. [Cross-entropy Loss](/docs/math/loss-functions)
2. [CNN 기초](/docs/architecture/cnn)
3. [ResNet](/docs/architecture/cnn/resnet)
4. [Classification](/docs/task/classification)

## Object Detection
1. [IoU & NMS](/docs/math/iou-nms)
2. [Anchor Box](/docs/math/anchor)
3. [Faster R-CNN](/docs/architecture/detection/faster-rcnn)
4. [YOLO](/docs/architecture/detection/yolo)
5. [Detection](/docs/task/detection)

## Segmentation
1. [Transposed Convolution](/docs/math/transposed-conv)
2. [U-Net](/docs/architecture/segmentation/unet)
3. [Mask R-CNN](/docs/architecture/segmentation/mask-rcnn)
4. [Segmentation](/docs/task/segmentation)

## Image Generation
1. [확률분포](/docs/math/probability)
2. [VAE](/docs/architecture/generative/vae)
3. [GAN](/docs/architecture/generative/gan)
4. [Diffusion](/docs/math/diffusion-process)
5. [Stable Diffusion](/docs/architecture/generative/stable-diffusion)
6. [ControlNet](/docs/architecture/generative/controlnet)
7. [DiT](/docs/architecture/transformer/dit)
8. [Generation](/docs/task/generation)

## Vision-Language (VLM)
1. [Contrastive Learning](/docs/math/contrastive)
2. [CLIP](/docs/architecture/multimodal/clip)
3. [VLM](/docs/architecture/multimodal/vlm)
4. [Vision-Language](/docs/task/vision-language)

## 3D Vision
1. [카메라 모델](/docs/math/geometry)
2. [NeRF](/docs/architecture/3d/nerf)
3. [3D Vision](/docs/task/3d-vision)

## 모델 배포
1. [Quantization](/docs/math/quantization)
2. [Deployment](/docs/task/deployment)
```

---

## 3. Bottom-Up 페이지

> 개념별 "순서"만 제공. 기초부터 어떤 순서로 콘텐츠를 봐야 하는지 링크.

```markdown
# Bottom-Up

기초 개념부터 차근차근 쌓아올립니다.

## Level 1: 수학 기초
- [선형대수](/docs/math/linear-algebra)
- [미적분 & Chain Rule](/docs/math/calculus)
- [확률/통계](/docs/math/probability)

## Level 2: 딥러닝 기초
- [Convolution](/docs/math/convolution)
- [Backpropagation](/docs/math/backpropagation)
- [Loss Functions](/docs/math/loss-functions)
- [Optimization](/docs/math/optimization)

## Level 3: 기본 아키텍처
- [CNN 기초](/docs/architecture/cnn)
- [AlexNet](/docs/architecture/cnn/alexnet)
- [VGG](/docs/architecture/cnn/vgg)
- [ResNet](/docs/architecture/cnn/resnet)

## Level 4: 기본 태스크
- [Classification](/docs/task/classification)
- [Detection](/docs/task/detection)
- [Segmentation](/docs/task/segmentation)

## Level 5: 고급 개념
- [Attention](/docs/math/attention)
- [Transformer](/docs/architecture/transformer)
- [ViT](/docs/architecture/transformer/vit)

## Level 6: 생성 모델
- [Diffusion Process](/docs/math/diffusion-process)
- [Stable Diffusion](/docs/architecture/generative/stable-diffusion)
- [ControlNet](/docs/architecture/generative/controlnet)
- [DiT](/docs/architecture/transformer/dit)
- [Generation](/docs/task/generation)

## Level 7: Multimodal & 3D
- [Contrastive Learning](/docs/math/contrastive)
- [CLIP](/docs/architecture/multimodal/clip)
- [VLM](/docs/architecture/multimodal/vlm)
- [Vision-Language](/docs/task/vision-language)
- [NeRF](/docs/architecture/3d/nerf)
- [3D Vision](/docs/task/3d-vision)
```

---

## 4. 공유 콘텐츠 목록

### 수학 (`/docs/math/`)

```
math/
├── linear-algebra/              # 선형대수
│   ├── _index.md               # 선형대수 개요
│   ├── matrix.md               # 행렬 연산
│   ├── eigenvalue.md           # 고유값, 고유벡터
│   └── svd.md                  # SVD 분해
├── calculus/                    # 미적분/최적화
│   ├── _index.md               # 미적분 개요
│   ├── gradient.md             # 편미분, Gradient
│   ├── backpropagation.md      # 역전파 알고리즘
│   └── chain-rule.md           # Chain Rule
├── probability/                 # 확률/통계
│   ├── _index.md               # 확률 개요
│   ├── bayes.md                # 베이즈 정리
│   ├── distribution.md         # 확률분포
│   └── sampling.md             # 샘플링 기법
├── convolution/                 # 합성곱 관련
│   ├── _index.md               # Convolution 개요
│   ├── conv2d.md               # 2D Convolution 연산
│   ├── pooling.md              # Pooling 연산
│   ├── receptive-field.md      # Receptive Field
│   └── transposed-conv.md      # Transposed Convolution
├── attention/                   # 어텐션 관련
│   ├── _index.md               # Attention 개요
│   ├── self-attention.md       # Self-Attention
│   ├── cross-attention.md      # Cross-Attention
│   └── positional-encoding.md  # Positional Encoding
├── normalization/               # 정규화 기법
│   ├── _index.md               # Normalization 개요
│   ├── batch-norm.md           # Batch Normalization
│   ├── layer-norm.md           # Layer Normalization
│   └── rms-norm.md             # RMSNorm
├── training/                    # 학습 기법
│   ├── loss/                   # 손실 함수
│   │   ├── _index.md           # Loss 개요
│   │   ├── cross-entropy.md    # Cross-Entropy Loss
│   │   ├── focal-loss.md       # Focal Loss
│   │   └── contrastive-loss.md # Contrastive Loss
│   ├── optimizer/              # 최적화 알고리즘
│   │   ├── _index.md           # Optimizer 개요
│   │   ├── sgd.md              # SGD
│   │   ├── adam.md             # Adam, AdamW
│   │   └── lr-scheduler.md     # Learning Rate Scheduler
│   ├── regularization/         # 정규화
│   │   ├── _index.md           # Regularization 개요
│   │   ├── dropout.md          # Dropout
│   │   ├── weight-decay.md     # Weight Decay
│   │   └── label-smoothing.md  # Label Smoothing
│   └── peft/                   # 파라미터 효율적 학습
│       ├── _index.md           # PEFT 개요
│       ├── lora.md             # LoRA
│       ├── qlora.md            # QLoRA
│       ├── adapter.md          # Adapter
│       └── prefix-tuning.md    # Prefix Tuning
├── geometry/                    # 기하학
│   ├── _index.md               # 기하학 개요
│   ├── camera-model.md         # 카메라 모델
│   └── homography.md           # 호모그래피
├── detection/                   # Detection 관련 수학
│   ├── _index.md               # Detection 수학 개요
│   ├── iou.md                  # IoU
│   ├── nms.md                  # NMS
│   └── anchor.md               # Anchor Box
├── diffusion/                   # Diffusion 관련
│   ├── _index.md               # Diffusion 개요
│   ├── ddpm.md                 # DDPM
│   └── score-matching.md       # Score Matching
└── quantization/                # 양자화
    ├── _index.md               # Quantization 개요
    └── ptq-qat.md              # PTQ vs QAT
```

### 아키텍처 (`/docs/architecture/`)

```
architecture/
├── classical/                    # Classical CV
│   ├── _index.md
│   └── sift-hog.md              # SIFT, HOG
├── cnn/                          # CNN 계열
│   ├── _index.md                # CNN 기본 구조
│   ├── alexnet.md
│   ├── vgg.md
│   └── resnet.md
├── detection/                    # Detection 모델
│   ├── _index.md
│   ├── faster-rcnn.md
│   └── yolo.md
├── segmentation/                 # Segmentation 모델
│   ├── _index.md
│   ├── unet.md
│   ├── mask-rcnn.md
│   └── sam.md
├── transformer/                  # Transformer 계열
│   ├── _index.md                # Transformer 기본
│   ├── vit.md
│   └── dit.md
├── generative/                   # 생성 모델
│   ├── _index.md
│   ├── vae.md
│   ├── gan.md
│   ├── stable-diffusion.md
│   └── controlnet.md
├── multimodal/                   # Multimodal 모델
│   ├── _index.md
│   ├── clip.md
│   └── vlm.md
└── 3d/                           # 3D 모델
    ├── _index.md
    ├── nerf.md
    └── 3dgs.md
```

### 태스크 (`/docs/task/`)

| 파일 | 내용 |
|------|------|
| `classification.md` | 문제 정의, 평가지표(Accuracy, Top-k), 데이터셋(ImageNet) |
| `detection.md` | 문제 정의, 평가지표(mAP, IoU), 데이터셋(COCO) |
| `segmentation.md` | 문제 정의, 평가지표(mIoU, Dice), 데이터셋 |
| `generation.md` | 문제 정의, 평가지표(FID, IS), 데이터셋 |
| `3d-vision.md` | 문제 정의, 평가지표, 데이터셋 |
| `self-supervised.md` | 문제 정의, 평가 방식 |
| `vision-language.md` | VQA, Image Captioning, 평가지표 |
| `deployment.md` | 최적화, 배포 파이프라인 |

### 실무 기술 (`/docs/engineering/`)

```
engineering/
├── data/                        # 데이터 엔지니어링
│   ├── _index.md
│   ├── augmentation.md          # 데이터 증강
│   ├── formats.md               # 데이터 포맷 (COCO, YOLO 등)
│   ├── pipeline.md              # 데이터 파이프라인
│   └── labeling.md              # 레이블링 도구/방법
├── deployment/                  # 모델 배포
│   ├── _index.md
│   ├── onnx.md                  # ONNX 변환
│   ├── tensorrt.md              # TensorRT 최적화
│   ├── serving.md               # 모델 서빙 (Triton 등)
│   └── optimization.md          # 양자화, 프루닝
└── hardware/                    # 하드웨어
    ├── _index.md
    ├── camera.md                # 카메라 센서, 렌즈
    ├── lighting.md              # 조명 설계
    └── edge.md                  # 엣지 디바이스 (Jetson 등)
```

### 기타 (`/docs/etc/`)

| 파일 | 내용 |
|------|------|
| `papers.md` | 필독 논문 리스트 + 링크 |
| `courses.md` | 추천 강의 (CS231n 등) |
| `books.md` | 추천 도서 |
| `tools.md` | 도구/라이브러리 (PyTorch, mmcv 등) |

---

## 5. 콘텐츠 간 연결 예시

```markdown
# ResNet (architecture/resnet.md)

## 선행 지식
- [Convolution](/docs/math/convolution)
- [VGG](/docs/architecture/vgg)

## 핵심 개념
Skip Connection을 통해...

## 후속 학습
- [ResNeXt](/docs/architecture/resnext)
- [DenseNet](/docs/architecture/densenet)

## 응용되는 태스크
- [Classification](/docs/task/classification)
- [Detection](/docs/task/detection) - backbone으로 사용

## 참고 자료
- [논문 목록](/docs/etc/papers#resnet)
```

---

## 6. 유지보수

| 작업 | 수정 위치 |
|------|----------|
| 수식 개념 추가 | `/docs/math/` |
| 모델 구조 추가 | `/docs/architecture/` |
| 태스크 추가 | `/docs/task/` |
| 외부 자료 추가 | `/docs/etc/` |
| 학습 순서 변경 | 입구 페이지 (타임라인, Top-Down, Bottom-Up) |

---

## 요약

```
입구 (3개)          공유 콘텐츠 (5개 카테고리)
─────────────       ─────────────────────────
타임라인      ──┐
                ├──→  math/         수식이 핵심
Top-Down     ──┤      architecture/ 모델 구조
                ├──→  task/         문제+평가+데이터
Bottom-Up    ──┘      engineering/  실무 기술
                      etc/          외부 자료
```
