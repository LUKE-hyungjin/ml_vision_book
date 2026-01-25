# Vision Engineer 지식 가이드 - 콘텐츠 구조

## 개요

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

**링크 규칙:**
- 입구 → 콘텐츠: O
- 콘텐츠 → 콘텐츠: O
- 입구 → 입구: X
- 콘텐츠 → 입구: X

---

## 콘텐츠 구조

### math/

```
math/
├── linear-algebra/          # 선형대수
│   ├── _index.md
│   ├── matrix.md
│   ├── eigenvalue.md
│   └── svd.md
├── calculus/                # 미적분
│   ├── _index.md
│   ├── gradient.md
│   ├── backpropagation.md
│   └── chain-rule.md
├── probability/             # 확률/통계
│   ├── _index.md
│   ├── bayes.md
│   ├── distribution.md
│   └── sampling.md
├── convolution/             # 합성곱
│   ├── _index.md
│   ├── conv2d.md
│   ├── pooling.md
│   ├── receptive-field.md
│   └── transposed-conv.md
├── attention/               # 어텐션
│   ├── _index.md
│   ├── self-attention.md
│   ├── cross-attention.md
│   └── positional-encoding.md
├── normalization/           # 정규화
│   ├── _index.md
│   ├── batch-norm.md
│   ├── layer-norm.md
│   └── rms-norm.md
├── training/                # 학습
│   ├── loss/
│   │   ├── _index.md
│   │   ├── cross-entropy.md
│   │   ├── focal-loss.md
│   │   └── contrastive-loss.md
│   ├── optimizer/
│   │   ├── _index.md
│   │   ├── sgd.md
│   │   ├── adam.md
│   │   └── lr-scheduler.md
│   ├── regularization/
│   │   ├── _index.md
│   │   ├── dropout.md
│   │   ├── weight-decay.md
│   │   └── label-smoothing.md
│   └── peft/
│       ├── _index.md
│       ├── lora.md
│       ├── qlora.md
│       ├── adapter.md
│       └── prefix-tuning.md
├── geometry/                # 기하학
│   ├── _index.md
│   ├── camera-model.md
│   └── homography.md
├── detection/               # Detection 수학
│   ├── _index.md
│   ├── iou.md
│   ├── nms.md
│   └── anchor.md
├── diffusion/               # Diffusion
│   ├── _index.md
│   ├── ddpm.md
│   └── score-matching.md
└── quantization/            # 양자화
    ├── _index.md
    └── ptq-qat.md
```

### architecture/

```
architecture/
├── classical/
│   ├── _index.md
│   └── sift-hog.md
├── cnn/
│   ├── _index.md
│   ├── alexnet.md
│   ├── vgg.md
│   └── resnet.md
├── detection/
│   ├── _index.md
│   ├── faster-rcnn.md
│   └── yolo.md
├── segmentation/
│   ├── _index.md
│   ├── unet.md
│   ├── mask-rcnn.md
│   └── sam.md
├── transformer/
│   ├── _index.md
│   ├── vit.md
│   └── dit.md
├── generative/
│   ├── _index.md
│   ├── vae.md
│   ├── gan.md
│   ├── stable-diffusion.md
│   └── controlnet.md
├── multimodal/
│   ├── _index.md
│   ├── clip.md
│   └── vlm.md
└── 3d/
    ├── _index.md
    ├── nerf.md
    └── 3dgs.md
```

### task/

```
task/
├── _index.md
├── classification.md
├── detection.md
├── segmentation.md
├── generation.md
├── 3d-vision.md
├── self-supervised.md
├── vision-language.md
└── deployment.md
```

### engineering/

```
engineering/
├── data/
│   ├── _index.md
│   ├── augmentation.md
│   ├── formats.md
│   ├── pipeline.md
│   └── labeling.md
├── deployment/
│   ├── _index.md
│   ├── onnx.md
│   ├── tensorrt.md
│   ├── serving.md
│   └── optimization.md
└── hardware/
    ├── _index.md
    ├── camera.md
    ├── lighting.md
    └── edge.md
```

### etc/

```
etc/
├── _index.md
├── papers.md
├── courses.md
├── books.md
└── tools.md
```
