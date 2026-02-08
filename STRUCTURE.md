# Vision Engineer 지식 가이드 - 콘텐츠 구조

## 개요

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  타임라인    │     │  Top-Down   │     │  Bottom-Up  │
│  (순서 제공)  │     │  (순서 제공)  │     │  (순서 제공)  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                         공유 콘텐츠                           │
│  ┌──────┐ ┌────────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌────┐ │
│  │ 수학  │ │컴포넌트│ │아키텍처│ │태스크│ │엔지니어링│ │기타│ │
│  └──────┘ └────────┘ └────────┘ └──────┘ └────────┘ └────┘ │
└──────────────────────────────────────────────────────────────┘
```

**학습 흐름:**
```
math/ (수학을 배운다) → components/ (부품을 배운다) → architecture/ (설계도를 읽는다)
                                                    → task/ (문제를 정의한다)
                                                    → engineering/ (현장에 적용한다)
```

**링크 규칙:**
- 입구 → 콘텐츠: O
- 콘텐츠 → 콘텐츠: O
- 입구 → 입구: X
- 콘텐츠 → 입구: X

**체크박스 범례:**
- [x] 검토 완료 (이미지 포함, 내용 확인됨)
- [ ] 미검토 (내용 개선 필요)
- *(planned)* 아직 파일 없음

---

## math/ — 순수 수학

> 판단 기준: "이 개념은 딥러닝 없이도 존재하는 수학인가?"

### math/linear-algebra/ — 선형대수

- [x] matrix.md — 선수지식: vector
- [x] vector.md — 선수지식: 없음
- [x] eigenvalue.md — 선수지식: matrix, vector
- [x] svd.md — 선수지식: matrix, eigenvalue

### math/calculus/ — 미적분

- [x] basics.md — 선수지식: 없음
- [x] gradient.md — 선수지식: calculus/basics, linear-algebra/vector
- [x] chain-rule.md — 선수지식: calculus/basics
- [x] backpropagation.md — 선수지식: chain-rule, gradient
- [x] optimization.md — 선수지식: gradient

### math/probability/ — 확률/통계

- [x] basics.md — 선수지식: 없음
- [x] random-variable.md — 선수지식: probability/basics
- [x] expectation.md — 선수지식: random-variable
- [x] distribution.md — 선수지식: random-variable, expectation
- [x] bayes.md — 선수지식: probability/basics
- [x] entropy.md — 선수지식: probability/basics, expectation
- [x] kl-divergence.md — 선수지식: entropy, distribution
- [x] mle.md — 선수지식: distribution, probability/basics
- [x] sampling.md — 선수지식: distribution

### math/geometry/ — 기하학

- [ ] camera-model.md — 선수지식: linear-algebra/matrix
- [ ] epipolar.md — 선수지식: camera-model, linear-algebra/matrix
- [ ] homography.md — 선수지식: linear-algebra/matrix

---

## components/ — 딥러닝 빌딩 블록

> 판단 기준: "이건 모델을 만들 때 사용하는 부품/기법인가?"

### components/convolution/ — 합성곱

- [ ] conv2d.md — 선수지식: math/linear-algebra/matrix
- [ ] pooling.md — 선수지식: conv2d
- [ ] receptive-field.md — 선수지식: conv2d, pooling
- [ ] transposed-conv.md — 선수지식: conv2d

### components/attention/ — 어텐션

- [ ] self-attention.md — 선수지식: math/linear-algebra/matrix, math/calculus/basics
- [ ] cross-attention.md — 선수지식: self-attention
- [ ] positional-encoding.md — 선수지식: self-attention

### components/normalization/ — 정규화

- [ ] batch-norm.md — 선수지식: math/probability/expectation
- [ ] layer-norm.md — 선수지식: batch-norm
- [ ] rms-norm.md — 선수지식: layer-norm

### components/activation/ — 활성화 함수 *(planned)*

- *(planned)* relu.md — 선수지식: math/calculus/basics
- *(planned)* gelu.md — 선수지식: relu, math/probability/distribution
- *(planned)* sigmoid.md — 선수지식: math/calculus/basics

### components/detection/ — Detection 연산

- [ ] iou.md — 선수지식: 없음
- [ ] anchor.md — 선수지식: iou
- [ ] nms.md — 선수지식: iou

### components/generative/ — 생성 모델 수학

- [x] ddpm.md — 선수지식: math/probability/distribution, math/probability/kl-divergence
- [ ] score-matching.md — 선수지식: math/calculus/gradient, math/probability/distribution
- [ ] flow-matching.md — 선수지식: math/probability/distribution, math/calculus/basics
- [ ] sampling.md — 선수지식: math/probability/sampling

### components/quantization/ — 양자화

- [ ] data-types.md — 선수지식: 없음
- [ ] ptq.md — 선수지식: data-types
- [ ] qat.md — 선수지식: ptq, components/training/loss/cross-entropy

### components/training/loss/ — 손실 함수

- [ ] cross-entropy.md — 선수지식: math/probability/entropy
- [ ] focal-loss.md — 선수지식: cross-entropy
- [ ] contrastive-loss.md — 선수지식: math/linear-algebra/vector

### components/training/optimizer/ — 최적화 알고리즘

- [ ] sgd.md — 선수지식: math/calculus/gradient
- [ ] adam.md — 선수지식: sgd
- [ ] lr-scheduler.md — 선수지식: sgd

### components/training/regularization/ — 정규화 기법

- [ ] dropout.md — 선수지식: math/probability/basics
- [ ] weight-decay.md — 선수지식: math/calculus/optimization
- [ ] label-smoothing.md — 선수지식: components/training/loss/cross-entropy

### components/training/peft/ — 파라미터 효율적 학습

- [x] lora.md — 선수지식: math/linear-algebra/matrix, math/linear-algebra/svd
- [ ] qlora.md — 선수지식: lora, components/quantization/data-types
- [ ] adapter.md — 선수지식: 없음
- [ ] prefix-tuning.md — 선수지식: components/attention/self-attention

---

## architecture/ — 모델/논문

> 판단 기준: "이건 특정 논문이 제안한 모델 구조인가?"

### architecture/classical/ — Classical CV

- [ ] sift-hog.md — 선수지식: components/convolution/conv2d

### architecture/cnn/ — CNN

- [x] alexnet.md — 선수지식: components/convolution/conv2d
- [x] vgg.md — 선수지식: alexnet
- [x] resnet.md — 선수지식: vgg, components/normalization/batch-norm
- *(planned)* inception.md — 선수지식: vgg

### architecture/transformer/ — Transformer

- [x] vit.md — 선수지식: components/attention/self-attention, components/normalization/layer-norm
- [ ] dit.md — 선수지식: vit, components/generative/ddpm

### architecture/detection/ — Detection

- [x] yolo.md — 선수지식: components/convolution/conv2d, components/detection/iou, components/detection/nms
- [ ] faster-rcnn.md — 선수지식: components/convolution/conv2d, components/detection/anchor, components/detection/nms

### architecture/segmentation/ — Segmentation

- [ ] unet.md — 선수지식: components/convolution/conv2d, components/convolution/transposed-conv
- [ ] mask-rcnn.md — 선수지식: faster-rcnn
- [ ] sam.md — 선수지식: vit, unet

### architecture/generative/ — Generative

- [x] gan.md — 선수지식: components/training/loss/cross-entropy
- [x] vae.md — 선수지식: math/probability/kl-divergence, math/probability/mle
- [x] ddpm.md — 선수지식: components/generative/ddpm, unet
- [x] stable-diffusion.md — 선수지식: ddpm, vae, components/attention/cross-attention
- [ ] dall-e.md — 선수지식: vae, components/attention/self-attention
- [x] stylegan.md — 선수지식: gan
- [ ] vqgan.md — 선수지식: gan, vae
- [x] controlnet.md — 선수지식: stable-diffusion
- [x] dit.md — 선수지식: vit, ddpm
- [x] flux.md — 선수지식: dit, stable-diffusion
- [x] qwen-image-edit.md — 선수지식: stable-diffusion

### architecture/multimodal/ — Multimodal

- [x] clip.md — 선수지식: components/attention/self-attention, components/training/loss/contrastive-loss
- [ ] vlm.md — 선수지식: clip, vit

### architecture/3d/ — 3D Vision

- [ ] nerf.md — 선수지식: math/geometry/camera-model, components/attention/positional-encoding
- [ ] 3dgs.md — 선수지식: math/geometry/camera-model

---

## task/ — 태스크

> 판단 기준: "이건 '무엇을 푸는가'에 대한 설명인가?"

- *(planned)* classification.md — 선수지식: 없음
- *(planned)* detection.md — 선수지식: components/detection/iou
- *(planned)* segmentation.md — 선수지식: 없음
- *(planned)* generation.md — 선수지식: math/probability/distribution
- *(planned)* 3d-vision.md — 선수지식: math/geometry/camera-model
- *(planned)* self-supervised.md — 선수지식: components/training/loss/contrastive-loss
- *(planned)* vision-language.md — 선수지식: components/attention/cross-attention
- *(planned)* deployment.md — 선수지식: 없음

---

## engineering/ — 실무 기술

> 판단 기준: "이건 실무에서 사용하는 도구/기법인가?"

### engineering/data/ — 데이터

- [x] pipeline.md — 선수지식: 없음
- [x] augmentation.md — 선수지식: 없음
- [ ] labeling.md — 선수지식: 없음
- [x] formats.md — 선수지식: 없음

### engineering/deployment/ — 배포

- [ ] serving.md — 선수지식: 없음
- [ ] optimization.md — 선수지식: 없음
- [ ] tensorrt.md — 선수지식: components/quantization/ptq
- [ ] onnx.md — 선수지식: 없음

### engineering/hardware/ — 하드웨어

- [ ] camera.md — 선수지식: math/geometry/camera-model
- [x] edge.md — 선수지식: 없음
- [ ] lighting.md — 선수지식: 없음

---

## etc/ — 기타 자료

> 판단 기준: "이건 외부 자료 모음인가?"

- *(planned)* papers.md
- *(planned)* courses.md
- *(planned)* books.md
- *(planned)* tools.md
