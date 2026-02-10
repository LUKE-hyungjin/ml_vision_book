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

- [x] vector.md — 선수지식: 없음
- [x] matrix.md — 선수지식: vector
- [x] vector-spaces.md — 선수지식: vector
- [x] linear-transformations.md — 선수지식: matrix, vector-spaces
- [x] linear-systems.md — 선수지식: matrix
- [x] eigenvalue.md — 선수지식: matrix, vector
- [x] svd.md — 선수지식: matrix, eigenvalue

### math/calculus/ — 미적분

- [x] basics.md — 선수지식: 없음
- [x] integral-basics.md — 선수지식: basics
- [x] taylor-series.md — 선수지식: basics
- [x] gradient.md — 선수지식: calculus/basics, linear-algebra/vector
- [x] chain-rule.md — 선수지식: calculus/basics
- [x] multivariate-calculus.md — 선수지식: gradient, chain-rule
- [x] backpropagation.md — 선수지식: chain-rule, gradient
- [x] optimization.md — 선수지식: gradient

### math/probability/ — 확률/통계

- [x] basics.md — 선수지식: 없음
- [x] random-variable.md — 선수지식: probability/basics
- [x] expectation.md — 선수지식: random-variable
- [x] covariance-correlation.md — 선수지식: expectation, random-variable
- [x] distribution.md — 선수지식: random-variable, expectation
- [x] joint-conditional.md — 선수지식: random-variable, distribution
- [x] multivariate-gaussian.md — 선수지식: distribution, covariance-correlation, linear-algebra/matrix
- [x] bayes.md — 선수지식: probability/basics
- [x] entropy.md — 선수지식: probability/basics, expectation
- [x] kl-divergence.md — 선수지식: entropy, distribution
- [x] mutual-information.md — 선수지식: entropy, kl-divergence, joint-conditional
- [x] mle.md — 선수지식: distribution, probability/basics
- [x] sampling.md — 선수지식: distribution
- [x] markov-chains.md — 선수지식: basics, random-variable

### math/geometry/ — 기하학

- *(planned)* homogeneous-coordinates.md — 선수지식: linear-algebra/matrix, linear-algebra/vector
- *(planned)* coordinate-transforms.md — 선수지식: linear-algebra/matrix, homogeneous-coordinates
- *(planned)* projective-geometry.md — 선수지식: homogeneous-coordinates, linear-algebra/matrix
- [ ] camera-model.md — 선수지식: linear-algebra/matrix, coordinate-transforms
- *(planned)* lens-distortion.md — 선수지식: camera-model
- [ ] epipolar.md — 선수지식: camera-model, homogeneous-coordinates
- [ ] homography.md — 선수지식: linear-algebra/matrix, homogeneous-coordinates

### math/signal-processing/ — 신호처리

- *(planned)* fourier-transform.md — 선수지식: calculus/basics, calculus/integral-basics
- *(planned)* sampling-aliasing.md — 선수지식: fourier-transform
- *(planned)* filters.md — 선수지식: fourier-transform

---

## components/ — 딥러닝 빌딩 블록

> 판단 기준: "이건 모델을 만들 때 사용하는 부품/기법인가?"

### components/convolution/ — 합성곱

- [ ] conv2d.md — 선수지식: math/linear-algebra/matrix
- [ ] pooling.md — 선수지식: conv2d
- [ ] receptive-field.md — 선수지식: conv2d, pooling
- [ ] transposed-conv.md — 선수지식: conv2d
- [ ] depthwise-separable-conv.md — 선수지식: conv2d
- [ ] dilated-conv.md — 선수지식: conv2d, receptive-field
- [ ] deformable-conv.md — 선수지식: conv2d
- [ ] grouped-conv.md — 선수지식: conv2d

### components/attention/ — 어텐션

- [ ] self-attention.md — 선수지식: math/linear-algebra/matrix, math/calculus/basics
- [ ] multi-head-attention.md — 선수지식: self-attention
- [ ] cross-attention.md — 선수지식: self-attention
- [ ] positional-encoding.md — 선수지식: self-attention
- [ ] window-attention.md — 선수지식: multi-head-attention
- [ ] flash-attention.md — 선수지식: multi-head-attention

### components/normalization/ — 정규화

- [x] batch-norm.md — 선수지식: math/probability/expectation
- [x] layer-norm.md — 선수지식: batch-norm
- [x] group-norm.md — 선수지식: batch-norm
- [x] instance-norm.md — 선수지식: batch-norm
- [x] rms-norm.md — 선수지식: layer-norm

### components/activation/ — 활성화 함수

- [x] relu.md — 선수지식: math/calculus/basics
- [x] sigmoid.md — 선수지식: math/calculus/basics
- [x] softmax.md — 선수지식: math/probability/basics, math/calculus/basics
- [x] gelu.md — 선수지식: relu, math/probability/distribution
- [x] swish-silu.md — 선수지식: sigmoid

### components/structural/ — 구조 패턴

- *(planned)* residual-block.md — 선수지식: conv2d
- *(planned)* skip-connection.md — 선수지식: 없음
- *(planned)* bottleneck-block.md — 선수지식: residual-block, conv2d
- *(planned)* encoder-decoder.md — 선수지식: conv2d, transposed-conv
- *(planned)* feature-pyramid-network.md — 선수지식: conv2d, skip-connection
- *(planned)* backbone-neck-head.md — 선수지식: 없음
- *(planned)* patch-embedding.md — 선수지식: conv2d, math/linear-algebra/matrix

### components/detection/ — Detection 연산

- [ ] iou.md — 선수지식: 없음
- [ ] anchor.md — 선수지식: iou
- [ ] nms.md — 선수지식: iou

### components/generative/ — 생성 모델 수학

- [x] ddpm.md — 선수지식: math/probability/distribution, math/probability/kl-divergence
- [ ] score-matching.md — 선수지식: math/calculus/gradient, math/probability/distribution
- [ ] flow-matching.md — 선수지식: math/probability/distribution, math/calculus/basics
- [ ] sampling.md — 선수지식: math/probability/sampling
- *(planned)* vae-math.md — 선수지식: math/probability/distribution, math/probability/kl-divergence, math/probability/mle
- *(planned)* classifier-free-guidance.md — 선수지식: ddpm
- *(planned)* noise-schedule.md — 선수지식: ddpm

### components/compression/ — 모델 경량화

- *(planned)* knowledge-distillation.md — 선수지식: components/training/loss/cross-entropy
- *(planned)* pruning.md — 선수지식: 없음
- *(planned)* nas.md — 선수지식: 없음

### components/quantization/ — 양자화

- [ ] data-types.md — 선수지식: 없음
- *(planned)* mixed-precision.md — 선수지식: data-types
- [ ] ptq.md — 선수지식: data-types
- [ ] qat.md — 선수지식: ptq, components/training/loss/cross-entropy

### components/training/loss/ — 손실 함수

- [ ] cross-entropy.md — 선수지식: math/probability/entropy
- [ ] focal-loss.md — 선수지식: cross-entropy
- [ ] contrastive-loss.md — 선수지식: math/linear-algebra/vector
- *(planned)* dice-loss.md — 선수지식: 없음
- *(planned)* smooth-l1-loss.md — 선수지식: 없음
- *(planned)* iou-loss.md — 선수지식: components/detection/iou
- *(planned)* triplet-loss.md — 선수지식: math/linear-algebra/vector
- *(planned)* mse-loss.md — 선수지식: math/calculus/basics
- *(planned)* perceptual-loss.md — 선수지식: components/convolution/conv2d

### components/training/optimizer/ — 최적화 알고리즘

- [ ] sgd.md — 선수지식: math/calculus/gradient
- [ ] adam.md — 선수지식: sgd
- *(planned)* adamw.md — 선수지식: adam, weight-decay
- [ ] lr-scheduler.md — 선수지식: sgd
- *(planned)* ema.md — 선수지식: 없음
- *(planned)* gradient-clipping.md — 선수지식: math/calculus/gradient

### components/training/regularization/ — 정규화 기법

- [ ] dropout.md — 선수지식: math/probability/basics
- [ ] weight-decay.md — 선수지식: math/calculus/optimization
- [ ] label-smoothing.md — 선수지식: components/training/loss/cross-entropy
- *(planned)* data-augmentation.md — 선수지식: 없음
- *(planned)* stochastic-depth.md — 선수지식: components/structural/residual-block

### components/training/peft/ — 파라미터 효율적 학습

- [x] lora.md — 선수지식: math/linear-algebra/matrix, math/linear-algebra/svd
- [ ] qlora.md — 선수지식: lora, components/quantization/data-types
- [ ] adapter.md — 선수지식: 없음
- [ ] prefix-tuning.md — 선수지식: components/attention/self-attention

### components/embedding/ — 임베딩/토큰화

- *(planned)* patch-embedding.md — 선수지식: conv2d, math/linear-algebra/matrix
- *(planned)* cls-token.md — 선수지식: components/attention/self-attention
- *(planned)* linear-projection.md — 선수지식: math/linear-algebra/matrix

### components/video/ — 영상(비디오) 처리

- *(planned)* optical-flow.md — 선수지식: conv2d, math/calculus/gradient
- *(planned)* temporal-modeling.md — 선수지식: conv2d, components/attention/self-attention

### components/self-supervised/ — 자기지도 학습

- *(planned)* contrastive-learning.md — 선수지식: components/training/loss/contrastive-loss
- *(planned)* masked-image-modeling.md — 선수지식: components/attention/self-attention, patch-embedding
- *(planned)* self-distillation.md — 선수지식: components/compression/knowledge-distillation

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
- *(planned)* densenet.md — 선수지식: resnet
- *(planned)* convnext.md — 선수지식: resnet, vit

### architecture/efficient/ — 경량화 모델

- *(planned)* mobilenet.md — 선수지식: components/convolution/depthwise-separable-conv
- *(planned)* efficientnet.md — 선수지식: mobilenet, components/compression/nas
- *(planned)* shufflenet.md — 선수지식: components/convolution/grouped-conv
- *(planned)* mobilevit.md — 선수지식: mobilenet, vit

### architecture/transformer/ — Transformer

- [x] vit.md — 선수지식: components/attention/self-attention, components/normalization/layer-norm
- *(planned)* deit.md — 선수지식: vit, components/compression/knowledge-distillation
- *(planned)* swin-transformer.md — 선수지식: vit, components/attention/window-attention
- *(planned)* beit.md — 선수지식: vit, components/self-supervised/masked-image-modeling
- [ ] dit.md — 선수지식: vit, components/generative/ddpm

### architecture/detection/ — Detection

- [x] yolo.md — 선수지식: components/convolution/conv2d, components/detection/iou, components/detection/nms
- [ ] faster-rcnn.md — 선수지식: components/convolution/conv2d, components/detection/anchor, components/detection/nms
- *(planned)* ssd.md — 선수지식: components/convolution/conv2d, components/detection/anchor
- *(planned)* detr.md — 선수지식: vit, components/attention/cross-attention
- *(planned)* rt-detr.md — 선수지식: detr
- *(planned)* grounding-dino.md — 선수지식: detr, clip

### architecture/segmentation/ — Segmentation

- [ ] unet.md — 선수지식: components/convolution/conv2d, components/convolution/transposed-conv
- *(planned)* deeplab.md — 선수지식: components/convolution/dilated-conv
- [ ] mask-rcnn.md — 선수지식: faster-rcnn
- *(planned)* mask2former.md — 선수지식: detr, mask-rcnn
- [ ] sam.md — 선수지식: vit, unet
- *(planned)* sam2.md — 선수지식: sam, components/video/temporal-modeling

### architecture/self-supervised/ — 자기지도 학습 모델

- *(planned)* simclr.md — 선수지식: components/training/loss/contrastive-loss
- *(planned)* moco.md — 선수지식: components/training/loss/contrastive-loss, components/training/optimizer/ema
- *(planned)* mae.md — 선수지식: vit, components/self-supervised/masked-image-modeling
- *(planned)* dinov2.md — 선수지식: vit, components/self-supervised/self-distillation

### architecture/generative/ — Generative

- [x] gan.md — 선수지식: components/training/loss/cross-entropy
- [x] vae.md — 선수지식: math/probability/kl-divergence, math/probability/mle
- [x] ddpm.md — 선수지식: components/generative/ddpm, unet
- [x] stable-diffusion.md — 선수지식: ddpm, vae, components/attention/cross-attention
- [ ] dall-e.md — 선수지식: vae, components/attention/self-attention
- [x] stylegan.md — 선수지식: gan
- [ ] vqgan.md — 선수지식: gan, vae
- [x] controlnet.md — 선수지식: stable-diffusion
- *(planned)* ip-adapter.md — 선수지식: stable-diffusion, clip
- [x] dit.md — 선수지식: vit, ddpm
- [x] flux.md — 선수지식: dit, stable-diffusion
- *(planned)* lcm.md — 선수지식: stable-diffusion, components/compression/knowledge-distillation
- [x] qwen-image-edit.md — 선수지식: stable-diffusion

### architecture/multimodal/ — Multimodal

- [x] clip.md — 선수지식: components/attention/self-attention, components/training/loss/contrastive-loss
- *(planned)* siglip.md — 선수지식: clip
- *(planned)* blip2.md — 선수지식: clip, vit
- *(planned)* llava.md — 선수지식: clip, vit
- *(planned)* qwen-vl.md — 선수지식: clip, vit
- [ ] vlm.md — 선수지식: clip, vit

### architecture/video/ — 비디오 이해

- *(planned)* video-swin.md — 선수지식: swin-transformer, components/video/temporal-modeling
- *(planned)* videomae.md — 선수지식: mae, components/video/temporal-modeling
- *(planned)* cogvideox.md — 선수지식: dit, components/video/temporal-modeling

### architecture/restoration/ — 영상 복원/향상

- *(planned)* esrgan.md — 선수지식: gan, components/structural/residual-block
- *(planned)* swinir.md — 선수지식: swin-transformer

### architecture/3d/ — 3D Vision

- [ ] nerf.md — 선수지식: math/geometry/camera-model, components/attention/positional-encoding
- [ ] 3dgs.md — 선수지식: math/geometry/camera-model
- *(planned)* depth-anything.md — 선수지식: vit, components/self-supervised/self-distillation

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
