# Vision Engineer 지식 가이드 - 콘텐츠 작성 가이드

이 문서는 Claude가 콘텐츠를 작성할 때 따라야 할 규칙입니다.

## 프로젝트 구조 (i18n 지원)

```
content/
├── ko/                              # 한국어
│   ├── _index.md                    # 홈페이지
│   ├── menu/index.md                # 사이드바 메뉴
│   └── docs/
│       ├── timeline/_index.md       # 입구: 타임라인
│       ├── topdown/_index.md        # 입구: Top-Down
│       ├── bottomup/_index.md       # 입구: Bottom-Up
│       ├── math/                    # 순수 수학
│       ├── components/              # 딥러닝 빌딩 블록
│       ├── architecture/            # 모델/논문
│       ├── task/                    # 태스크
│       ├── engineering/             # 실무 기술
│       └── etc/                     # 기타 자료
└── en/                              # English
    ├── _index.md
    ├── menu/index.md
    └── docs/
        ├── timeline/_index.md
        ├── topdown/_index.md
        ├── bottomup/_index.md
        ├── math/
        ├── components/
        ├── architecture/
        ├── task/
        ├── engineering/
        └── etc/
```

## i18n 규칙

### 콘텐츠 추가 시
1. **양쪽 언어에 동일한 파일 경로로 생성**
   - 한국어: `content/ko/docs/components/convolution/conv2d.md`
   - 영어: `content/en/docs/components/convolution/conv2d.md`

2. **파일명은 영어로 통일** (URL 일관성)
   - O: `linear-algebra.md`
   - X: `선형대수.md`

3. **링크는 언어 prefix 포함하여 작성**
   ```markdown
   # 한국어 콘텐츠에서
   [Convolution](/ko/docs/components/convolution/conv2d)

   # 영어 콘텐츠에서
   [Convolution](/en/docs/components/convolution/conv2d)
   ```

### 번역 우선순위
1. 한국어 먼저 작성
2. 영어 버전 생성 (구조는 동일, 내용만 번역)

## 핵심 원칙

### 이미지 규칙

모든 콘텐츠에는 **시각 자료**가 필수입니다.

1. **논문 이미지**: 원본 논문에서 핵심 Figure 추출하여 사용
   - 저장 위치: `static/images/{카테고리}/{모델명}/`
   - 예: `static/images/architecture/cnn/alexnet-fig2.png`

2. **SVG 다이어그램**: 개념 이해를 돕는 직접 제작 다이어그램
   - 저장 위치: `static/images/{카테고리}/{주제}/ko/`, `static/images/{카테고리}/{주제}/en/`
   - 한국어/영어 버전 별도 제작
   - 다크 테마 배경(`#1a1a2e`) 권장
   - 요소 간 마진 충분히 확보 (겹침 방지)

3. **나노바나나 이미지 생성 프롬프트**: 콘텐츠 작성 시 필요한 다이어그램에 대해 **나노바나나 프롬프트**를 함께 작성합니다
   - 콘텐츠에 필요한 시각 자료를 파악한 뒤, 각 이미지에 대한 프롬프트를 작성
   - 프롬프트는 콘텐츠와 별도로 사용자에게 제공 (사용자가 검토 후 생성)
   - 프롬프트 작성 시 포함할 내용:
     - 이미지의 목적 (어떤 개념을 설명하는지)
     - 포함해야 할 요소 (수식, 화살표, 레이블 등)
     - 스타일 (다크 테마 배경 `#1a1a2e`, 깔끔한 다이어그램 스타일)
     - 한국어/영어 버전 구분

4. **Hugo에서 이미지 삽입**:
   ```markdown
   {{</* figure src="/images/architecture/cnn/alexnet-fig2.png" caption="AlexNet 구조" */>}}
   ```

### 쉬운 설명 규칙

**대학교 1학년 수준**으로 작성합니다. 고등학교 수학(미적분, 확률)은 알지만, 전공 지식은 없다고 가정합니다.

1. **선수지식 명시**: [STRUCTURE.md](./STRUCTURE.md)의 의존관계를 참고하여 선수지식 링크를 작성합니다
   - STRUCTURE.md에 각 파일의 선수지식이 정의되어 있음
   - 예: "선수지식: [행렬](/ko/docs/math/linear-algebra/matrix), [벡터](/ko/docs/math/linear-algebra/vector)"

2. **비유 사용**: 복잡한 개념은 일상적인 비유로 먼저 설명
   - 예: "Attention은 책에서 중요한 부분에 형광펜을 칠하는 것과 같습니다"

3. **수식 기호 설명**: 모든 수식의 각 기호가 무엇을 의미하는지 명시
   - 예: "$\nabla_x$ : x에 대한 기울기(gradient) - 어느 방향으로 가야 값이 커지나?"

4. **단계별 설명**: 한 번에 하나의 개념만
   - 수식 → 각 기호의 의미 → 직관적 해석 → 코드

5. **시각화 우선**: 텍스트보다 그림이 먼저
   - 수식만 나열하지 말고, 다이어그램으로 흐름 보여주기

6. **"왜?"를 먼저**: 개념 설명 전에 "왜 필요한가?"부터 설명
   - 예: "깊은 네트워크에서 gradient가 사라지는 문제가 있었습니다. 이를 해결하기 위해..."

### 링크 규칙 (절대 위반 금지)

| 출발 | 도착 | 허용 |
|------|------|------|
| 입구 (timeline, topdown, bottomup) | 콘텐츠 (math, components, architecture, task, engineering, etc) | O |
| 콘텐츠 | 콘텐츠 | O |
| 입구 | 입구 | **X** |
| 콘텐츠 | 입구 | **X** |

### 콘텐츠 분류 기준

| 디렉토리 | 기준 | 판단 질문 |
|----------|------|----------|
| `math/` | 순수 수학 | "이 개념은 딥러닝 없이도 존재하는 수학인가?" |
| `components/` | 딥러닝 구성 요소 | "이건 모델을 만들 때 사용하는 부품/기법인가?" |
| `architecture/` | 모델/네트워크 구조 | "이건 특정 논문이 제안한 모델 구조인가?" |
| `task/` | 문제 정의 + 평가 + 데이터셋 | "이건 '무엇을 푸는가'에 대한 설명인가?" |
| `engineering/` | 실무 기술 | "이건 실무에서 사용하는 도구/기법인가?" |
| `etc/` | 외부 링크/자료 | "이건 외부 자료 모음인가?" |

### math/ vs components/ 구분 규칙

| 질문 | math/ | components/ |
|------|-------|-------------|
| 딥러닝 없이도 존재하는 개념인가? | O | X |
| 특정 모델에 종속되지 않는 부품인가? | X | O |
| 예시 | 행렬, 미분, 확률분포, 카메라 모델 | Conv2D, Attention, BatchNorm, SGD |

### Math 하위 분류

| 디렉토리 | 내용 | 예시 |
|----------|------|------|
| `math/linear-algebra/` | 선형대수 기초 | matrix, vector, eigenvalue, SVD, linear systems |
| `math/calculus/` | 미적분/최적화 | gradient, backprop, chain rule, Jacobian, Taylor series |
| `math/probability/` | 확률/통계 | bayes, distribution, sampling, covariance, multivariate Gaussian |
| `math/geometry/` | 기하학 | camera model, homography, homogeneous coordinates, coordinate transforms |
| `math/signal-processing/` | 신호처리 | Fourier transform, sampling/aliasing, filters |

### Components 하위 분류

| 디렉토리 | 내용 | 예시 |
|----------|------|------|
| `components/convolution/` | 합성곱 관련 | conv2d, pooling, depthwise-separable, dilated conv |
| `components/attention/` | 어텐션 관련 | self-attention, multi-head, cross-attention, window attention |
| `components/normalization/` | 정규화 기법 | batch norm, layer norm, group norm, instance norm |
| `components/activation/` | 활성화 함수 | relu, gelu, sigmoid, softmax, swish/silu |
| `components/structural/` | 구조 패턴 | residual block, skip connection, FPN, encoder-decoder |
| `components/detection/` | Detection 연산 | IoU, NMS, anchor box |
| `components/generative/` | 생성 모델 수학 | DDPM 수학, score matching, flow matching, CFG |
| `components/compression/` | 모델 경량화 | knowledge distillation, pruning, NAS |
| `components/quantization/` | 양자화 | data types, mixed precision, PTQ, QAT |
| `components/training/loss/` | 손실 함수 | cross-entropy, focal, dice, smooth-l1, IoU loss |
| `components/training/optimizer/` | 최적화 알고리즘 | SGD, Adam, AdamW, LR scheduler, EMA |
| `components/training/regularization/` | 정규화 기법 | dropout, weight decay, data augmentation, stochastic depth |
| `components/training/peft/` | 파라미터 효율적 학습 | LoRA, QLoRA, adapter, prefix tuning |
| `components/embedding/` | 임베딩/토큰화 | patch embedding, CLS token, linear projection |
| `components/video/` | 영상(비디오) 처리 | optical flow, temporal modeling |
| `components/self-supervised/` | 자기지도 학습 | contrastive learning, masked image modeling |

### Architecture 하위 분류

| 디렉토리 | 내용 | 예시 |
|----------|------|------|
| `architecture/classical/` | Classical CV | SIFT, HOG |
| `architecture/cnn/` | CNN 아키텍처 | AlexNet, VGG, ResNet, ConvNeXt |
| `architecture/efficient/` | 경량화 모델 | MobileNet, EfficientNet, ShuffleNet |
| `architecture/transformer/` | Transformer | ViT, DeiT, Swin Transformer, BEiT |
| `architecture/detection/` | Detection | YOLO, Faster R-CNN, DETR, RT-DETR |
| `architecture/segmentation/` | Segmentation | U-Net, DeepLab, Mask R-CNN, SAM |
| `architecture/self-supervised/` | 자기지도 학습 모델 | SimCLR, MoCo, MAE, DINOv2 |
| `architecture/generative/` | 생성 모델 | GAN, VAE, DDPM, Stable Diffusion, DiT |
| `architecture/multimodal/` | 멀티모달 | CLIP, SigLIP, BLIP-2, LLaVA |
| `architecture/video/` | 비디오 이해 | Video Swin, VideoMAE, CogVideoX |
| `architecture/restoration/` | 영상 복원/향상 | ESRGAN, SwinIR |
| `architecture/3d/` | 3D Vision | NeRF, 3DGS, Depth Anything |

## 콘텐츠 작성 템플릿

### math/ 템플릿

**한국어** (`content/ko/docs/math/xxx.md`):
```markdown
---
title: "개념명"
weight: 10
math: true
---

# 개념명

{{%/* hint info */%}}
**선수지식**: [필요한 개념1](/ko/docs/math/xxx) | [필요한 개념2](/ko/docs/math/xxx)
{{%/* /hint */%}}

## 한 줄 요약
> **핵심 아이디어를 한 문장으로**

## 왜 필요한가?
이 개념이 왜 필요한지 비유와 함께 설명

## 수식
$$
수식
$$

**각 기호의 의미:**
- $x$ : 설명
- $y$ : 설명

### 직관적 이해
수식의 의미를 비유나 그림으로 설명

## 구현
\`\`\`python
# 간단한 코드 예시 (주석 필수)
\`\`\`

## 관련 콘텐츠
- [관련 수학](/ko/docs/math/xxx)
- [이 개념을 사용하는 컴포넌트](/ko/docs/components/xxx)
- [이 개념을 사용하는 아키텍처](/ko/docs/architecture/xxx)
```

**English** (`content/en/docs/math/xxx.md`):
```markdown
---
title: "Concept Name"
weight: 10
math: true
---

# Concept Name

{{%/* hint info */%}}
**Prerequisites**: [Concept1](/en/docs/math/xxx) | [Concept2](/en/docs/math/xxx)
{{%/* /hint */%}}

## One-line Summary
> **Core idea in one sentence**

## Why is this needed?
Explain with analogies

## Formula
$$
formula
$$

**Symbol meanings:**
- $x$ : explanation
- $y$ : explanation

### Intuition
Explain the meaning of the formula with analogies or diagrams

## Implementation
\`\`\`python
# Simple code example (comments required)
\`\`\`

## Related Content
- [Related Math](/en/docs/math/xxx)
- [Components using this concept](/en/docs/components/xxx)
- [Architectures using this concept](/en/docs/architecture/xxx)
```

### components/ 템플릿

math/ 템플릿과 동일한 구조를 사용합니다. 단, 다음이 다릅니다:
- 선수지식에 math/ 링크가 포함됨 (순수 수학에 의존하므로)
- "왜 필요한가?"에서 **이 부품이 어떤 모델에서 쓰이는지** 언급

**한국어** (`content/ko/docs/components/xxx.md`):
```markdown
---
title: "컴포넌트명"
weight: 10
math: true
---

# 컴포넌트명

{{%/* hint info */%}}
**선수지식**: [필요한 수학](/ko/docs/math/xxx) | [필요한 컴포넌트](/ko/docs/components/xxx)
{{%/* /hint */%}}

## 한 줄 요약
> **핵심 아이디어를 한 문장으로**

## 왜 필요한가?
이 부품이 왜 필요한지 비유와 함께 설명
어떤 모델에서 사용되는지 언급

## 수식
$$
수식
$$

**각 기호의 의미:**
- $x$ : 설명
- $y$ : 설명

### 직관적 이해
수식의 의미를 비유나 그림으로 설명

## 구현
\`\`\`python
# 간단한 코드 예시 (주석 필수)
\`\`\`

## 관련 콘텐츠
- [선수 수학](/ko/docs/math/xxx)
- [관련 컴포넌트](/ko/docs/components/xxx)
- [이 컴포넌트를 사용하는 아키텍처](/ko/docs/architecture/xxx)
```

### architecture/ 템플릿

**한국어** (`content/ko/docs/architecture/xxx.md`):
```markdown
---
title: "모델명"
weight: 10
math: true
---

# 모델명

{{%/* hint info */%}}
**선수지식**: [필요한 수학](/ko/docs/math/xxx) | [필요한 컴포넌트](/ko/docs/components/xxx) | [이전 모델](/ko/docs/architecture/xxx)
{{%/* /hint */%}}

## 한 줄 요약
> **이 모델의 핵심 기여**

## 왜 이 모델인가?
- 어떤 문제가 있었고, 어떻게 해결했는지 비유와 함께 설명

## 개요
- **논문**: 논문명 (연도)
- **저자**: 저자명
- **핵심 기여**: 1줄 요약

## 구조
### 전체 아키텍처
{{</* figure src="/images/..." caption="설명" */>}}

### 핵심 컴포넌트
| 컴포넌트 | 역할 | 비유 |
|----------|------|------|
| 컴포넌트1 | 역할 | 일상적 비유 |

## 학습
- Loss 함수 (수식 + 각 기호 설명)
- 학습 방법

## 코드
\`\`\`python
# 핵심 부분 구현 또는 사용법 (주석 필수)
\`\`\`

## 관련 콘텐츠
- [선행 수학](/ko/docs/math/xxx)
- [사용된 컴포넌트](/ko/docs/components/xxx)
- [이전 모델](/ko/docs/architecture/xxx)
- [후속 모델](/ko/docs/architecture/xxx)
- [적용 태스크](/ko/docs/task/xxx)
```

**English** (`content/en/docs/architecture/xxx.md`):
```markdown
---
title: "Model Name"
weight: 10
math: true
---

# Model Name

{{%/* hint info */%}}
**Prerequisites**: [Required Math](/en/docs/math/xxx) | [Required Components](/en/docs/components/xxx) | [Previous Model](/en/docs/architecture/xxx)
{{%/* /hint */%}}

## One-line Summary
> **Key contribution of this model**

## Why this model?
- What problem existed, how it was solved (with analogies)

## Overview
- **Paper**: Paper name (year)
- **Authors**: Author names
- **Key contribution**: One-line summary

## Architecture
### Overall Architecture
{{</* figure src="/images/..." caption="Description" */>}}

### Key Components
| Component | Role | Analogy |
|-----------|------|---------|
| Component1 | Role | Everyday analogy |

## Training
- Loss function (formula + symbol explanations)
- Training method

## Code
\`\`\`python
# Core implementation or usage (comments required)
\`\`\`

## Related Content
- [Prerequisites](/en/docs/math/xxx)
- [Components used](/en/docs/components/xxx)
- [Previous model](/en/docs/architecture/xxx)
- [Subsequent model](/en/docs/architecture/xxx)
- [Applied tasks](/en/docs/task/xxx)
```

### task/ 템플릿

**한국어** (`content/ko/docs/task/xxx.md`):
```markdown
---
title: "태스크명"
weight: 10
---

# 태스크명

## 문제 정의
- 입력: 무엇을 받는가
- 출력: 무엇을 내는가
- 목표: 무엇을 달성하려는가

## 평가 지표
| 지표 | 설명 | 수식 |
|------|------|------|
| 지표1 | 설명 | 수식 |

## 주요 데이터셋
| 데이터셋 | 규모 | 특징 |
|----------|------|------|
| 데이터셋1 | 크기 | 특징 |

## 주요 접근법
1. 접근법1: 간단 설명 → [상세](/ko/docs/architecture/xxx)
2. 접근법2: 간단 설명 → [상세](/ko/docs/architecture/xxx)

## 관련 콘텐츠
- [필요한 수학](/ko/docs/math/xxx)
- [필요한 컴포넌트](/ko/docs/components/xxx)
- [관련 아키텍처](/ko/docs/architecture/xxx)
```

**English** (`content/en/docs/task/xxx.md`):
```markdown
---
title: "Task Name"
weight: 10
---

# Task Name

## Problem Definition
- Input: What it receives
- Output: What it produces
- Goal: What it aims to achieve

## Metrics
| Metric | Description | Formula |
|--------|-------------|---------|
| Metric1 | Description | Formula |

## Major Datasets
| Dataset | Size | Features |
|---------|------|----------|
| Dataset1 | Size | Features |

## Main Approaches
1. Approach1: Brief description → [Details](/en/docs/architecture/xxx)
2. Approach2: Brief description → [Details](/en/docs/architecture/xxx)

## Related Content
- [Required math](/en/docs/math/xxx)
- [Required components](/en/docs/components/xxx)
- [Related architectures](/en/docs/architecture/xxx)
```

### etc/ 템플릿

**한국어** (`content/ko/docs/etc/xxx.md`):
```markdown
---
title: "자료 종류"
weight: 10
---

# 자료 종류

## 카테고리1
| 이름 | 링크 | 설명 |
|------|------|------|
| 자료1 | [링크](url) | 설명 |

## 카테고리2
...
```

**English** (`content/en/docs/etc/xxx.md`):
```markdown
---
title: "Resource Type"
weight: 10
---

# Resource Type

## Category1
| Name | Link | Description |
|------|------|-------------|
| Resource1 | [Link](url) | Description |

## Category2
...
```

## 입구 페이지 수정 규칙

입구 페이지(timeline, topdown, bottomup)는 **순서만** 제공합니다.

### 콘텐츠 추가 시
1. 먼저 `math/`, `components/`, `architecture/`, `task/`, `engineering/`, `etc/` 중 적절한 곳에 **양쪽 언어로** 파일 생성
2. **양쪽 언어의** 입구 페이지에서 적절한 위치에 링크 추가

### 입구 페이지 수정 시
- 순서나 그룹핑만 변경
- 새로운 설명 텍스트 추가하지 않음 (링크만)

## Hugo 프론트매터

```yaml
---
title: "페이지 제목"
weight: 10                    # 정렬 순서 (낮을수록 위)
math: true                    # 수식 사용 시
bookCollapseSection: true     # 하위 페이지 접기 (섹션용)
bookHidden: true              # 메뉴에서 숨기기
---
```

## 링크 문법

Hugo Book 테마에서 내부 링크 (언어 prefix 포함):
```markdown
# 한국어 콘텐츠에서
[표시 텍스트](/ko/docs/카테고리/파일명)

# 영어 콘텐츠에서
[표시 텍스트](/en/docs/카테고리/파일명)
```

예시:
```markdown
# 한국어
[Conv2D](/ko/docs/components/convolution/conv2d)
[ResNet](/ko/docs/architecture/cnn/resnet)
[행렬](/ko/docs/math/linear-algebra/matrix)

# 영어
[Conv2D](/en/docs/components/convolution/conv2d)
[ResNet](/en/docs/architecture/cnn/resnet)
[Matrix](/en/docs/math/linear-algebra/matrix)
```

## 수식 작성 규칙

Hugo 설정에서 `passthrough`가 활성화되어 있어 수식 구분자 내부는 마크다운 파서가 처리하지 않습니다. 따라서 **일반 LaTeX 문법 그대로 사용** 가능합니다.

### 인라인 수식

```markdown
$\mathbb{E}[X]$           # 정상 작동
$P(A) \geq 0$             # 정상 작동
$\sum_i x_i$              # 정상 작동
```

### 디스플레이 수식

```markdown
$$
\mathbb{E}[X] = \sum_x x \cdot P(X=x)
$$
```

### 주의: 중괄호 이스케이프

집합 표기 등에서 LaTeX 문법상 중괄호는 이스케이프가 필요합니다 (Hugo가 아닌 LaTeX 자체 규칙):

```markdown
$\{A_i\}$                 # 정상 작동 (LaTeX 문법)
```

---

## 체크리스트

콘텐츠 작성 후 확인:
- [ ] 올바른 디렉토리에 파일을 만들었는가? (math vs components vs architecture 구분)
- [ ] **한국어와 영어 양쪽에** 파일을 만들었는가?
- [ ] 프론트매터가 올바른가?
- [ ] **STRUCTURE.md를 참고하여 선수지식을 작성했는가?**
- [ ] 입구→콘텐츠, 콘텐츠→콘텐츠 링크만 있는가?
- [ ] 입구→입구, 콘텐츠→입구 링크가 없는가?
- [ ] **양쪽 언어의** 관련 입구 페이지에 링크를 추가했는가?
- [ ] **수식이 올바른 LaTeX 문법으로 작성되었는가?**
- [ ] **나노바나나 프롬프트를 작성했는가?** (시각 자료가 필요한 경우)

## 참고

전체 구조와 콘텐츠 목록은 [STRUCTURE.md](./STRUCTURE.md) 참조
