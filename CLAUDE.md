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
│       ├── math/                    # 콘텐츠: 수학
│       ├── architecture/            # 콘텐츠: 아키텍처
│       ├── task/                    # 콘텐츠: 태스크
│       └── etc/                     # 콘텐츠: 기타 자료
└── en/                              # English
    ├── _index.md
    ├── menu/index.md
    └── docs/
        ├── timeline/_index.md
        ├── topdown/_index.md
        ├── bottomup/_index.md
        ├── math/
        ├── architecture/
        ├── task/
        └── etc/
```

## i18n 규칙

### 콘텐츠 추가 시
1. **양쪽 언어에 동일한 파일 경로로 생성**
   - 한국어: `content/ko/docs/math/convolution.md`
   - 영어: `content/en/docs/math/convolution.md`

2. **파일명은 영어로 통일** (URL 일관성)
   - O: `linear-algebra.md`
   - X: `선형대수.md`

3. **링크는 언어 prefix 포함하여 작성**
   ```markdown
   # 한국어 콘텐츠에서
   [Convolution](/ko/docs/math/convolution)

   # 영어 콘텐츠에서
   [Convolution](/en/docs/math/convolution)
   ```

### 번역 우선순위
1. 한국어 먼저 작성
2. 영어 버전 생성 (구조는 동일, 내용만 번역)

## 핵심 원칙

### 링크 규칙 (절대 위반 금지)

| 출발 | 도착 | 허용 |
|------|------|------|
| 입구 (timeline, topdown, bottomup) | 콘텐츠 (math, architecture, task, etc) | O |
| 콘텐츠 | 콘텐츠 | O |
| 입구 | 입구 | **X** |
| 콘텐츠 | 입구 | **X** |

### 콘텐츠 분류 기준

| 디렉토리 | 기준 | 판단 질문 |
|----------|------|----------|
| `math/` | 수식이 핵심 | "이 개념을 설명하려면 수식이 필수인가?" |
| `architecture/` | 모델/네트워크 구조 | "이건 특정 모델의 구조를 설명하는가?" |
| `task/` | 문제 정의 + 평가 + 데이터셋 | "이건 '무엇을 푸는가'에 대한 설명인가?" |
| `etc/` | 외부 링크/자료 | "이건 외부 자료 모음인가?" |

### Math 하위 분류

| 디렉토리 | 내용 | 예시 |
|----------|------|------|
| `math/linear-algebra/` | 선형대수 기초 | matrix, vector, eigenvalue, SVD |
| `math/calculus/` | 미적분/최적화 | gradient, backprop, chain rule |
| `math/probability/` | 확률/통계 | bayes, distribution, sampling |
| `math/convolution/` | 합성곱 관련 | conv2d, pooling, receptive field |
| `math/attention/` | 어텐션 관련 | self-attention, cross-attention, positional encoding |
| `math/normalization/` | 정규화 기법 | batch norm, layer norm, RMSNorm |
| `math/training/loss/` | 손실 함수 | cross-entropy, focal loss, contrastive loss |
| `math/training/optimizer/` | 최적화 알고리즘 | SGD, Adam, AdamW, LR scheduler |
| `math/training/regularization/` | 정규화 기법 | dropout, weight decay, label smoothing |
| `math/training/peft/` | 파라미터 효율적 학습 | LoRA, QLoRA, adapter, prefix tuning |
| `math/geometry/` | 기하학 | camera model, homography |
| `math/detection/` | Detection 수학 | IoU, NMS, anchor box |
| `math/diffusion/` | Diffusion 수학 | DDPM, score matching |
| `math/quantization/` | 양자화 | PTQ, QAT |

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

## 개요
이 개념이 왜 필요한지 1-2문장으로 설명

## 수식
$$
수식
$$

### 직관적 이해
수식의 의미를 비유나 그림으로 설명

## 구현
\`\`\`python
# 간단한 코드 예시
\`\`\`

## 관련 콘텐츠
- [관련 수학](/ko/docs/math/xxx)
- [이 개념을 사용하는 아키텍처](/ko/docs/architecture/xxx)
- [이 개념이 쓰이는 태스크](/ko/docs/task/xxx)
```

**English** (`content/en/docs/math/xxx.md`):
```markdown
---
title: "Concept Name"
weight: 10
math: true
---

# Concept Name

## Overview
1-2 sentences explaining why this concept is needed

## Formula
$$
formula
$$

### Intuition
Explain the meaning of the formula with analogies or diagrams

## Implementation
\`\`\`python
# Simple code example
\`\`\`

## Related Content
- [Related Math](/en/docs/math/xxx)
- [Architectures using this concept](/en/docs/architecture/xxx)
- [Tasks using this concept](/en/docs/task/xxx)
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

## 개요
- 언제 나왔는지
- 어떤 문제를 해결했는지
- 핵심 기여 1줄 요약

## 구조
### 전체 아키텍처
(다이어그램 또는 설명)

### 핵심 컴포넌트
각 컴포넌트 설명

## 학습
- Loss 함수
- 학습 방법

## 코드
\`\`\`python
# 핵심 부분 구현 또는 사용법
\`\`\`

## 관련 콘텐츠
- [선행 지식](/ko/docs/math/xxx)
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

## Overview
- When it was released
- What problem it solved
- Key contribution in one line

## Architecture
### Overall Architecture
(Diagram or description)

### Key Components
Description of each component

## Training
- Loss function
- Training method

## Code
\`\`\`python
# Core implementation or usage
\`\`\`

## Related Content
- [Prerequisites](/en/docs/math/xxx)
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
1. Approach1: Brief description → [Details](/docs/architecture/xxx)
2. Approach2: Brief description → [Details](/docs/architecture/xxx)

## Related Content
- [Required math](/docs/math/xxx)
- [Related architectures](/docs/architecture/xxx)
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
1. 먼저 `math/`, `architecture/`, `task/`, `etc/` 중 적절한 곳에 **양쪽 언어로** 파일 생성
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

Hugo Book 테마에서 내부 링크 (언어 prefix 없이):
```markdown
[표시 텍스트](/docs/카테고리/파일명)
```

예시:
```markdown
[Convolution](/docs/math/convolution)
[ResNet](/docs/architecture/resnet)
[Detection](/docs/task/detection)
```

## 체크리스트

콘텐츠 작성 후 확인:
- [ ] 올바른 디렉토리에 파일을 만들었는가?
- [ ] **한국어와 영어 양쪽에** 파일을 만들었는가?
- [ ] 프론트매터가 올바른가?
- [ ] 입구→콘텐츠, 콘텐츠→콘텐츠 링크만 있는가?
- [ ] 입구→입구, 콘텐츠→입구 링크가 없는가?
- [ ] **양쪽 언어의** 관련 입구 페이지에 링크를 추가했는가?

## 참고

전체 구조와 콘텐츠 목록은 [STRUCTURE.md](./STRUCTURE.md) 참조
