---
title: "Qwen Image Edit"
weight: 5
math: true
---

# Qwen Image Edit

{{% hint info %}}
**선수지식**: [Diffusion 기초](/ko/docs/math/generative/ddpm) | [VLM (Vision-Language Model)](/ko/docs/architecture/multimodal/vlm) | [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion)
{{% /hint %}}

## 한 줄 요약

> **"자연어로 이미지를 편집한다 - 말로 그림을 고친다!"**

---

## 왜 Qwen Image Edit인가?

> **비유**: 포토샵을 배우려면 복잡한 도구를 익혀야 합니다. 하지만 "이 사람 머리색을 금발로 바꿔줘"라고 **말만 하면** 알아서 바꿔준다면?

**기존 문제:**
- 이미지 편집 도구 = 전문 기술 필요
- Inpainting = 마스크를 직접 그려야 함

**Qwen Image Edit 해결:**
- 자연어 명령으로 편집
- 편집 영역 자동 인식
- 원본 스타일 유지

---

## 개요

- **모델**: Qwen2-VL 기반 이미지 편집 모델
- **특징**: 멀티모달 이해 + Diffusion 생성
- **핵심 기여**: 언어 이해력으로 정밀한 이미지 편집

---

## 구조

### 전체 아키텍처

```
입력 이미지 + 편집 명령 ("배경을 해변으로 바꿔줘")
                    ↓
┌─────────────────────────────────────────────────┐
│              Qwen2-VL (이해)                     │
│  - 이미지 이해: "현재 배경은 도시"                 │
│  - 명령 이해: "해변으로 바꿔야 함"                 │
│  - 편집 영역 파악: "배경 부분만"                   │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│           Diffusion (생성/편집)                  │
│  - 편집 영역에 노이즈 추가                        │
│  - 조건(해변)에 맞게 복원                         │
│  - 비편집 영역은 유지                            │
└─────────────────────────────────────────────────┘
                    ↓
              편집된 이미지
```

### 핵심 컴포넌트

| 컴포넌트 | 역할 | 비유 |
|----------|------|------|
| **Qwen2-VL** | 이미지+텍스트 이해 | "통역사" - 명령을 이해 |
| **편집 영역 추론** | 어디를 바꿀지 결정 | "기획자" - 작업 범위 설정 |
| **Diffusion** | 실제 이미지 생성 | "화가" - 그림 수정 |
| **Blending** | 편집/비편집 영역 합성 | "편집자" - 자연스럽게 합치기 |

---

## 작동 방식

### 1단계: 이미지 이해

```
입력: 도시 배경의 사람 사진
Qwen2-VL 분석:
  - 전경: 사람 (보존해야 함)
  - 배경: 도시 건물들 (편집 대상)
  - 조명: 낮, 자연광
```

### 2단계: 명령 해석

```
명령: "배경을 해변으로 바꿔줘"
해석:
  - 동작: 교체 (replace)
  - 대상: 배경
  - 목표: 해변
  - 제약: 사람은 유지, 조명 일관성
```

### 3단계: 편집 실행

```
1. 배경 영역에 노이즈 추가 (Forward)
2. "해변" 조건으로 복원 (Reverse)
3. 사람 영역과 자연스럽게 블렌딩
```

---

## 편집 유형

| 유형 | 예시 명령 | 설명 |
|------|----------|------|
| **교체** | "배경을 산으로 바꿔줘" | 특정 영역을 다른 것으로 |
| **추가** | "선글라스를 씌워줘" | 새로운 객체 추가 |
| **제거** | "뒤에 있는 사람 지워줘" | 원치 않는 객체 제거 |
| **스타일** | "유화 스타일로 바꿔줘" | 전체 스타일 변환 |
| **속성** | "머리색을 금발로" | 객체 속성 변경 |

---

## Inpainting과의 차이

| | 기존 Inpainting | Qwen Image Edit |
|---|----------------|-----------------|
| **마스크** | 사용자가 직접 그림 | 자동 추론 |
| **명령** | 없음 (프롬프트만) | 자연어 명령 |
| **이해** | 이미지만 | 이미지 + 컨텍스트 |
| **정밀도** | 마스크 의존 | 의미적 이해 |

---

## 구현 예시

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 모델 로드
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 이미지 편집 요청
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "photo.jpg"},
            {"type": "text", "text": "배경을 해변으로 바꿔줘"}
        ]
    }
]

# 처리
inputs = processor(messages, return_tensors="pt")
outputs = model.generate(**inputs)

# 결과: 편집된 이미지 또는 편집 지시사항
```

---

## 장단점

### 장점

1. **직관적**: 자연어로 편집 가능
2. **마스크 불필요**: 편집 영역 자동 인식
3. **컨텍스트 이해**: 이미지 전체 맥락 고려
4. **일관성**: 스타일/조명 자동 매칭

### 단점

1. **정밀도**: 픽셀 단위 정교한 편집은 어려움
2. **예측 불가**: 결과가 기대와 다를 수 있음
3. **계산량**: VLM + Diffusion 조합으로 무거움

---

## 관련 콘텐츠

- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - 기반 Diffusion 모델
- [ControlNet](/ko/docs/architecture/generative/controlnet) - 조건부 이미지 생성
- [VLM](/ko/docs/architecture/multimodal/vlm) - Vision-Language Model
- [Diffusion 수학](/ko/docs/math/generative/ddpm) - Forward/Reverse Process
