---
title: "VLM"
weight: 2
math: true
---

# VLM (Vision-Language Models)

## 개요

Vision-Language Model은 이미지와 텍스트를 함께 이해하고 생성하는 모델입니다.

## 발전 과정

```
CLIP (2021) → BLIP (2022) → BLIP-2 (2023) → LLaVA (2023) → GPT-4V (2023)
```

---

## 주요 모델들

### 1. BLIP (Bootstrapping Language-Image Pre-training)

#### 개요

- **논문**: BLIP: Bootstrapping Language-Image Pre-training (2022)
- **저자**: Salesforce Research
- **핵심**: 캡션 생성과 필터링을 통한 데이터 품질 향상

#### 구조

```
          Image ─────→ Image Encoder ─────┐
                                           ├──→ Multi-task Learning
          Text  ─────→ Text Encoder  ─────┘

Tasks: ITC (Image-Text Contrastive)
       ITM (Image-Text Matching)
       LM  (Language Modeling)
```

---

### 2. BLIP-2

#### 개요

- **논문**: BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs (2023)
- **저자**: Salesforce Research
- **핵심**: 사전학습된 Vision/Language 모델을 효율적으로 연결

#### 구조

```
┌─────────────────────────────────────────────────────────┐
│                       BLIP-2                             │
│                                                          │
│   Image ──→ [Frozen Image Encoder] ──→ Image Features   │
│                                              ↓           │
│                                    ┌─────────────────┐  │
│                                    │    Q-Former     │  │
│                                    │ (Learnable)     │  │
│                                    │ 32 Query Tokens │  │
│                                    └────────┬────────┘  │
│                                              ↓           │
│   Text ───────────────────────→ [Frozen LLM] ──→ Output │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Q-Former

Vision과 Language 모델을 연결하는 경량 모듈:

```python
# Q-Former: 32개의 학습 가능한 쿼리
query_tokens = nn.Parameter(torch.zeros(1, 32, hidden_dim))

# Image features에서 정보 추출
image_features = frozen_vit(image)
query_output = q_former(query_tokens, image_features)

# LLM에 입력
llm_input = [query_output, text_tokens]
output = frozen_llm(llm_input)
```

---

### 3. LLaVA (Large Language and Vision Assistant)

#### 개요

- **논문**: Visual Instruction Tuning (2023)
- **저자**: Microsoft Research
- **핵심**: Visual instruction tuning으로 GPT-4 수준의 대화 능력

#### 구조

```
┌─────────────────────────────────────────────────────────┐
│                       LLaVA                              │
│                                                          │
│   Image ──→ CLIP ViT ──→ Linear Projection ─┐           │
│                                              ↓           │
│   "Describe" ──→ Tokenizer ──→ LLM (Vicuna) ──→ Caption │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### 학습 단계

| 단계 | 데이터 | 목적 |
|------|--------|------|
| Stage 1 | CC3M (필터링) | Vision-Language 정렬 |
| Stage 2 | LLaVA-Instruct-150K | Instruction following |

#### 구현 예시

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 이미지 + 질문
inputs = processor(
    text="<image>\nWhat is shown in this image?",
    images=image,
    return_tensors="pt"
)

output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

### 4. GPT-4V (GPT-4 with Vision)

#### 특징

- OpenAI의 멀티모달 모델
- 이미지 이해 + 텍스트 생성
- 복잡한 추론 능력

#### 사용 예시

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지에서 무엇이 보이나요?"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
)
```

---

## VLM 아키텍처 비교

| 모델 | Vision Encoder | Language Model | 연결 방식 |
|------|---------------|----------------|-----------|
| CLIP | ViT | Transformer | Contrastive |
| BLIP | ViT | BERT/GPT | Multi-task |
| BLIP-2 | Frozen ViT | Frozen LLM | Q-Former |
| LLaVA | CLIP ViT | Vicuna/LLaMA | Linear |
| GPT-4V | 비공개 | GPT-4 | 비공개 |

---

## 학습 방법

### 1. Contrastive Learning

이미지-텍스트 쌍의 유사도 학습:

$$L_{ITC} = -\log \frac{\exp(sim(I, T^+)/\tau)}{\sum_j \exp(sim(I, T_j)/\tau)}$$

### 2. Image-Text Matching

이미지와 텍스트가 매칭되는지 분류:

$$L_{ITM} = \text{CrossEntropy}(p_{match}, y)$$

### 3. Language Modeling

이미지 조건부 텍스트 생성:

$$L_{LM} = -\sum_t \log P(w_t | w_{<t}, I)$$

---

## VLM의 능력

### 1. Visual Question Answering (VQA)

```
Image: [고양이 사진]
Q: "이 동물의 색깔은?"
A: "주황색과 흰색 줄무늬입니다."
```

### 2. Image Captioning

```
Image: [해변 사진]
Caption: "맑은 날 파란 하늘 아래 모래사장에서 사람들이 휴식을 취하고 있습니다."
```

### 3. Visual Reasoning

```
Image: [수학 문제 사진]
Q: "이 문제를 풀어주세요."
A: "주어진 방정식을 정리하면... 답은 x = 5입니다."
```

### 4. OCR + 이해

```
Image: [메뉴판 사진]
Q: "가장 비싼 메뉴는?"
A: "스테이크 35,000원이 가장 비쌉니다."
```

---

## 최신 동향

### 오픈소스 모델들

| 모델 | 파라미터 | 특징 |
|------|----------|------|
| LLaVA-1.5 | 7B/13B | 효율적인 학습 |
| InternVL | 6B-25B | 강력한 성능 |
| Qwen-VL | 7B | 다국어 지원 |
| CogVLM | 17B | 세밀한 이해 |

### 발전 방향

1. **고해상도 처리**: 더 세밀한 이미지 이해
2. **비디오 이해**: 시간적 정보 처리
3. **다중 이미지**: 여러 이미지 동시 처리
4. **Grounding**: 객체 위치 지정

---

## 구현 팁

### 효율적인 추론

```python
# 4-bit 양자화로 메모리 절약
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=bnb_config
)
```

### 배치 처리

```python
# 여러 이미지 동시 처리
inputs = processor(
    text=["<image>\nDescribe this."] * len(images),
    images=images,
    return_tensors="pt",
    padding=True
)
```

---

## 관련 콘텐츠

- [CLIP](/ko/docs/architecture/multimodal/clip) - 기반 모델
- [ViT](/ko/docs/architecture/transformer/vit) - Vision encoder
- [Transformer](/ko/docs/architecture/transformer) - 기본 아키텍처

