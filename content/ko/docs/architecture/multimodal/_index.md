---
title: "Multimodal"
weight: 7
bookCollapseSection: true
---

# Multimodal 모델

이미지와 텍스트 등 여러 모달리티를 함께 이해하는 모델들입니다.

## 발전 과정

```
2021: CLIP - 이미지-텍스트 contrastive learning
  ↓
2022: Flamingo - Few-shot visual reasoning
  ↓
2023: LLaVA, GPT-4V - Vision Language Model
  ↓
2024: 더 강력한 VLM들 (Claude 3, Gemini)
```

---

## 주요 모델

| 모델 | 유형 | 특징 |
|------|------|------|
| [CLIP](/ko/docs/architecture/multimodal/clip) | Encoder | 이미지-텍스트 임베딩 정렬 |
| [VLM](/ko/docs/architecture/multimodal/vlm) | LLM 기반 | 이미지 이해 + 텍스트 생성 |

---

## 두 가지 접근

### Dual Encoder (CLIP 방식)

```
Image → Image Encoder → Image Embedding
                                ↓ (cosine similarity)
Text  → Text Encoder  → Text Embedding
```

- 빠른 검색/매칭
- Zero-shot classification

### LLM 기반 (VLM 방식)

```
Image → Vision Encoder → Visual Tokens
                              ↓
Text  ───────────────────────→ LLM → Response
```

- 복잡한 추론 가능
- 자연어로 답변

---

## 관련 콘텐츠

- [Contrastive Learning](/ko/docs/math/contrastive) - CLIP의 수학적 기초
- [ViT](/ko/docs/architecture/transformer/vit) - 이미지 인코더
- [Vision-Language 태스크](/ko/docs/task/vision-language) - VQA, Captioning
