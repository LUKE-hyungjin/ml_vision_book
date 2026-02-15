---
title: "VLM"
weight: 2
math: true
---

# VLM (Vision-Language Models)

{{% hint info %}}
**Prerequisites**: [CLIP](/en/docs/architecture/multimodal/clip) | [ViT](/en/docs/architecture/transformer/vit) | [Self-Attention](/en/docs/components/attention/self-attention)
{{% /hint %}}

## One-line Summary
> **A VLM maps images into language-friendly features, then lets an LLM answer or generate text grounded on those visual features.**

## Why is this needed?
A text-only LLM cannot directly read pixels. In practice we need a model that can:
- describe an image,
- answer visual questions,
- reason over charts/UI/documents,
- connect OCR-like reading with language reasoning.

VLMs solve this by combining a **vision encoder** (for pixels) and a **language model** (for text generation).

## Core pipeline (beginner view)
1. **Vision encoding**: image \(\rightarrow\) visual tokens/features
2. **Modality bridge**: align visual features to the LLM token space (projection/Q-Former/cross-attention)
3. **Language decoding**: generate answer conditioned on image + prompt

Think of it as: camera notes \(\rightarrow\) translator \(\rightarrow\) writer.

## Typical architectures

| Family | Vision side | Bridge | Language side | Example |
|---|---|---|---|---|
| Contrastive pretraining | ViT | shared embedding space | text encoder | CLIP |
| Frozen backbone + lightweight bridge | frozen ViT | Q-Former | frozen LLM | BLIP-2 |
| Instruction-tuned VLM | CLIP ViT | linear projector | Vicuna/LLaMA-like | LLaVA |

## Key training objectives

### 1) Image-Text Contrastive (ITC)
$$
L_{ITC} = -\log \frac{\exp(\mathrm{sim}(I, T^+)/\tau)}{\sum_j \exp(\mathrm{sim}(I, T_j)/\tau)}
$$

**Symbols:**
- \(I\): image embedding
- \(T^+\): matched text embedding
- \(T_j\): candidate text embeddings
- \(\mathrm{sim}(\cdot,\cdot)\): similarity (often cosine)
- \(\tau\): temperature

### 2) Image-Text Matching (ITM)
$$
L_{ITM} = \mathrm{CrossEntropy}(p_{match}, y)
$$

- \(p_{match}\): predicted probability that image and text match
- \(y\): binary label (match/non-match)

### 3) Language Modeling (LM)
$$
L_{LM} = -\sum_t \log P(w_t \mid w_{<t}, I)
$$

- \(w_t\): next token to predict
- \(I\): visual context

## Minimal implementation example (LLaVA-style inference)
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "<image>\nWhat is the main object and what is it doing?"
inputs = processor(text=prompt, images=image, return_tensors="pt")

output_ids = model.generate(**inputs, max_new_tokens=120)
answer = processor.decode(output_ids[0], skip_special_tokens=True)
print(answer)
```

## Practical debugging checklist
- [ ] **Image token injection check**: did you include image tokens/placeholders in prompt format required by the model?
- [ ] **Resolution mismatch**: does your processor/model pair expect a fixed image size?
- [ ] **Hallucination control**: when uncertain, does prompting require evidence-based answers ("say unknown if unclear")?
- [ ] **OCR-heavy tasks**: is input resolution high enough for small text?
- [ ] **Latency/memory**: test 4-bit quantization before reducing max tokens too aggressively.

## Common mistakes (FAQ)
**Q1. Is a bigger LLM always better for vision tasks?**  
A. Not always. If the visual bridge or image resolution is weak, larger language capacity alone does not fix grounding.

**Q2. Why does VLM answer confidently but wrongly?**  
A. Often due to weak visual grounding or prompt setup. Add "answer only from visible evidence" and verify preprocessing.

**Q3. Can VLM replace OCR completely?**  
A. For clean large text, often yes. For dense/small text documents, dedicated OCR + VLM reasoning is still common.

## Where VLMs are used
- Visual Question Answering (VQA)
- Image captioning
- Document/UI understanding
- Multimodal assistants and agents

## Related Content
- [CLIP](/en/docs/architecture/multimodal/clip)
- [ViT](/en/docs/architecture/transformer/vit)
- [Self-Attention](/en/docs/components/attention/self-attention)
- [Cross-Attention](/en/docs/components/attention/cross-attention)
