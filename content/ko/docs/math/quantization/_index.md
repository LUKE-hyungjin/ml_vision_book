---
title: "양자화"
weight: 10
bookCollapseSection: true
math: true
---

# 양자화 (Quantization)

모델 가중치와 활성화를 낮은 비트로 표현하여 효율성을 높입니다.

## 왜 양자화인가?

| 비트 | 메모리 (7B 모델) | 속도 |
|------|-----------------|------|
| FP32 | 28GB | 기준 |
| FP16 | 14GB | ~2x |
| INT8 | 7GB | ~2-4x |
| INT4 | 3.5GB | ~4-8x |

## 핵심 개념

| 개념 | 설명 |
|------|------|
| [PTQ](/ko/docs/math/quantization/ptq) | 학습 후 양자화 |
| [QAT](/ko/docs/math/quantization/qat) | 양자화 인식 학습 |
| [Data Types](/ko/docs/math/quantization/data-types) | FP16, BF16, INT8 등 |

## 기본 수식

$$
Q(x) = \text{round}\left(\frac{x}{s}\right) + z
$$

- s: 스케일 (scale)
- z: 영점 (zero point)

## 관련 콘텐츠

- [QLoRA](/ko/docs/math/training/peft/qlora)
- [LoRA](/ko/docs/math/training/peft/lora)
