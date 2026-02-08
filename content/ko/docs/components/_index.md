---
title: "컴포넌트"
weight: 15
bookCollapseSection: true
---

# 딥러닝 빌딩 블록

모델을 만들 때 사용하는 부품과 기법입니다.

## 핵심 연산

| 카테고리 | 설명 |
|----------|------|
| [합성곱](/ko/docs/components/convolution) | Conv2D, Pooling, Receptive Field |
| [어텐션](/ko/docs/components/attention) | Self-Attention, Cross-Attention |
| [정규화](/ko/docs/components/normalization) | BatchNorm, LayerNorm, RMSNorm |

## 학습

| 카테고리 | 설명 |
|----------|------|
| [손실 함수](/ko/docs/components/training/loss) | Cross-Entropy, Focal Loss, Contrastive Loss |
| [최적화](/ko/docs/components/training/optimizer) | SGD, Adam, LR Scheduler |
| [정규화 기법](/ko/docs/components/training/regularization) | Dropout, Weight Decay, Label Smoothing |
| [PEFT](/ko/docs/components/training/peft) | LoRA, QLoRA, Adapter |

## 도메인별

| 카테고리 | 설명 |
|----------|------|
| [Detection 연산](/ko/docs/components/detection) | IoU, NMS, Anchor |
| [생성 모델 수학](/ko/docs/components/generative) | DDPM 수학, Score Matching, Flow Matching |
| [양자화](/ko/docs/components/quantization) | Data Types, PTQ, QAT |
