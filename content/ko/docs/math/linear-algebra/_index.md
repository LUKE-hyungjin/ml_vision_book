---
title: "선형대수"
weight: 1
bookCollapseSection: true
math: true
---

# 선형대수 (Linear Algebra)

딥러닝의 모든 연산은 선형대수 위에서 이루어집니다. 이미지, 텍스트, 모델 파라미터 모두 행렬과 벡터로 표현됩니다.

## 왜 선형대수인가?

- **이미지**: H x W x C 텐서 (3차원 배열)
- **배치 데이터**: B x H x W x C 텐서
- **모델 가중치**: 행렬 연산의 연속
- **Attention**: Query, Key, Value 모두 행렬 연산

## 핵심 개념

| 개념 | 설명 | 딥러닝 적용 |
|------|------|------------|
| [행렬 연산](/ko/docs/math/linear-algebra/matrix) | 덧셈, 곱셈, 전치 | FC Layer, Conv |
| [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue) | 행렬의 본질적 특성 | PCA, 안정성 분석 |
| [SVD](/ko/docs/math/linear-algebra/svd) | 행렬 분해 | 압축, LoRA |

## 관련 콘텐츠

- [Convolution](/ko/docs/math/convolution) - 행렬 연산의 특수한 형태
- [Attention](/ko/docs/math/attention) - QKV 행렬 연산
- [LoRA](/ko/docs/math/training/peft/lora) - 저랭크 행렬 분해 활용
