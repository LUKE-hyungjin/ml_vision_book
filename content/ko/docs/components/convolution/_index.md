---
title: "합성곱"
weight: 1
bookCollapseSection: true
math: true
---

# 합성곱 (Convolution)

{{% hint info %}}
**선수지식**: [행렬](/ko/docs/math/linear-algebra/matrix)
{{% /hint %}}

> **한 줄 요약**: Convolution은 작은 필터를 공유하며 이미지의 **지역적 특징을 추출**하는 CNN의 핵심 연산입니다.

## 왜 Convolution인가?

이미지를 이해하려면 모든 픽셀을 한꺼번에 볼 필요 없이, **작은 영역의 패턴**(에지, 텍스처, 형태)을 먼저 찾고, 이를 조합하면 됩니다.

Convolution의 3가지 핵심 성질:
1. **파라미터 공유** — 동일 필터를 이미지 전체에 적용 → 파라미터 수 대폭 감소
2. **지역적 연결** — 인접 픽셀끼리만 연결 → 공간 구조 보존
3. **이동 불변성** — 어디에 있든 같은 패턴을 감지

## 학습 순서

| 순서 | 개념 | 핵심 질문 | 딥러닝에서의 역할 |
|------|------|----------|------------------|
| 1 | [Conv2D](/ko/docs/components/convolution/conv2d) | 필터가 특징을 어떻게 추출하나? | CNN의 기본 빌딩 블록 |
| 2 | [Pooling](/ko/docs/components/convolution/pooling) | 공간을 줄이면서 정보를 보존하려면? | 다운샘플링, 위치 불변성 |
| 3 | [Receptive Field](/ko/docs/components/convolution/receptive-field) | 출력 하나가 입력의 얼마를 보나? | 모델 설계의 핵심 기준 |
| 4 | [Transposed Conv](/ko/docs/components/convolution/transposed-conv) | 줄인 걸 다시 키우려면? | 디코더, 생성 모델 |

## 관련 콘텐츠

- [행렬](/ko/docs/math/linear-algebra/matrix) — 합성곱의 행렬 표현
- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — Conv 뒤에 함께 사용
- [AlexNet](/ko/docs/architecture/cnn/alexnet) — Conv를 깊게 쌓은 최초의 모델
- [ResNet](/ko/docs/architecture/cnn/resnet) — 현대 CNN의 표준
