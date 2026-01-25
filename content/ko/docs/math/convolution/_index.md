---
title: "합성곱"
weight: 4
bookCollapseSection: true
math: true
---

# 합성곱 (Convolution)

CNN의 핵심 연산으로, 이미지의 지역적 특징을 추출합니다.

## 왜 Convolution인가?

이미지 처리에서 Fully Connected Layer의 문제:
- **파라미터 폭발**: 224×224×3 이미지 → 150,528개 입력
- **공간 정보 손실**: 픽셀 위치 관계 무시
- **이동 불변성 없음**: 같은 패턴이 다른 위치에 있으면 인식 못함

Convolution의 장점:
- **파라미터 공유**: 동일 필터를 전체 이미지에 적용
- **지역적 연결**: 인접 픽셀만 연결
- **이동 불변성**: 같은 패턴 → 같은 반응

## 핵심 개념

| 개념 | 설명 | 중요성 |
|------|------|--------|
| [Conv2D](/ko/docs/math/convolution/conv2d) | 2D 합성곱 연산 | CNN의 기본 연산 |
| [Pooling](/ko/docs/math/convolution/pooling) | 다운샘플링 | 불변성, 연산량 감소 |
| [Receptive Field](/ko/docs/math/convolution/receptive-field) | 수용 영역 | 컨텍스트 범위 |
| [Transposed Conv](/ko/docs/math/convolution/transposed-conv) | 업샘플링 | 디코더, 생성 모델 |

## 관련 콘텐츠

- [행렬](/ko/docs/math/linear-algebra/matrix) - 합성곱의 행렬 표현
- [Batch Normalization](/ko/docs/math/normalization/batch-norm) - CNN에서 자주 사용
