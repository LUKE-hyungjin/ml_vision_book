---
title: "SIFT & HOG"
weight: 1
math: true
---

# SIFT & HOG

## 개요

딥러닝 이전 시대의 대표적인 특징 추출 기법입니다.

| 기법 | 연도 | 핵심 아이디어 |
|------|------|--------------|
| SIFT | 2004 | 스케일 불변 키포인트 검출 |
| HOG | 2005 | 그래디언트 방향 히스토그램 |

---

## SIFT (Scale-Invariant Feature Transform)

### 핵심 아이디어

이미지의 크기나 회전에 관계없이 동일한 특징점을 찾아내는 알고리즘입니다.

### 동작 과정

1. **Scale-space 생성**: Gaussian blur를 다양한 스케일로 적용
2. **DoG (Difference of Gaussian)**: 인접한 스케일 간 차이 계산
3. **키포인트 검출**: DoG에서 극값(extrema) 찾기
4. **방향 할당**: 키포인트 주변의 그래디언트 방향 계산
5. **디스크립터 생성**: 128차원 벡터로 특징 표현

### 수식

Gaussian blur:
$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

Difference of Gaussian:
$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)$$

### 특징

- **장점**: 스케일, 회전, 조명 변화에 강건
- **단점**: 계산 비용이 높음, 실시간 처리 어려움

---

## HOG (Histogram of Oriented Gradients)

### 핵심 아이디어

이미지를 셀 단위로 나누고, 각 셀에서 그래디언트 방향의 히스토그램을 계산합니다.

### 동작 과정

1. **그래디언트 계산**: 각 픽셀에서 x, y 방향 그래디언트 계산
2. **셀 분할**: 이미지를 8x8 픽셀 셀로 분할
3. **히스토그램 생성**: 각 셀에서 9개 방향(0°~180°)의 히스토그램 계산
4. **블록 정규화**: 2x2 셀을 하나의 블록으로 묶어 정규화
5. **특징 벡터 생성**: 모든 블록의 히스토그램을 연결

### 수식

그래디언트 크기와 방향:
$$G = \sqrt{G_x^2 + G_y^2}$$
$$\theta = \arctan\left(\frac{G_y}{G_x}\right)$$

### 특징

- **장점**: 조명 변화에 강건, 계산이 상대적으로 빠름
- **단점**: 스케일 변화에 민감
- **주요 용도**: 보행자 검출 (Dalal & Triggs, 2005)

---

## 구현 예시

```python
import cv2

# SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)

# HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(image)
```

---

## 딥러닝과의 비교

| 측면 | SIFT/HOG | CNN |
|------|----------|-----|
| 특징 설계 | 수작업 | 학습 기반 |
| 데이터 요구량 | 적음 | 많음 |
| 성능 | 제한적 | 우수 |
| 해석 가능성 | 높음 | 낮음 |
| 일반화 | 제한적 | 우수 |

현재는 대부분의 태스크에서 딥러닝이 SIFT/HOG를 대체했지만, 특징 매칭이나 엣지 케이스에서는 여전히 사용됩니다.

---

## 관련 콘텐츠

- [선형대수](/ko/docs/math/linear-algebra) - 그래디언트 계산의 기초
- [CNN 기초](/ko/docs/architecture/cnn) - SIFT/HOG를 대체한 딥러닝 기법
