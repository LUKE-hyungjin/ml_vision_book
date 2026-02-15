---
title: "SIFT & HOG"
weight: 1
math: true
---

# SIFT & HOG

{{% hint info %}}
**선수지식**: [행렬](/ko/docs/math/linear-algebra/matrix) | [Gradient](/ko/docs/math/calculus/gradient) | [Conv2D](/ko/docs/components/convolution/conv2d)
{{% /hint %}}

## 한 줄 요약
> **SIFT와 HOG는 딥러닝 이전에 널리 쓰인 고전 특징 추출기로, 이미지를 ‘학습 가능한 숫자 특징’으로 바꿔주는 방법입니다.**

## 왜 필요한가?
CNN이 없던 시절에도 물체를 찾고 이미지를 매칭해야 했습니다.
SIFT/HOG는 사람 손으로 설계한 규칙으로 안정적인 시각 단서를 뽑아, 분류기/검출기가 최종 판단하게 했습니다.

비유:
- **SIFT**: 지도에서 확대/회전해도 찾기 쉬운 랜드마크 찍기
- **HOG**: 지역별로 “엣지가 어느 방향으로 많은지” 요약하기

지금도 다음 상황에서 유용합니다.
- 데이터가 매우 적어 딥러닝이 과적합될 때
- CPU 기반 경량 베이스라인이 필요할 때
- 해석 가능한 파이프라인이 필요할 때

---

## SIFT (Scale-Invariant Feature Transform)

### 핵심 아이디어
스케일/회전이 바뀌어도 반복해서 잡히는 키포인트를 찾고, 각 키포인트 주변을 128차원 디스크립터로 표현합니다.

### 동작 순서
1. Gaussian blur로 scale-space 생성
2. DoG(Difference of Gaussian) 계산
3. DoG 극값으로 키포인트 후보 검출
4. 지배적 방향(orientation) 할당
5. 128D 디스크립터 생성

### 수식
Gaussian scale-space:
$$
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
$$

Difference of Gaussian:
$$
D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)
$$

**기호 설명**
- $I(x,y)$: 입력 이미지 밝기
- $G(x,y,\sigma)$: 표준편차 $\sigma$를 가진 Gaussian 커널
- $L(x,y,\sigma)$: 스케일 $\sigma$에서 블러링된 이미지
- $k$: 스케일 간 간격 배수(예: $\sqrt{2}$)
- $D(x,y,\sigma)$: 키포인트 검출용 DoG 응답

### 장단점
- 장점: 스케일/회전에 강건
- 단점: 계산량이 커서 대규모 실시간 파이프라인에는 부담

---

## HOG (Histogram of Oriented Gradients)

### 핵심 아이디어
이미지를 작은 셀로 나눈 뒤, 각 셀에서 그래디언트 방향 분포(히스토그램)를 계산합니다.

### 동작 순서
1. x/y 방향 그래디언트 계산
2. 셀 분할(보통 8x8)
3. 방향 히스토그램(보통 9 bins) 생성
4. 블록(보통 2x2 셀) 단위 정규화
5. 전체 특징 벡터로 연결

### 수식
$$
G = \sqrt{G_x^2 + G_y^2}, \qquad
\theta = \arctan\left(\frac{G_y}{G_x}\right)
$$

**기호 설명**
- $G_x, G_y$: x/y 방향 그래디언트
- $G$: 엣지 강도(크기)
- $\theta$: 엣지 방향 각도

### 장단점
- 장점: 계산이 상대적으로 가볍고 엣지 기반 물체(예: 보행자)에 강함
- 단점: 큰 스케일 변화에는 취약(멀티스케일 처리 필요)

---

## 최소 구현 (OpenCV)

```python
import cv2

# 1) SIFT 키포인트 + 디스크립터
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
print(f"#keypoints: {len(keypoints)}, descriptor shape: {descriptors.shape if descriptors is not None else None}")

# 2) HOG 보행자 검출 베이스라인
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(img)
print(f"detections: {len(boxes)}")
```

---

## 실무 디버깅 체크리스트
- [ ] 입력 전처리(그레이스케일/리사이즈)가 학습/추론에서 동일한가?
- [ ] 해상도가 너무 작아 키포인트/엣지가 사라지지 않았는가?
- [ ] HOG `winStride`, `scale`가 너무 공격적이지 않은가?
- [ ] SIFT 매칭 시 Lowe ratio test를 적용했는가?
- [ ] NMS 또는 confidence threshold 없이 그대로 박스를 쓰고 있지 않은가?

## 자주 하는 실수 (FAQ)
**Q1. SIFT 키포인트가 거의 안 잡힙니다.**  
A. 텍스처가 부족하거나 blur가 심할 수 있습니다. 이미지 대비를 높이거나 파라미터를 완화해 보세요.

**Q2. HOG는 왜 false positive가 많나요?**  
A. 슬라이딩 윈도우 기반이라 배경 패턴을 사람으로 오인하기 쉽습니다. `detectMultiScale` 파라미터와 NMS 임계값을 먼저 조정하세요.

**Q3. 지금도 SIFT/HOG를 배워야 하나요?**  
A. 네. 최신 SOTA 성능은 CNN/Transformer가 유리하지만, 고전 특징은 원리 이해·디버깅·저자원 베이스라인에 여전히 유용합니다.

## 증상 → 원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 확인할 것 |
|---|---|---|
| SIFT 매칭 점이 극단적으로 적음 | 텍스처 부족/과도한 블러 | 이미지 품질, contrast, keypoint 파라미터 |
| HOG 검출 박스가 너무 많음 | threshold/NMS 미조정 | confidence cutoff, NMS 적용 여부 |
| 스케일 바뀌면 성능 급락 | 단일 스케일 설정 | 이미지 피라미드, 멀티스케일 탐색 |
| 조명 바뀌면 오검출 증가 | 전처리 불일치 | histogram equalization/정규화 일관성 |

---

## 딥러닝과의 비교

| 측면 | SIFT/HOG | CNN |
|---|---|---|
| 특징 설계 | 수작업 | 데이터 기반 학습 |
| 데이터 요구량 | 적음~중간 | 중간~많음 |
| 성능 상한 | 제한적 | 일반적으로 더 높음 |
| 해석 가능성 | 높음 | 상대적으로 낮음 |
| 배포 난이도 | CPU 친화적 | 정확도는 높지만 계산량 큼 |

## 관련 콘텐츠
- [Conv2D](/ko/docs/components/convolution/conv2d)
- [YOLO](/ko/docs/architecture/detection/yolo)
- [Faster R-CNN](/ko/docs/architecture/detection/faster-rcnn)
