---
title: "Homography"
weight: 2
math: true
---

# Homography

## 개요

Homography는 두 평면 사이의 투영 변환을 나타내는 3×3 행렬입니다.

## 수식

$$
\begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

~는 스케일까지 동일함을 의미 (homogeneous 좌표)

실제 좌표:
$$
u' = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}}, \quad v' = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}
$$

## 자유도

- 8 자유도 (3×3=9개 원소, 스케일 1개 제거)
- 최소 4개의 대응점 필요 (각 점이 2개 방정식 제공)

## 특수한 경우

| 변환 | 자유도 | 행렬 형태 |
|------|--------|----------|
| 이동 | 2 | [[1,0,tx],[0,1,ty],[0,0,1]] |
| 유클리드 | 3 | [[cos,-sin,tx],[sin,cos,ty],[0,0,1]] |
| 유사 | 4 | [[s·cos,-s·sin,tx],[s·sin,s·cos,ty],[0,0,1]] |
| 아핀 | 6 | [[a,b,tx],[c,d,ty],[0,0,1]] |
| 투영 | 8 | [[h11,h12,h13],[h21,h22,h23],[h31,h32,1]] |

## Homography 계산

### DLT (Direct Linear Transform)

4개 이상의 대응점으로 계산:

```python
import numpy as np

def compute_homography_dlt(src_pts, dst_pts):
    """
    src_pts, dst_pts: (N, 2) 대응점, N >= 4
    """
    N = len(src_pts)
    A = []

    for i in range(N):
        x, y = src_pts[i]
        u, v = dst_pts[i]

        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.array(A)

    # SVD로 해 찾기
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    return H / H[2, 2]  # 정규화
```

### OpenCV 사용

```python
import cv2

# 대응점
src_pts = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
dst_pts = np.float32([[10, 20], [110, 30], [105, 120], [5, 110]])

# Homography 계산
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# 이미지 변환
warped = cv2.warpPerspective(img, H, (width, height))
```

## RANSAC

아웃라이어가 있을 때 robust한 추정:

```python
def ransac_homography(src_pts, dst_pts, threshold=3.0, max_iters=1000):
    best_H = None
    best_inliers = 0

    for _ in range(max_iters):
        # 4개 랜덤 샘플
        idx = np.random.choice(len(src_pts), 4, replace=False)
        H = compute_homography_dlt(src_pts[idx], dst_pts[idx])

        # 모든 점에 대해 에러 계산
        src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        proj = (H @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]

        errors = np.linalg.norm(proj - dst_pts, axis=1)
        inliers = np.sum(errors < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H

    return best_H
```

## 응용

### 파노라마 스티칭

```python
# 특징점 매칭
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Homography
src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# 스티칭
result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
result[:, :img2.shape[1]] = img2
```

### 문서 스캔

```python
# 문서 코너 검출 후
corners = detect_document_corners(img)

# 목표 사각형
dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# 변환
H = cv2.getPerspectiveTransform(corners, dst)
scanned = cv2.warpPerspective(img, H, (width, height))
```

## 관련 콘텐츠

- [Camera Model](/ko/docs/math/geometry/camera-model)
- [SVD](/ko/docs/math/linear-algebra/svd)
- [행렬](/ko/docs/math/linear-algebra/matrix)
