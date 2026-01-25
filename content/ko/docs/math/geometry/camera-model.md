---
title: "Camera Model"
weight: 1
math: true
---

# Camera Model

## 개요

카메라 모델은 3D 세계의 점이 2D 이미지 평면에 어떻게 투영되는지 설명합니다.

## Pinhole Camera Model

가장 기본적인 카메라 모델:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} K [R | t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

## 내부 파라미터 (Intrinsic Matrix)

카메라 고유의 특성:

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

- **f_x, f_y**: 초점 거리 (픽셀 단위)
- **c_x, c_y**: 주점 (principal point), 보통 이미지 중심

## 외부 파라미터 (Extrinsic Matrix)

카메라의 위치와 방향:

$$
[R | t] = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix}
$$

- **R**: 3×3 회전 행렬
- **t**: 3×1 이동 벡터

## 투영 과정

```python
import numpy as np

def project_3d_to_2d(points_3d, K, R, t):
    """
    points_3d: (N, 3) 월드 좌표
    K: (3, 3) 내부 파라미터
    R: (3, 3) 회전 행렬
    t: (3,) 이동 벡터
    """
    # 1. 월드 → 카메라 좌표
    points_cam = (R @ points_3d.T).T + t  # (N, 3)

    # 2. 카메라 → 이미지 좌표 (정규화)
    points_norm = points_cam[:, :2] / points_cam[:, 2:3]  # (N, 2)

    # 3. 정규화 → 픽셀 좌표
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * points_norm[:, 0] + cx
    v = fy * points_norm[:, 1] + cy

    return np.stack([u, v], axis=1)  # (N, 2)
```

## 렌즈 왜곡

실제 카메라는 렌즈 왜곡 존재:

### 방사 왜곡 (Radial Distortion)

$$
\begin{aligned}
x_{distorted} &= x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
y_{distorted} &= y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
\end{aligned}
$$

### 접선 왜곡 (Tangential Distortion)

$$
\begin{aligned}
x_{distorted} &= x + 2p_1 xy + p_2(r^2 + 2x^2) \\
y_{distorted} &= y + p_1(r^2 + 2y^2) + 2p_2 xy
\end{aligned}
$$

## OpenCV 캘리브레이션

```python
import cv2
import numpy as np

# 체커보드 패턴 검출
pattern_size = (9, 6)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

obj_points = []  # 3D 점
img_points = []  # 2D 점

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)

# 캘리브레이션
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print(f"Intrinsic Matrix:\n{K}")
print(f"Distortion Coefficients: {dist}")
```

## 왜곡 보정

```python
# 왜곡 제거
img_undistorted = cv2.undistort(img, K, dist)

# 또는 맵 생성 (반복 사용 시 효율적)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, img.shape[:2][::-1], cv2.CV_32FC1)
img_undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

## 역투영 (Backprojection)

2D → 3D 광선:

```python
def backproject_to_ray(pixel, K):
    """픽셀 좌표를 카메라 좌표계의 광선으로"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = pixel
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0

    ray = np.array([x, y, z])
    return ray / np.linalg.norm(ray)  # 정규화
```

## 관련 콘텐츠

- [Homography](/ko/docs/math/geometry/homography)
- [행렬](/ko/docs/math/linear-algebra/matrix)
- [Epipolar Geometry](/ko/docs/math/geometry/epipolar)
