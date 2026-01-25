---
title: "Epipolar Geometry"
weight: 3
math: true
---

# Epipolar Geometry

## 개요

Epipolar Geometry는 두 카메라 시점 사이의 기하학적 관계를 설명합니다. 스테레오 비전과 3D 복원의 핵심입니다.

## 핵심 개념

```
        O₁ ←───────── Baseline ─────────→ O₂
       /  \                              /  \
      /    \                            /    \
     /      \       Epipolar Plane     /      \
    /        \          ↓             /        \
   e₁ ········ p₁ ═══════════════ p₂ ········ e₂
              ↑                    ↑
         Epipolar Line        Epipolar Line
```

- **에피폴 (Epipole)**: 다른 카메라 중심이 투영된 점
- **에피폴라 평면**: 두 카메라 중심과 3D 점을 포함하는 평면
- **에피폴라 라인**: 에피폴라 평면과 이미지 평면의 교선

## Fundamental Matrix (F)

두 이미지의 대응점 관계:

$$
p_2^T F p_1 = 0
$$

- 7 자유도 (3×3 - 스케일 - rank 2 제약)
- 최소 7개 대응점 필요 (8점 알고리즘 권장)

### 에피폴라 라인

```python
# 이미지 1의 점 p1에 대응하는 이미지 2의 에피폴라 라인
l2 = F @ p1  # (a, b, c) -> ax + by + c = 0

# 이미지 2의 점 p2에 대응하는 이미지 1의 에피폴라 라인
l1 = F.T @ p2
```

## Essential Matrix (E)

캘리브레이션된 카메라의 경우:

$$
E = K_2^T F K_1
$$

- 5 자유도 (회전 3 + 이동 2, 스케일 제외)
- $E = [t]_× R$ (t: 이동, R: 회전)

### E에서 R, t 복원

```python
import cv2
import numpy as np

# Essential Matrix 계산
E, mask = cv2.findEssentialMat(pts1, pts2, K)

# R, t 복원
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# 4가지 해 중 유효한 것 선택 (cheirality check)
```

## 8점 알고리즘

```python
def eight_point_algorithm(pts1, pts2):
    """
    pts1, pts2: (N, 2) 정규화된 좌표, N >= 8
    """
    N = len(pts1)
    A = []

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    A = np.array(A)

    # SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Rank-2 제약 적용
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    return F
```

## 정규화

수치 안정성을 위해 좌표 정규화 필수:

```python
def normalize_points(pts):
    """좌표를 평균 0, 평균 거리 sqrt(2)로 정규화"""
    mean = pts.mean(axis=0)
    pts_centered = pts - mean

    scale = np.sqrt(2) / np.mean(np.linalg.norm(pts_centered, axis=1))

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])

    pts_normalized = (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2]

    return pts_normalized, T
```

## 삼각측량 (Triangulation)

두 시점에서 3D 점 복원:

```python
def triangulate(P1, P2, pts1, pts2):
    """
    P1, P2: (3, 4) 투영 행렬
    pts1, pts2: (N, 2) 대응점
    """
    points_3d = []

    for p1, p2 in zip(pts1, pts2):
        A = np.array([
            p1[0] * P1[2] - P1[0],
            p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0],
            p2[1] * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Homogeneous → Euclidean

        points_3d.append(X)

    return np.array(points_3d)

# OpenCV
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = (points_4d[:3] / points_4d[3]).T
```

## 스테레오 정류 (Rectification)

에피폴라 라인을 수평으로 정렬:

```python
# 스테레오 정류
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, dist1, K2, dist2, img_size, R, t
)

# 정류 맵 생성
map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, img_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, img_size, cv2.CV_32FC1)

# 정류 적용
img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
```

## 관련 콘텐츠

- [Camera Model](/ko/docs/math/geometry/camera-model)
- [Homography](/ko/docs/math/geometry/homography)
- [SVD](/ko/docs/math/linear-algebra/svd)
