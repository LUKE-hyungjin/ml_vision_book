---
title: "기하학"
weight: 7
bookCollapseSection: true
math: true
---

# 기하학 (Geometry)

컴퓨터 비전에서 3D 공간과 2D 이미지 사이의 관계를 다룹니다.

## 핵심 개념

| 개념 | 설명 | 응용 |
|------|------|------|
| [Camera Model](/ko/docs/math/geometry/camera-model) | 3D→2D 투영 | 카메라 캘리브레이션 |
| [Homography](/ko/docs/math/geometry/homography) | 평면 간 변환 | 이미지 정합, 파노라마 |
| [Epipolar Geometry](/ko/docs/math/geometry/epipolar) | 두 시점 관계 | 스테레오 비전, SfM |

## 좌표계

```
월드 좌표 (X, Y, Z)
      ↓ 외부 파라미터 [R|t]
카메라 좌표 (x, y, z)
      ↓ 내부 파라미터 K
이미지 좌표 (u, v)
```

## 관련 콘텐츠

- [행렬](/ko/docs/math/linear-algebra/matrix)
- [SVD](/ko/docs/math/linear-algebra/svd)
