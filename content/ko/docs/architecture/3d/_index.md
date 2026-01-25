---
title: "3D Vision"
weight: 8
bookCollapseSection: true
---

# 3D Vision

3차원 공간을 이해하고 재구성하는 모델들입니다.

## 모델 목록

| 모델 | 연도 | 핵심 기여 |
|------|------|----------|
| [NeRF](/ko/docs/architecture/3d/nerf) | 2020 | Neural Radiance Fields |
| [3D Gaussian Splatting](/ko/docs/architecture/3d/3dgs) | 2023 | 실시간 렌더링 |

## 발전 과정

```
NeRF (2020) → Instant-NGP (2022) → 3D Gaussian Splatting (2023)
```

## 공통 개념

### Volume Rendering

3D 공간을 2D 이미지로 렌더링:

$$C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt$$

### Novel View Synthesis

학습하지 않은 시점에서 이미지 생성

