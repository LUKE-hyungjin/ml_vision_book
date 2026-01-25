---
title: "3D Gaussian Splatting"
weight: 2
math: true
---

# 3D Gaussian Splatting

## 개요

- **논문**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering (2023)
- **저자**: Bernhard Kerbl et al. (Inria, Max Planck Institute)
- **핵심 기여**: 실시간 렌더링이 가능한 명시적 3D 표현

## 핵심 아이디어

> "3D 점들을 Gaussian으로 표현하여 빠르게 렌더링"

NeRF와 달리 명시적인 점 기반 표현으로 실시간 렌더링을 달성합니다.

---

## NeRF vs 3D Gaussian Splatting

| 측면 | NeRF | 3DGS |
|------|------|------|
| 표현 | 암시적 (MLP) | 명시적 (점 집합) |
| 렌더링 | Ray marching | Rasterization |
| 속도 | 느림 (분 단위) | 실시간 (100+ FPS) |
| 학습 | 느림 (시간~일) | 빠름 (분~시간) |
| 편집 | 어려움 | 용이 |

---

## 구조

### 3D Gaussian 표현

각 점은 다음 속성을 가짐:

```
┌─────────────────────────────────────────┐
│           3D Gaussian                    │
│                                          │
│   위치 (μ):     (x, y, z)               │
│   공분산 (Σ):   3×3 행렬                │
│   색상 (c):     SH coefficients         │
│   불투명도 (α): scalar                   │
│                                          │
└─────────────────────────────────────────┘
```

### 수학적 정의

3D Gaussian 분포:

$$G(x) = \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$

### 공분산 행렬 분해

양의 준정부호를 보장하기 위해:

$$\Sigma = RSS^TR^T$$

- $R$: 회전 행렬 (quaternion으로 표현)
- $S$: 스케일 행렬 (대각 행렬)

```python
# 학습 가능한 파라미터
position = torch.zeros(N, 3)      # 위치
scale = torch.ones(N, 3)          # 스케일
rotation = torch.zeros(N, 4)      # 쿼터니언
opacity = torch.zeros(N, 1)       # 불투명도
sh_coeffs = torch.zeros(N, 16, 3) # 색상 (SH)
```

---

## Spherical Harmonics (SH)

### 왜 SH를 사용하는가?

시점에 따른 색상 변화(반사, 하이라이트)를 표현:

```
정면에서 본 색: 밝은 빨강
측면에서 본 색: 어두운 빨강
```

### SH 계수

0차부터 3차까지 총 16개 계수:

$$c(d) = \sum_{l=0}^{3} \sum_{m=-l}^{l} c_{lm} Y_{lm}(d)$$

---

## 렌더링 파이프라인

### Splatting 과정

```
┌─────────────────────────────────────────────────────────┐
│              Gaussian Splatting Pipeline                 │
│                                                          │
│   1. 3D Gaussians ──→ Project to 2D                     │
│                              ↓                           │
│   2. 2D Gaussians ──→ Sort by depth                     │
│                              ↓                           │
│   3. Sorted Gaussians ──→ Alpha blending (front-to-back)│
│                              ↓                           │
│   4. Final Image                                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2D 투영

3D Gaussian을 2D로 투영:

$$\Sigma_{2D} = JW\Sigma W^TJ^T$$

- $W$: 뷰 변환 행렬
- $J$: 야코비안 (어파인 근사)

### Alpha Blending

깊이 순서대로 색상 합성:

$$C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

---

## 학습

### Adaptive Density Control

학습 중 Gaussian 개수를 동적으로 조절:

```
┌─────────────────────────────────────────┐
│        Adaptive Density Control          │
│                                          │
│   큰 gradient + 작은 scale → Clone      │
│   큰 gradient + 큰 scale   → Split      │
│   낮은 opacity            → Remove      │
│                                          │
└─────────────────────────────────────────┘
```

### 손실 함수

$$L = (1-\lambda)L_1 + \lambda L_{D-SSIM}$$

- $L_1$: 픽셀 단위 손실
- $L_{D-SSIM}$: 구조적 유사도 손실

### 학습 과정

```python
for iteration in range(30000):
    # 1. 렌더링
    rendered = render(gaussians, camera)

    # 2. 손실 계산
    loss = l1_loss(rendered, gt_image) + ssim_loss(rendered, gt_image)

    # 3. 역전파
    loss.backward()
    optimizer.step()

    # 4. Adaptive control (매 100 iteration)
    if iteration % 100 == 0:
        densify_and_prune(gaussians)
```

---

## 구현 예시

### 데이터 구조

```python
class GaussianModel:
    def __init__(self):
        # 위치
        self._xyz = torch.empty(0)
        # 색상 (SH coefficients)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        # 스케일
        self._scaling = torch.empty(0)
        # 회전 (quaternion)
        self._rotation = torch.empty(0)
        # 불투명도
        self._opacity = torch.empty(0)

    @property
    def get_covariance(self):
        """공분산 행렬 계산"""
        S = self.scaling_activation(self._scaling)
        R = self.rotation_activation(self._rotation)
        L = build_scaling_rotation(S, R)
        return L @ L.transpose(1, 2)
```

### 초기화 (from Point Cloud)

```python
def create_from_pcd(self, pcd):
    """SfM 점군으로부터 초기화"""
    points = torch.tensor(pcd.points)
    colors = torch.tensor(pcd.colors)

    self._xyz = nn.Parameter(points)

    # 색상을 SH로 변환
    fused_color = RGB2SH(colors)
    self._features_dc = nn.Parameter(fused_color)

    # 초기 스케일: 인접 점까지 거리
    dist = torch.cdist(points, points)
    dist[dist == 0] = float('inf')
    init_scale = torch.log(dist.min(dim=1)[0])
    self._scaling = nn.Parameter(init_scale.repeat(1, 3))

    # 초기 회전: identity
    self._rotation = nn.Parameter(torch.zeros(len(points), 4))
    self._rotation[:, 0] = 1  # w=1 for identity quaternion

    # 초기 불투명도
    self._opacity = nn.Parameter(inverse_sigmoid(0.1 * torch.ones(len(points), 1)))
```

### CUDA 렌더러

```python
def render(viewpoint_camera, gaussians, bg_color):
    """CUDA 기반 고속 렌더링"""
    # 2D 투영
    screenspace_points = project_to_screen(gaussians, viewpoint_camera)

    # 타일 기반 래스터라이제이션
    rendered_image, radii = rasterize_gaussians(
        means3D=gaussians.get_xyz,
        means2D=screenspace_points,
        shs=gaussians.get_features,
        colors_precomp=None,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        bg=bg_color
    )

    return rendered_image
```

---

## 성능

### 품질 비교 (PSNR)

| 데이터셋 | NeRF | Instant-NGP | 3DGS |
|----------|------|-------------|------|
| Synthetic | 31.0 | 33.2 | **33.3** |
| Real-World | 26.8 | 29.5 | **29.4** |

### 속도

| 방법 | 학습 시간 | 렌더링 FPS |
|------|----------|------------|
| NeRF | 수 일 | 0.05 |
| Instant-NGP | 5분 | 10 |
| 3DGS | 30분 | **100+** |

---

## 장점과 단점

### 장점

- 실시간 렌더링 (100+ FPS)
- 빠른 학습 (30분)
- 명시적 표현으로 편집 용이
- 고품질 결과

### 단점

- 메모리 사용량 (수백만 점)
- 초기 점군 품질 의존
- 복잡한 기하학에서 아티팩트

---

## 응용

### 1. VR/AR

실시간 렌더링으로 몰입형 경험

### 2. 게임

사실적인 환경 재구성

### 3. 디지털 트윈

실제 공간의 3D 복제

### 4. 영화/VFX

빠른 프리비주얼라이제이션

---

## 후속 연구

| 모델 | 개선점 |
|------|--------|
| **4D-GS** | 동적 장면 |
| **GSGEN** | 텍스트→3D 생성 |
| **GaussianAvatar** | 사람 모델링 |
| **Compact-3DGS** | 압축 |
| **Mip-Splatting** | 안티앨리어싱 |

---

## 관련 콘텐츠

- [NeRF](/ko/docs/architecture/3d/nerf) - 암시적 표현
- [Transformer](/ko/docs/architecture/transformer) - 최신 3D 모델에서 활용

