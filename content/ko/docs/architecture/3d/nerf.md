---
title: "NeRF"
weight: 1
math: true
---

# NeRF (Neural Radiance Fields)

## 개요

- **논문**: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (2020)
- **저자**: Ben Mildenhall et al. (UC Berkeley, Google)
- **핵심 기여**: 신경망으로 3D 장면을 암시적으로 표현

## 핵심 아이디어

> "MLP로 3D 공간의 색상과 밀도를 학습"

여러 시점의 이미지만으로 3D 장면을 완전히 재구성합니다.

---

## 입출력

### 입력

- **위치**: $(x, y, z)$ - 3D 공간의 점
- **방향**: $(\theta, \phi)$ - 시선 방향

### 출력

- **색상**: $(r, g, b)$ - RGB 값
- **밀도**: $\sigma$ - 불투명도

```
(x, y, z, θ, φ) → MLP → (r, g, b, σ)
```

---

## 구조

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                        NeRF                              │
│                                                          │
│   Position (x,y,z) ──→ Positional ──→ ┌─────────────┐   │
│                        Encoding        │             │   │
│                            ↓           │   MLP      │   │
│                       γ(x,y,z)    ──→ │  (8 layers)│──→ σ (density)
│                                        │   256 dim  │   │
│   Direction (θ,φ) ──→ Positional ──→  │             │   │
│                        Encoding        └──────┬──────┘   │
│                            ↓                  ↓          │
│                       γ(θ,φ)     ──→ [Additional MLP]──→ (r,g,b) color
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Positional Encoding

고주파 정보를 학습하기 위해 위치를 고차원으로 매핑:

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), ..., \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

- 위치: $L=10$ (60차원)
- 방향: $L=4$ (24차원)

### 왜 Positional Encoding이 필요한가?

MLP는 저주파 함수에 편향되어 있어서, 고주파 디테일(텍스처, 엣지)을 학습하기 어렵습니다.

---

## Volume Rendering

### 광선 추적

카메라에서 픽셀을 통해 광선을 쏘고, 광선 위의 점들을 샘플링:

$$r(t) = o + td$$

- $o$: 카메라 원점
- $d$: 광선 방향
- $t$: 광선 위의 거리

### 색상 계산

광선 위의 모든 점에서 색상과 밀도를 적분:

$$C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt$$

여기서 투과율:

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(r(s)) ds\right)$$

### 이산화 (실제 구현)

$$\hat{C}(r) = \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) c_i$$

$$T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

---

## Hierarchical Sampling

### 2단계 샘플링

효율적인 렌더링을 위해 두 개의 네트워크 사용:

```
1. Coarse Network: 균등 샘플링으로 대략적인 밀도 파악
2. Fine Network: 밀도 높은 영역에 집중 샘플링
```

```python
# Coarse: 64 samples (균등)
t_coarse = torch.linspace(near, far, 64)
weights = render_weights(coarse_network, t_coarse)

# Fine: 64 + 128 samples (가중 샘플링)
t_fine = sample_pdf(t_coarse, weights, 128)
t_all = torch.sort(torch.cat([t_coarse, t_fine]))[0]
color = render(fine_network, t_all)
```

---

## 학습

### 손실 함수

렌더링된 색상과 실제 이미지 픽셀의 차이:

$$L = \sum_{r \in R} \left[ \| \hat{C}_c(r) - C(r) \|^2 + \| \hat{C}_f(r) - C(r) \|^2 \right]$$

### 학습 데이터

- 다양한 시점의 이미지 (보통 100장 이상)
- 각 이미지의 카메라 포즈 (COLMAP으로 추정)

### 학습 시간

- 단일 장면: 1-2일 (V100 GPU 기준)
- 매우 느린 것이 단점

---

## 구현 예시

### 기본 MLP

```python
import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = [4]  # skip connection at layer 4

        # Position encoding layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips
             else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        # Density output
        self.sigma_linear = nn.Linear(W, 1)

        # Color output
        self.feature_linear = nn.Linear(W, W)
        self.views_linear = nn.Linear(W + input_ch_views, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x, d):
        # x: position, d: direction
        h = x
        for i, layer in enumerate(self.pts_linears):
            h = torch.relu(layer(h))
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)

        sigma = self.sigma_linear(h)

        feature = self.feature_linear(h)
        h = torch.cat([feature, d], dim=-1)
        h = torch.relu(self.views_linear(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return rgb, sigma
```

### Positional Encoding

```python
def positional_encoding(x, L):
    """
    x: (..., D)
    returns: (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, device=x.device)
    x_freq = x[..., None] * freqs  # (..., D, L)

    encoded = torch.cat([
        torch.sin(x_freq),
        torch.cos(x_freq)
    ], dim=-1)  # (..., D, 2L)

    return encoded.flatten(-2)  # (..., D * 2L)
```

### Volume Rendering

```python
def render_rays(network, rays_o, rays_d, near, far, N_samples):
    # Sample points along ray
    t = torch.linspace(near, far, N_samples)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t[..., :, None]

    # Get colors and densities
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = rays_d[..., None, :].expand_as(pts).reshape(-1, 3)

    rgb, sigma = network(
        positional_encoding(pts_flat, L=10),
        positional_encoding(dirs_flat, L=4)
    )

    rgb = rgb.reshape(*pts.shape[:-1], 3)
    sigma = sigma.reshape(*pts.shape[:-1])

    # Volume rendering
    delta = t[..., 1:] - t[..., :-1]
    alpha = 1 - torch.exp(-sigma[..., :-1] * delta)

    T = torch.cumprod(1 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)

    weights = alpha * T
    rgb_map = (weights[..., None] * rgb[..., :-1, :]).sum(dim=-2)

    return rgb_map
```

---

## NeRF 변형들

### 속도 개선

| 모델 | 특징 | 속도 향상 |
|------|------|----------|
| **Instant-NGP** | Hash encoding | 1000x |
| **Plenoxels** | Voxel grid | 100x |
| **TensoRF** | Tensor decomposition | 100x |

### 기능 확장

| 모델 | 기능 |
|------|------|
| **Mip-NeRF** | 안티앨리어싱 |
| **NeRF-W** | 야외 장면, 조명 변화 |
| **D-NeRF** | 동적 장면 |
| **NeRF++** | 무한 장면 |

---

## Instant-NGP

### 핵심 아이디어

다해상도 해시 테이블로 위치 인코딩 대체:

```
Position → Multi-resolution Hash → Small MLP → (rgb, σ)
```

### 장점

- 학습: 수 분
- 렌더링: 실시간
- 품질 유지

---

## 한계점

- **학습 시간**: 장면당 수 시간 ~ 수 일
- **렌더링 속도**: 실시간 어려움
- **정적 장면**: 동적 장면 처리 어려움
- **메모리**: 고해상도 시 많은 샘플 필요

---

## 관련 콘텐츠

- [3D Gaussian Splatting](/ko/docs/architecture/3d/3dgs) - 실시간 대안
- [MLP](/ko/docs/math/mlp) - 기본 네트워크
- [ViT](/ko/docs/architecture/transformer/vit) - Vision 모델 비교

