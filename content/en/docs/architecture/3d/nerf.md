---
title: "NeRF"
weight: 1
math: true
---

# NeRF (Neural Radiance Fields)

{{% hint info %}}
**Prerequisites**: [Camera Model](/en/docs/math/geometry/camera-model) | [Positional Encoding](/en/docs/components/attention/positional-encoding)
{{% /hint %}}

## One-line Summary
> **NeRF learns a continuous 3D scene by predicting color and density at any 3D location and view direction.**

## Why this model?
Classic 3D reconstruction pipelines often need explicit meshes, depth maps, or heavy geometry engineering.
NeRF showed that with only multi-view images + camera poses, we can train a neural network to render novel views with high visual quality.

Think of it as learning a "smart fog field" in 3D space:
- density decides how much light is blocked,
- color decides what color that point contributes.

## Overview
- **Paper**: *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* (2020)
- **Authors**: Ben Mildenhall et al.
- **Key contribution**: Implicit 3D scene representation + differentiable volume rendering

## Architecture
### Input and output
NeRF MLP takes:
- 3D position: $(x, y, z)$
- viewing direction: $(\theta, \phi)$ (or 3D unit ray direction)

and predicts:
- RGB color: $(r, g, b)$
- volume density: $\sigma$

$$
(x, y, z, d) \rightarrow \text{MLP} \rightarrow (r, g, b, \sigma)
$$

### Why positional encoding is needed
A plain MLP tends to learn low-frequency signals first.
To recover sharp textures/edges, NeRF maps input coordinates into high-frequency features:

$$
\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))
$$

**Symbol meanings:**
- $p$: scalar coordinate component (e.g., one of $x,y,z$)
- $L$: number of frequency bands
- $\gamma(p)$: encoded feature vector used as MLP input

## Rendering equation (core math)
For a camera ray $r(t) = o + td$, rendered color is:

$$
C(r) = \int_{t_n}^{t_f} T(t)\,\sigma(r(t))\,c(r(t), d)\,dt
$$

with transmittance:

$$
T(t)=\exp\left(-\int_{t_n}^{t}\sigma(r(s))ds\right)
$$

**Symbol meanings:**
- $o$: camera origin
- $d$: ray direction
- $t$: distance along the ray
- $\sigma$: density (opacity per unit distance)
- $c$: RGB color at sampled point
- $T(t)$: probability that light survives up to distance $t$

In code, this is approximated by discrete samples along each ray.

## Minimal implementation (ray rendering sketch)
```python
import torch


def volume_render(rgb, sigma, delta):
    """
    rgb:   [N, 3] sampled colors along one ray
    sigma: [N]    sampled densities
    delta: [N]    distance to next sample
    """
    alpha = 1.0 - torch.exp(-sigma * delta)              # opacity at each sample
    T = torch.cumprod(torch.cat([torch.ones(1), 1 - alpha[:-1] + 1e-10]), dim=0)
    weights = T * alpha                                   # contribution weights
    return (weights[:, None] * rgb).sum(dim=0)           # final RGB
```

## Practical debugging checklist
- [ ] **Pose sanity check**: Are camera intrinsics/extrinsics correct? (wrong poses break everything)
- [ ] **Near/Far bounds**: Are ray sampling bounds too wide/narrow?
- [ ] **Sample count**: Too few samples causes blur/holes; too many is very slow
- [ ] **Encoding bands**: Too low $L$ underfits details; too high can overfit/noise
- [ ] **Loss curve + rendered snapshots**: Monitor both numeric loss and visual artifacts

## Common mistakes (FAQ)
**Q1. NeRF only needs images, so no geometry is needed at all, right?**  
A. You still need camera poses (geometry signal). NeRF is not "pose-free" by default.

**Q2. Why does training take so long?**  
A. NeRF evaluates an MLP at many sampled points per ray and per image. Rendering cost is high.

**Q3. My render is blurry even when loss decreases. Why?**  
A. Common causes: poor pose quality, insufficient high-frequency encoding, or too few ray samples.

## Related Content
- [3D Gaussian Splatting](/en/docs/architecture/3d/3dgs)
- [Camera Model](/en/docs/math/geometry/camera-model)
- [Positional Encoding](/en/docs/components/attention/positional-encoding)
