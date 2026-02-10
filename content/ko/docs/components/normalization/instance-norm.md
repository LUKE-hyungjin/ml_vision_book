---
title: "Instance Normalization"
weight: 5
math: true
---

# Instance Normalization

{{% hint info %}}
**선수지식**: [Batch Normalization](/ko/docs/components/normalization/batch-norm)
{{% /hint %}}

> **한 줄 요약**: Instance Normalization은 **각 샘플의 각 채널**에서 독립적으로 정규화하여, 이미지의 **스타일(콘트라스트, 밝기)을 제거**하고 콘텐츠만 남기는 기법입니다.

## 왜 Instance Normalization이 필요한가?

### 문제 상황: "스타일을 바꾸고 싶은데, 원본 스타일이 방해합니다"

Style Transfer에서는 콘텐츠 이미지의 **스타일(색감, 콘트라스트)**을 제거하고 새로운 스타일을 입혀야 합니다:

```python
# Style Transfer 파이프라인
content_image = load("photo.jpg")       # 스타일: 사진풍
style_image = load("starry_night.jpg")  # 스타일: 고흐 화풍

output = style_transfer(content_image, style_image)
# → content의 스타일을 제거하고, style의 스타일을 입힘
```

**BatchNorm의 문제**: 배치 내 이미지들의 평균 스타일로 정규화합니다. 하지만 Style Transfer에서는 **각 이미지마다 다른 스타일을 가지므로**, 배치 평균은 의미가 없습니다.

### 핵심 관찰: "채널 통계 = 스타일 정보"

연구자들은 특징 맵의 **채널별 평균과 분산**이 이미지의 스타일을 인코딩한다는 것을 발견했습니다:

```
채널별 평균/분산:
  밝은 사진:  높은 평균, 낮은 분산  → "밝고 균일한 스타일"
  어두운 사진: 낮은 평균, 높은 분산  → "어둡고 대비가 강한 스타일"

→ 평균/분산을 정규화하면 = 스타일 정보 제거!
```

### 해결: "각 이미지, 각 채널을 독립적으로 정규화하자!"

```
BatchNorm:    배치 전체에서 통계 → 이미지 간 스타일 차이 무시
InstanceNorm: 각 이미지, 각 채널에서 독립 → 이미지별 스타일 제거!
```

---

## 수식

### Instance Normalization

입력 $(B, C, H, W)$에서 **각 $(b, c)$ 쌍마다** 독립적으로 정규화:

$$
\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_{b,c}}{\sqrt{\sigma_{b,c}^2 + \epsilon}}
$$

$$
y_{b,c,h,w} = \gamma_c \cdot \hat{x}_{b,c,h,w} + \beta_c
$$

**평균과 분산은 공간 축에서만 계산:**

$$
\mu_{b,c} = \frac{1}{H \cdot W} \sum_{h,w} x_{b,c,h,w}
$$

$$
\sigma_{b,c}^2 = \frac{1}{H \cdot W} \sum_{h,w} (x_{b,c,h,w} - \mu_{b,c})^2
$$

**각 기호의 의미:**
- $\mu_{b,c}$ : 샘플 $b$, 채널 $c$의 공간 평균 — **배치 축도 채널 축도 없음!**
- $\sigma_{b,c}^2$ : 샘플 $b$, 채널 $c$의 공간 분산
- $\gamma_c, \beta_c$ : 채널별 학습 가능한 스케일/시프트 (선택적)

### 정규화 축 비교 (시각적)

```
텐서: (B, C, H, W)

BatchNorm:     dim=(0, 2, 3) 평균 → C개 통계
               배치 + 공간 전체에서 → "이 채널의 전형적 값"

InstanceNorm:  dim=(2, 3) 평균 → B×C개 통계
               공간에서만 → "이 이미지, 이 채널의 값"

LayerNorm:     dim=(1, 2, 3) 평균 → B개 통계
               채널 + 공간 전체에서 → "이 이미지 전체의 값"

GroupNorm:     dim=(그룹 내 채널, 2, 3) → B×G개 통계
               그룹 내에서 → "이 이미지, 이 그룹의 값"
```

---

## 구현

```python
import torch
import torch.nn as nn

# === PyTorch InstanceNorm ===
# affine=False가 기본값! (스타일 제거가 목적이므로)
inst_norm = nn.InstanceNorm2d(num_features=64)

# affine=True로 하면 γ, β 학습
inst_norm_affine = nn.InstanceNorm2d(64, affine=True)

x = torch.randn(4, 64, 32, 32)  # (B, C, H, W)
y = inst_norm(x)
print(f"입력: {x.shape} → 출력: {y.shape}")


# === 수동 구현 ===
class ManualInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # (B, C, H, W) → H, W 축에서만 통계 계산
        mean = x.mean(dim=(2, 3), keepdim=True)      # (B, C, 1, 1)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)  # (B, C, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = self.gamma[None, :, None, None] * x_norm + \
                     self.beta[None, :, None, None]
        return x_norm


# === GroupNorm으로도 구현 가능 ===
# InstanceNorm = GroupNorm(G=C)
gn_as_in = nn.GroupNorm(num_groups=64, num_channels=64)  # G=C → InstanceNorm!
```

---

## Style Transfer에서의 활용

### Adaptive Instance Normalization (AdaIN)

Style Transfer의 핵심 기법: 콘텐츠 특징의 통계를 **스타일 특징의 통계로 교체**합니다.

$$
\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)
$$

**해석:**
1. $\frac{x - \mu(x)}{\sigma(x)}$ : 콘텐츠의 스타일(통계) 제거 — **Instance Norm!**
2. $\sigma(y) \cdot (\cdots) + \mu(y)$ : 스타일 이미지의 통계를 입힘

```python
def adaptive_instance_norm(content_feat, style_feat):
    """AdaIN: 콘텐츠 통계 → 스타일 통계로 교체"""
    # 콘텐츠/스타일의 채널별 평균, 분산
    c_mean = content_feat.mean(dim=(2, 3), keepdim=True)
    c_std = content_feat.std(dim=(2, 3), keepdim=True) + 1e-5

    s_mean = style_feat.mean(dim=(2, 3), keepdim=True)
    s_std = style_feat.std(dim=(2, 3), keepdim=True) + 1e-5

    # 콘텐츠 정규화 + 스타일 통계 적용
    normalized = (content_feat - c_mean) / c_std   # Instance Norm
    return s_std * normalized + s_mean              # 스타일 입히기
```

### StyleGAN의 핵심 메커니즘

StyleGAN은 AdaIN을 발전시켜 **스타일 벡터**로 이미지 생성을 제어합니다:

```python
class StyleGANBlock(nn.Module):
    """StyleGAN의 스타일 적용 블록 (간략화)"""
    def __init__(self, ch, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch)
        # 스타일 벡터 → γ, β 예측
        self.style_to_gamma = nn.Linear(style_dim, ch)
        self.style_to_beta = nn.Linear(style_dim, ch)

    def forward(self, x, style):
        # 1. Instance Norm으로 기존 스타일 제거
        x = self.norm(x)
        # 2. 스타일 벡터에서 새로운 γ, β 생성
        gamma = self.style_to_gamma(style)[:, :, None, None]
        beta = self.style_to_beta(style)[:, :, None, None]
        # 3. 새로운 스타일 적용
        return gamma * x + beta
```

---

## BatchNorm vs InstanceNorm 비교

| | BatchNorm | InstanceNorm |
|---|---|---|
| 정규화 축 | B, H, W | **H, W만** |
| 배치 의존 | O | **X** |
| Train/Eval 차이 | O | **X** (기본) |
| 제거하는 것 | 배치 간 변동 | **개별 이미지의 스타일** |
| affine 기본값 | True ($\gamma, \beta$ 학습) | **False** (정규화만) |
| 주 사용처 | 분류, Detection | **Style Transfer, GAN** |

### 왜 분류에서는 InstanceNorm이 좋지 않을까?

```
분류: "이 이미지가 고양이인가 강아지인가?"
→ 밝기, 대비 등의 스타일 정보도 분류에 유용할 수 있음
→ InstanceNorm이 이런 정보를 제거하면 성능 하락!

Style Transfer: "스타일만 바꾸고 싶다"
→ 원본 스타일은 제거해야 함
→ InstanceNorm이 딱 맞음!
```

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === InstanceNorm이 스타일(통계)을 제거하는 과정 ===
print("=== 스타일 제거 확인 ===")

inst_norm = nn.InstanceNorm2d(3)

# 밝은 이미지와 어두운 이미지
bright = torch.randn(1, 3, 32, 32) + 5.0   # 높은 평균
dark = torch.randn(1, 3, 32, 32) - 3.0     # 낮은 평균

print("정규화 전:")
print(f"  밝은 이미지 — 평균: {bright.mean():.2f}, 분산: {bright.var():.2f}")
print(f"  어두운 이미지 — 평균: {dark.mean():.2f}, 분산: {dark.var():.2f}")

bright_norm = inst_norm(bright)
dark_norm = inst_norm(dark)

print("\n정규화 후:")
print(f"  밝은 이미지 — 평균: {bright_norm.mean():.4f}, 분산: {bright_norm.var():.4f}")
print(f"  어두운 이미지 — 평균: {dark_norm.mean():.4f}, 분산: {dark_norm.var():.4f}")
# → 둘 다 평균≈0, 분산≈1로 통일! (스타일 제거됨)

# === 통계 수 비교 ===
print("\n=== 정규화별 통계 개수 ===")
B, C, H, W = 4, 64, 32, 32

print(f"입력: ({B}, {C}, {H}, {W})")
print(f"BatchNorm:    {C}개 통계 (채널당 1개)")
print(f"InstanceNorm: {B * C}개 통계 (샘플×채널당 1개)")
print(f"LayerNorm:    {B}개 통계 (샘플당 1개)")
print(f"GroupNorm(32): {B * 32}개 통계 (샘플×그룹당 1개)")

# === AdaIN 데모 ===
print("\n=== AdaIN 데모 ===")
content = torch.randn(1, 64, 16, 16) * 2 + 3   # 평균 3, std 2
style = torch.randn(1, 64, 16, 16) * 0.5 - 1   # 평균 -1, std 0.5

result = adaptive_instance_norm(content, style)

print(f"콘텐츠 통계: mean={content.mean(dim=(2,3)).mean():.2f}, "
      f"std={content.std(dim=(2,3)).mean():.2f}")
print(f"스타일 통계: mean={style.mean(dim=(2,3)).mean():.2f}, "
      f"std={style.std(dim=(2,3)).mean():.2f}")
print(f"결과 통계:   mean={result.mean(dim=(2,3)).mean():.2f}, "
      f"std={result.std(dim=(2,3)).mean():.2f}")
# → 결과의 통계가 스타일의 통계와 비슷해짐!
```

---

## 핵심 정리

| 항목 | InstanceNorm | BatchNorm | GroupNorm |
|------|-------------|-----------|-----------|
| 정규화 단위 | 각 이미지, 각 채널 | 배치 전체, 각 채널 | 각 이미지, 각 그룹 |
| 제거하는 정보 | **스타일 (대비, 밝기)** | 배치 간 변동 | 그룹 내 변동 |
| 핵심 사용처 | **Style Transfer, GAN** | CNN 분류 | Detection, Diffusion |
| GroupNorm 관계 | G=C | - | 1 < G < C |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| InstanceNorm | 실시간 Style Transfer | 스타일 정보 제거 |
| AdaIN | Style Transfer, StyleGAN | 스타일 교체의 핵심 |
| Conditional IN | SPADE (Semantic Image Synthesis) | 공간적 스타일 제어 |
| InstanceNorm → GN | 통합 관점 | GroupNorm(G=C)로 이해 |

---

## 관련 콘텐츠

- [Batch Normalization](/ko/docs/components/normalization/batch-norm) — 선수 지식: 배치 기반 정규화
- [Group Normalization](/ko/docs/components/normalization/group-norm) — InstanceNorm의 일반화 (G=C가 InstanceNorm)
- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — Transformer의 정규화
