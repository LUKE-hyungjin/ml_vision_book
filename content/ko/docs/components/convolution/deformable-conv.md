---
title: "Deformable Convolution"
weight: 7
math: true
---

# Deformable Convolution (변형 합성곱)

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d)
{{% /hint %}}

> **한 줄 요약**: Deformable Convolution은 고정된 정사각형 격자 대신, **학습 가능한 오프셋**으로 커널의 샘플링 위치를 자유롭게 변형하여 **불규칙한 형태의 객체**를 더 잘 포착합니다.

## 왜 Deformable Convolution이 필요한가?

### 문제 상황: "정사각형 필터로 비정형 객체를 봐야 합니다"

일반 Conv의 커널은 항상 **고정된 정사각형 격자**입니다:

```
일반 3×3 Conv의 샘플링 위치:
■ ■ ■
■ ■ ■    ← 항상 이 모양, 예외 없음!
■ ■ ■
```

하지만 현실의 객체는 정사각형이 아닙니다:

```
긴 파이프:  ■ ■ ■ ■ ■ ■ ■    (가로로 길쭉)

구부러진 도로:    ■
              ■ ■             (곡선 형태)
            ■ ■
          ■

군중 속 사람:   ■
              ■ ■             (세로로 길쭉)
              ■ ■
                ■
```

정사각형 필터로 이런 객체를 포착하면 **배경 픽셀까지 포함**하거나 **객체 일부를 놓칩니다**.

### 핵심 관찰: "필터가 객체 형태에 맞춰 변형되면 좋겠다"

| 기존 방법 | 한계 |
|----------|------|
| 큰 커널 | 주변 배경까지 포함 |
| 다양한 커널 크기 혼합 | 미리 정한 모양만 가능 |
| Data augmentation | 학습 시만 도움, 추론 시 고정 |
| [Dilated Conv](/ko/docs/components/convolution/dilated-conv) | 정사각형 구조는 동일 |

### 해결: "커널의 위치를 학습하자!"

카메라에 비유하면:
- **일반 Conv** = 고정된 격자 센서 — 항상 같은 패턴으로 빛을 수집
- **Deformable Conv** = 유연한 센서 — 객체 형태에 맞게 센서 위치가 이동

![Deformable Convolution 오프셋](/images/components/convolution/ko/deformable-conv-offsets.jpeg)

---

## 수식

### 일반 Conv (복습)

$$
y(\mathbf{p}_0) = \sum_{\mathbf{p}_n \in \mathcal{R}} w(\mathbf{p}_n) \cdot x(\mathbf{p}_0 + \mathbf{p}_n)
$$

**각 기호의 의미:**
- $\mathbf{p}_0$ : 출력 위치 (현재 처리 중인 위치)
- $\mathcal{R}$ : 커널의 정규 격자. 예: 3×3이면 $\{(-1,-1), (-1,0), ..., (1,1)\}$
- $w(\mathbf{p}_n)$ : 커널 가중치
- $x(\mathbf{p}_0 + \mathbf{p}_n)$ : 입력에서 샘플링하는 위치 — **고정!**

### Deformable Conv

$$
y(\mathbf{p}_0) = \sum_{\mathbf{p}_n \in \mathcal{R}} w(\mathbf{p}_n) \cdot x(\mathbf{p}_0 + \mathbf{p}_n + \Delta \mathbf{p}_n)
$$

차이점은 $\Delta \mathbf{p}_n$뿐입니다!

**추가된 기호:**
- $\Delta \mathbf{p}_n$ : **학습 가능한 오프셋** — 각 커널 위치마다 (dx, dy) 이동량
  - 이 값은 별도의 Conv 레이어가 예측
  - 실수값이므로 $\Delta \mathbf{p}_n = (0.5, -0.3)$ 같은 비정수 이동도 가능

### Bilinear Interpolation

오프셋이 실수일 수 있으므로, 정수 격자 사이의 값을 **이중선형 보간**으로 구합니다:

$$
x(\mathbf{p}) = \sum_{\mathbf{q}} G(\mathbf{q}, \mathbf{p}) \cdot x(\mathbf{q})
$$

- $\mathbf{q}$ : $\mathbf{p}$ 주변 4개 정수 좌표
- $G(\mathbf{q}, \mathbf{p}) = \max(0, 1-|q_x - p_x|) \cdot \max(0, 1-|q_y - p_y|)$ : 거리 기반 가중치

```
p = (1.3, 2.7)이면 주변 4개 점에서 보간:

(1,2)───(2,2)
  │   p   │
(1,3)───(2,3)

x(p) = 0.7×0.3×x(1,2) + 0.3×0.3×x(2,2)
     + 0.7×0.7×x(1,3) + 0.3×0.7×x(2,3)
```

이 연산은 미분 가능하므로 **역전파로 오프셋을 학습**할 수 있습니다!

---

## 구조

### 전체 아키텍처

```
입력 특징 맵 (C × H × W)
       │
       ├──→ [오프셋 Conv] ──→ 오프셋 맵 (2K² × H × W)
       │         3×3 Conv       각 위치마다 K² 개의 (dx, dy)
       │
       └──→ [Deformable Conv] ←── 오프셋 적용
                  │
                  ▼
            출력 (C' × H × W)
```

**오프셋 Conv의 출력 채널 수:**
- $3 \times 3$ 커널 → $9$개 위치 → 각각 $(dx, dy)$ → $2 \times 9 = 18$ 채널

### 구현

```python
import torch
import torch.nn as nn

class DeformableConv2d(nn.Module):
    """Deformable Convolution v1 (간략화된 구현)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        K2 = kernel_size * kernel_size

        # 오프셋 예측 Conv: 입력에서 각 위치의 (dx, dy)를 예측
        self.offset_conv = nn.Conv2d(
            in_ch, 2 * K2,      # 2 × K² 채널 (각 커널 위치의 dx, dy)
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        # 오프셋 초기값: 0 (일반 Conv처럼 시작)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # 실제 Convolution
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        # 1. 오프셋 예측
        offsets = self.offset_conv(x)  # (B, 2K², H, W)

        # 2. 오프셋을 적용하여 입력을 재샘플링
        x_deformed = self._deform_input(x, offsets)

        # 3. 일반 Conv 적용
        return self.conv(x_deformed)

    def _deform_input(self, x, offsets):
        """오프셋을 적용하여 입력을 변형 (간략화)"""
        # 실제 구현에서는 grid_sample 또는 전용 CUDA 커널 사용
        # torchvision.ops.deform_conv2d 권장
        pass
```

### 실전에서는 torchvision 사용

```python
import torchvision.ops as ops

# torchvision의 구현 (CUDA 최적화)
class DeformConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        K2 = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_ch, 2 * K2, kernel_size,
                                      padding=kernel_size // 2)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        offsets = self.offset_conv(x)
        return ops.deform_conv2d(
            x, offsets, self.weight, self.bias,
            padding=1
        )
```

---

## Deformable Conv v2: Modulated Deformable Conv

### v1의 문제

오프셋만으로는 "이 위치를 아예 무시하고 싶다"는 것을 표현하기 어렵습니다.

### v2의 해결: 가중치(modulation) 추가

$$
y(\mathbf{p}_0) = \sum_{\mathbf{p}_n \in \mathcal{R}} w(\mathbf{p}_n) \cdot \Delta m_n \cdot x(\mathbf{p}_0 + \mathbf{p}_n + \Delta \mathbf{p}_n)
$$

**추가된 기호:**
- $\Delta m_n$ : **modulation scalar** — 각 커널 위치의 중요도 (0~1)
  - $0$: 이 위치 완전 무시
  - $1$: 이 위치 완전 사용

```python
class DeformConvV2Block(nn.Module):
    """Deformable Conv v2: 오프셋 + modulation"""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        K2 = kernel_size * kernel_size

        # 오프셋 (2K²) + modulation (K²) = 3K² 채널
        self.offset_modulation_conv = nn.Conv2d(
            in_ch, 3 * K2, kernel_size,
            padding=kernel_size // 2
        )
        nn.init.zeros_(self.offset_modulation_conv.weight)
        nn.init.zeros_(self.offset_modulation_conv.bias)

        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        out = self.offset_modulation_conv(x)
        K2 = self.weight.shape[-1] ** 2

        offsets = out[:, :2*K2]
        mask = torch.sigmoid(out[:, 2*K2:])  # 0~1로 제한

        return ops.deform_conv2d(
            x, offsets, self.weight, self.bias,
            padding=1, mask=mask
        )
```

---

## 학습된 오프셋 시각화

학습 후 오프셋을 시각화하면, 커널이 **객체 형태에 맞게 변형**된 것을 확인할 수 있습니다:

```
일반 Conv:          학습된 Deformable Conv:

■ ■ ■               ■     ■
■ ■ ■    →            ■ ■          객체(사람)의 윤곽을 따라
■ ■ ■               ■   ■         샘플링 위치가 이동!
                      ■ ■
```

실제 논문에서 관찰된 패턴:
- **큰 객체**: 오프셋이 바깥으로 퍼짐 (넓게 봄)
- **작은 객체**: 오프셋이 안쪽으로 모임 (집중해서 봄)
- **긴 객체**: 오프셋이 객체 방향으로 늘어남

---

## 파라미터 비교

```python
import torch.nn as nn

C = 256

# 일반 Conv
conv = nn.Conv2d(C, C, 3, padding=1, bias=False)
# 파라미터: 3 × 3 × 256 × 256 = 589,824

# Deformable Conv (v1)
deform_offset = nn.Conv2d(C, 2*9, 3, padding=1)   # 오프셋: 4,626
deform_conv = nn.Conv2d(C, C, 3, padding=1, bias=False)   # Conv: 589,824
# 총: 594,450 (약 0.8% 증가)

# Deformable Conv (v2)
deform_v2_offset = nn.Conv2d(C, 3*9, 3, padding=1)  # 오프셋+mask: 6,939
# 총: 596,763 (약 1.2% 증가)

print("추가 비용: 매우 적음!")
print("→ 오프셋 Conv는 출력 채널이 18~27로 매우 작기 때문")
```

**핵심**: 파라미터는 거의 안 늘지만, 연산량(FLOPs)은 bilinear interpolation 때문에 증가합니다.

---

## 사용처

### Detection: Deformable DETR 이전의 Detection 모델

```python
# Faster R-CNN, Cascade R-CNN 등에서
# backbone의 일부 Conv를 Deformable로 교체
class ResNetWithDCN(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 3, 4에서만 Deformable Conv 사용 (고수준 특징)
        self.stage1 = ResBlock(conv_type='regular')
        self.stage2 = ResBlock(conv_type='regular')
        self.stage3 = ResBlock(conv_type='deformable')  # ← 여기서부터!
        self.stage4 = ResBlock(conv_type='deformable')
```

### Segmentation: 불규칙 경계 처리

```python
# 의미론적 분할에서 객체 경계를 정밀하게 포착
# 특히 가늘고 긴 객체(기둥, 도로 표시) 인식에 효과적
```

---

## 코드로 확인하기

```python
import torch
import torchvision.ops as ops

# === torchvision의 Deformable Conv 사용 ===
B, C, H, W = 1, 64, 32, 32
K = 3

x = torch.randn(B, C, H, W)

# 오프셋 예측
offset_conv = nn.Conv2d(C, 2 * K * K, K, padding=1)
nn.init.zeros_(offset_conv.weight)
nn.init.zeros_(offset_conv.bias)
offsets = offset_conv(x)  # (1, 18, 32, 32)

# Deformable Conv 가중치
weight = torch.randn(128, C, K, K)
bias = torch.zeros(128)

# Deformable Conv 실행
output = ops.deform_conv2d(x, offsets, weight, bias, padding=1)
print(f"입력: {x.shape}")          # [1, 64, 32, 32]
print(f"오프셋: {offsets.shape}")   # [1, 18, 32, 32]
print(f"출력: {output.shape}")      # [1, 128, 32, 32]

# === 오프셋 초기값 = 0이면 일반 Conv와 동일 ===
conv_normal = nn.Conv2d(C, 128, K, padding=1, bias=True)
# 같은 가중치 사용
conv_weight = conv_normal.weight.data.clone()
conv_bias = conv_normal.bias.data.clone()

# 오프셋이 0이면 결과 동일
zero_offsets = torch.zeros(B, 2*K*K, H, W)
y_deform = ops.deform_conv2d(x, zero_offsets, conv_weight, conv_bias, padding=1)
y_normal = conv_normal(x)

print(f"\n오프셋=0일 때 차이: {(y_deform - y_normal).abs().max().item():.6f}")
# 거의 0 → 일반 Conv와 동일!
```

---

## 핵심 정리

| 항목 | 일반 Conv | Deformable Conv |
|------|----------|-----------------|
| 샘플링 위치 | 고정 정사각형 격자 | 학습된 자유 형태 |
| 추가 파라미터 | 없음 | 오프셋 Conv (~1% 추가) |
| 학습 가능한 것 | 커널 가중치 | 가중치 + **샘플링 위치** |
| 장점 | 빠르고 단순 | 불규칙 객체 포착 |
| 단점 | 형태 적응 불가 | 연산량 증가, 구현 복잡 |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| Deformable Conv v1 | DCN (ICCV 2017) | 기하학적 변환 학습 |
| Deformable Conv v2 | DCNv2 (CVPR 2019) | modulation으로 정밀도 향상 |
| Deformable Attention | Deformable DETR | Transformer에도 같은 아이디어 적용 |
| Offset learning | InternImage | 대규모 모델에서 DCN 활용 |

---

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d) — 선수 지식: 일반 합성곱
- [Dilated Convolution](/ko/docs/components/convolution/dilated-conv) — 또 다른 RF 확장 방법 (고정 패턴)
- [Receptive Field](/ko/docs/components/convolution/receptive-field) — Deformable Conv의 RF는 데이터 의존적
