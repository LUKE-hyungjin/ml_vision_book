---
title: "Dilated Convolution"
weight: 6
math: true
---

# Dilated Convolution (팽창 합성곱)

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d) | [Receptive Field](/ko/docs/components/convolution/receptive-field)
{{% /hint %}}

> **한 줄 요약**: Dilated Convolution은 커널 원소 사이에 **간격(hole)**을 넣어, 파라미터 수 변화 없이 **수용 영역(Receptive Field)을 기하급수적으로** 키우는 기법입니다.

## 왜 Dilated Convolution이 필요한가?

### 문제 상황: "넓게 보면서도 해상도를 유지하고 싶습니다"

**Semantic Segmentation**에서는 **모든 픽셀**에 클래스 라벨을 붙여야 합니다:

```python
# 입력: (3, 512, 512) → 출력: (21, 512, 512)
# 모든 픽셀에 대해 "이 픽셀이 뭐냐?" 판단
```

문제는 두 가지 요구가 충돌한다는 것입니다:

**1. 넓은 문맥 필요** — "이 픽셀이 사람인지 자동차인지" 판단하려면 주변을 넓게 봐야 합니다
- 일반 3×3 Conv 하나의 Receptive Field: **3×3** (너무 좁음!)

**2. 해상도 유지 필요** — 픽셀 단위 예측이므로 특징 맵을 줄이면 안 됩니다
- Pooling이나 Stride로 넓게 보면 → **해상도 손실!**

### 기존 해결 방법들의 문제

| 방법 | Receptive Field | 해상도 | 단점 |
|------|----------------|--------|------|
| 3×3 Conv 쌓기 | 점진적 증가 | 유지 | 레이어 매우 많이 필요 |
| 큰 커널 (7×7, 11×11) | 즉시 넓음 | 유지 | 파라미터 폭발 |
| Pooling + Upsampling | 넓음 | **손실** | 세밀한 경계 잃음 |

### 해결: "커널에 구멍을 뚫자!"

돋보기에 비유하면:
- **일반 Conv** = 작은 돋보기로 가까이 보기
- **Dilated Conv** = 같은 돋보기인데 **줌을 당겨서** 넓은 영역을 보기

파라미터 수는 같은데, 보는 영역만 넓어집니다!

![Dilated Convolution 시각화](/images/components/convolution/ko/dilated-conv-rates.png)

---

## 수식

### Dilated Convolution 연산

$$
\text{output}(i, j) = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \text{input}(i + d \cdot m, \; j + d \cdot n) \cdot \text{kernel}(m, n)
$$

**각 기호의 의미:**
- $d$ : **dilation rate** (팽창률) — 커널 원소 사이의 간격
  - $d=1$: 일반 Conv (간격 없음)
  - $d=2$: 원소 사이에 1칸 간격
  - $d=4$: 원소 사이에 3칸 간격
- $K$ : 커널 크기 (보통 3)
- 나머지는 일반 Conv와 동일

### 실효 커널 크기

dilation rate $d$일 때, $K \times K$ 커널의 실효 크기:

$$
K_{\text{eff}} = K + (K-1)(d-1) = d(K-1) + 1
$$

| 커널 크기 $K$ | Dilation $d$ | 실효 크기 $K_{\text{eff}}$ | 파라미터 수 |
|:---:|:---:|:---:|:---:|
| 3 | 1 | **3×3** | 9 |
| 3 | 2 | **5×5** | 9 |
| 3 | 4 | **9×9** | 9 |
| 3 | 8 | **17×17** | 9 |

파라미터는 항상 9개인데, 보는 영역은 3×3 → 17×17로 **32배 넓어집니다!**

### 직관적 이해

```
dilation=1 (일반):     dilation=2:           dilation=4:
■ ■ ■                  ■ · ■ · ■             ■ · · · ■ · · · ■
■ ■ ■                  · · · · ·             · · · · · · · · ·
■ ■ ■                  ■ · ■ · ■             · · · · · · · · ·
  3×3                  · · · · ·             · · · · · · · · ·
                       ■ · ■ · ■             ■ · · · ■ · · · ■
                         5×5                 · · · · · · · · ·
                                             · · · · · · · · ·
                                             · · · · · · · · ·
                                             ■ · · · ■ · · · ■
                                               9×9

■ = 커널이 실제로 보는 위치 (파라미터 9개)
· = 건너뛴 위치 (연산 없음)
```

---

## 출력 크기 계산

$$
O = \frac{I - d(K-1) - 1 + 2P}{S} + 1
$$

**Same padding** (출력 크기 = 입력 크기)을 위한 패딩:

$$
P = \frac{d(K-1)}{2}
$$

```python
import torch.nn as nn

# Same padding 예시
nn.Conv2d(64, 64, 3, padding=1, dilation=1)   # P=1: 일반
nn.Conv2d(64, 64, 3, padding=2, dilation=2)   # P=2: dilation=2
nn.Conv2d(64, 64, 3, padding=4, dilation=4)   # P=4: dilation=4

# 규칙: padding = dilation (K=3일 때)
```

---

## Dilation Rate 설계 전략

### 1. 순차적 증가 (DeepLab 스타일)

```python
# Receptive Field를 점진적으로 키움
layers = nn.Sequential(
    nn.Conv2d(256, 256, 3, padding=1,  dilation=1),   # RF:  3×3
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=2,  dilation=2),   # RF:  7×7
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=4,  dilation=4),   # RF: 15×15
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=8,  dilation=8),   # RF: 31×31
    nn.ReLU(inplace=True),
)
# 4개 레이어만으로 31×31 RF 확보!
# 일반 Conv로 같은 RF: 15개 레이어 필요
```

### 2. ASPP (Atrous Spatial Pyramid Pooling) — DeepLab v2/v3

**다양한 스케일의 문맥을 동시에** 포착합니다:

```python
class ASPP(nn.Module):
    """DeepLab v3의 핵심: 여러 dilation rate를 병렬로 적용"""
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        # 여러 dilation rate의 Conv를 병렬로!
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv_d6  = nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6)
        self.conv_d12 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)
        self.conv_d18 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)

        # Global Average Pooling (가장 넓은 문맥)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1),
        )

        # 합치고 1×1로 압축
        self.project = nn.Conv2d(out_ch * 5, out_ch, 1)

    def forward(self, x):
        size = x.shape[2:]
        features = [
            self.conv1x1(x),                  # 점 수준
            self.conv_d6(x),                  # 가까운 문맥
            self.conv_d12(x),                 # 중간 문맥
            self.conv_d18(x),                 # 먼 문맥
            nn.functional.interpolate(        # 전역 문맥
                self.pool(x), size=size,
                mode='bilinear', align_corners=False
            ),
        ]
        return self.project(torch.cat(features, dim=1))
```

```
ASPP의 직관:

입력 이미지 위에 여러 "줌 레벨"의 필터를 동시에 적용:

  rate=6:  가까운 동네          "이 픽셀 주변에 뭐가 있지?"
  rate=12: 넓은 이웃            "좀 더 넓게 보면?"
  rate=18: 먼 문맥              "전체적으로 어떤 장면이지?"
  GAP:     전체 이미지          "이 사진은 실내? 실외?"

→ 이 정보를 합쳐서 최종 판단!
```

### 3. Gridding Problem과 해결

연속된 같은 dilation rate는 **빈 공간(dead zone)**을 만들 수 있습니다:

```
dilation=2를 2번 쌓으면:
■ · ■ · ■       체크보드 패턴으로만 보게 됨!
· · · · ·       → 일부 픽셀 정보가 완전히 무시됨
■ · ■ · ■
· · · · ·
■ · ■ · ■

해결: HDC (Hybrid Dilated Convolution)
→ [1, 2, 3] 또는 [1, 2, 5]처럼 공약수 없는 조합 사용
```

```python
# Gridding Problem 방지: 서로 다른 dilation rate 혼합
hdc_block = nn.Sequential(
    nn.Conv2d(256, 256, 3, padding=1, dilation=1),   # d=1
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=2, dilation=2),   # d=2
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=5, dilation=5),   # d=5 (2와 공약수 없음!)
    nn.ReLU(inplace=True),
)
```

---

## Receptive Field 비교

같은 레이어 수일 때:

| 구성 | 3 레이어 후 RF | 파라미터 (채널 256) |
|------|:---:|---:|
| 3×3 Conv × 3 | **7 × 7** | 3 × 9 × 256² = 1,769,472 |
| Dilated [1,2,4] | **15 × 15** | 3 × 9 × 256² = 1,769,472 |
| 5×5 Conv × 3 | **13 × 13** | 3 × 25 × 256² = 4,915,200 |

Dilated Conv는 **같은 파라미터로 2배 이상 넓은 RF**를 확보합니다.

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === Dilation에 따른 실효 크기 확인 ===
print("=== Dilation Rate와 실효 커널 크기 ===")
for d in [1, 2, 4, 8, 16]:
    K_eff = 3 + 2 * (d - 1)   # K=3일 때
    padding = d               # Same padding
    conv = nn.Conv2d(64, 64, 3, padding=padding, dilation=d)
    x = torch.randn(1, 64, 64, 64)
    y = conv(x)
    print(f"dilation={d:>2}: 실효 크기 {K_eff:>2}×{K_eff:<2}  "
          f"padding={padding:>2}  출력={y.shape[-1]}×{y.shape[-1]}")

# dilation= 1: 실효 크기  3×3   padding= 1  출력=64×64
# dilation= 2: 실효 크기  5×5   padding= 2  출력=64×64
# dilation= 4: 실효 크기  9×9   padding= 4  출력=64×64
# dilation= 8: 실효 크기 17×17  padding= 8  출력=64×64
# dilation=16: 실효 크기 33×33  padding=16  출력=64×64

# === 일반 Conv vs Dilated Conv 비교 ===
print("\n=== 파라미터 수 비교 ===")

# 같은 실효 크기를 얻으려면?
conv_5x5 = nn.Conv2d(64, 64, 5, padding=2, bias=False)      # 5×5
conv_d2  = nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False)  # dilation=2 → 실효 5×5

print(f"5×5 Conv:      {sum(p.numel() for p in conv_5x5.parameters()):,} 파라미터")
print(f"3×3 dilation=2: {sum(p.numel() for p in conv_d2.parameters()):,} 파라미터")
# 5×5 Conv:      102,400 파라미터
# 3×3 dilation=2:  36,864 파라미터  (64% 절감!)

# === Receptive Field 비교 ===
print("\n=== 3개 레이어 쌓기: RF 비교 ===")

# 일반 3×3 × 3: RF = 7×7
normal_rf = 1
for _ in range(3):
    normal_rf = normal_rf + 2  # 일반 3×3은 매 레이어 +2

# Dilated [1,2,4]: RF = ?
dilated_rf = 1
for d in [1, 2, 4]:
    dilated_rf = dilated_rf + 2 * d  # dilation d의 3×3은 +2d

print(f"일반 3×3 × 3:    RF = {normal_rf}×{normal_rf}")
print(f"Dilated [1,2,4]: RF = {dilated_rf}×{dilated_rf}")
# 일반 3×3 × 3:    RF = 7×7
# Dilated [1,2,4]: RF = 15×15
```

---

## 핵심 정리

| 항목 | 일반 Conv | Dilated Conv |
|------|----------|-------------|
| 커널 형태 | 빽빽하게 채움 | 간격을 두고 배치 |
| Receptive Field | $2L+1$ (L 레이어) | 기하급수적 증가 |
| 파라미터 | $K^2 \cdot C_{in} \cdot C_{out}$ | **동일!** |
| 해상도 | stride 없이 유지 가능 | stride 없이 유지 가능 |
| 핵심 사용처 | 범용 | Segmentation, 음성 처리 |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| Dilated Conv | DeepLab v1/v2/v3 | Segmentation 해상도 유지 |
| ASPP | DeepLab v2/v3 | 다중 스케일 문맥 포착 |
| HDC | Multi-scale 모델 | Gridding problem 해결 |
| WaveNet | 음성 생성 | 긴 시퀀스의 넓은 문맥 |

---

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d) — 선수 지식: 기본 합성곱 연산
- [Receptive Field](/ko/docs/components/convolution/receptive-field) — Dilated Conv의 RF 계산
- [Pooling](/ko/docs/components/convolution/pooling) — 다운샘플링의 대안으로서의 Dilated Conv
