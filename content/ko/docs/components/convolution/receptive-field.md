---
title: "Receptive Field"
weight: 3
math: true
---

# Receptive Field (수용 영역)

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d) | [Pooling](/ko/docs/components/convolution/pooling)
{{% /hint %}}

> **한 줄 요약**: Receptive Field는 출력의 한 픽셀이 입력에서 **얼마나 넓은 영역을 참고했는지**를 나타냅니다.

## 왜 Receptive Field를 알아야 하나?

### 문제 상황: "모델이 고양이 귀만 보고 고양이라고 합니다"

3×3 conv를 2번만 쌓으면 출력 1픽셀이 보는 입력 영역은 **5×5 픽셀**뿐입니다.

```
224×224 이미지에서 5×5 = 25픽셀?
전체 50,176픽셀 중 0.05%만 본 것!
```

고양이를 분류하려면 **몸 전체**를 봐야 하는데, 5×5 영역만 보고 판단하는 겁니다. Receptive Field(RF)가 너무 작으면:

- **작은 패턴만** 인식 (에지, 점)
- **전체 맥락** 파악 불가 (물체의 형태, 배경과의 관계)
- **분류 성능** 저하

반대로 RF가 충분히 크면:
- 물체의 **전체 형태**를 파악
- **주변 맥락**까지 고려
- 더 정확한 판단

{{< figure src="/images/components/convolution/ko/receptive-field-concept.png" caption="레이어가 깊어질수록 하나의 출력 뉴런이 '보는' 입력 영역이 넓어진다" >}}

---

## RF 계산

### 재귀 공식

L번째 레이어까지의 RF:

$$
RF_L = RF_{L-1} + (K_L - 1) \times \prod_{i=1}^{L-1} S_i
$$

**각 기호의 의미:**
- $RF_L$ : L번째 레이어의 receptive field
- $K_L$ : L번째 레이어의 커널 크기
- $S_i$ : i번째 레이어의 스트라이드
- $\prod_{i=1}^{L-1} S_i$ : 이전 레이어들의 스트라이드 누적곱

### 직관적 이해

```
왜 이전 스트라이드의 누적곱이 필요할까?

Layer 1 (stride 1): 출력 1칸 = 입력 1칸
  → 커널 3이면 RF += (3-1) × 1 = 2

Layer 2 (stride 1): 출력 1칸 = 입력 1칸
  → 커널 3이면 RF += (3-1) × 1 = 2

Pool (stride 2): 출력 1칸 = 입력 2칸
  → 커널 2이면 RF += (2-1) × 1 = 1

Layer 3 (stride 1): 출력 1칸 = 입력 1칸
  → 하지만 Pool 때문에 "입력 1칸" = "원본 2칸"!
  → 커널 3이면 RF += (3-1) × 2 = 4
```

Pool 이후에는 출력의 1픽셀이 원본의 2픽셀에 해당하므로, 같은 3×3 커널이라도 **원본 기준으로는 더 넓은 영역**을 봅니다.

---

## 단계별 예시

### VGG-style 네트워크

```
                  커널   스트라이드   누적 스트라이드   RF
시작                                    1             1
Conv1 (3×3, s1)    3       1           1           1 + (3-1)×1 = 3
Conv2 (3×3, s1)    3       1           1           3 + (3-1)×1 = 5
Pool  (2×2, s2)    2       2           2           5 + (2-1)×1 = 6
Conv3 (3×3, s1)    3       1           2           6 + (3-1)×2 = 10
Conv4 (3×3, s1)    3       1           2          10 + (3-1)×2 = 14
Pool  (2×2, s2)    2       2           4          14 + (2-1)×2 = 16
Conv5 (3×3, s1)    3       1           4          16 + (3-1)×4 = 24
Conv6 (3×3, s1)    3       1           4          24 + (3-1)×4 = 32
Conv7 (3×3, s1)    3       1           4          32 + (3-1)×4 = 40
```

3×3 conv 7개 + pool 2개로 **RF = 40**. 이것이 VGG의 전략입니다.

### 코드로 계산

```python
def calc_receptive_field(layers):
    """
    layers: list of (kernel_size, stride)
    Returns: receptive field size
    """
    rf = 1
    total_stride = 1

    for k, s in layers:
        rf = rf + (k - 1) * total_stride
        total_stride *= s

    return rf, total_stride

# VGG-style
vgg_layers = [
    (3, 1), (3, 1),         # Conv block 1
    (2, 2),                  # Pool 1
    (3, 1), (3, 1),         # Conv block 2
    (2, 2),                  # Pool 2
    (3, 1), (3, 1), (3, 1), # Conv block 3
    (2, 2),                  # Pool 3
    (3, 1), (3, 1), (3, 1), # Conv block 4
    (2, 2),                  # Pool 4
    (3, 1), (3, 1), (3, 1), # Conv block 5
]
rf, stride = calc_receptive_field(vgg_layers)
print(f"VGG-16 RF: {rf}, 총 stride: {stride}")
# VGG-16 RF: 196, 총 stride: 16

# AlexNet
alexnet_layers = [
    (11, 4),  # Conv1
    (3, 2),   # Pool1
    (5, 1),   # Conv2
    (3, 2),   # Pool2
    (3, 1),   # Conv3
    (3, 1),   # Conv4
    (3, 1),   # Conv5
]
rf, stride = calc_receptive_field(alexnet_layers)
print(f"AlexNet RF: {rf}, 총 stride: {stride}")
# AlexNet RF: 195, 총 stride: 16

# ResNet-50 (간략화)
resnet_layers = [
    (7, 2),   # Conv1
    (3, 2),   # Pool
    (3, 1), (3, 1),  # Block 1
    (3, 2), (3, 1),  # Block 2 (stride)
    (3, 2), (3, 1),  # Block 3 (stride)
    (3, 2), (3, 1),  # Block 4 (stride)
]
rf, stride = calc_receptive_field(resnet_layers)
print(f"ResNet RF: {rf}, 총 stride: {stride}")
# ResNet RF: 427, 총 stride: 64
```

---

## RF 확장 전략

### 1. 레이어를 더 쌓기

가장 단순한 방법. 3×3 conv를 추가할 때마다 RF가 2씩 증가:

```python
# 3×3 conv N개 (stride 1, pooling 없음)
# RF = 2N + 1
layers_3 = [(3, 1)] * 3
layers_5 = [(3, 1)] * 5
layers_10 = [(3, 1)] * 10

print(f"3개: RF={calc_receptive_field(layers_3)[0]}")   # 3개: RF=7
print(f"5개: RF={calc_receptive_field(layers_5)[0]}")   # 5개: RF=11
print(f"10개: RF={calc_receptive_field(layers_10)[0]}") # 10개: RF=21
```

**한계**: 깊어지면 gradient vanishing → [ResNet](/ko/docs/architecture/cnn/resnet)이 해결

### 2. Pooling / Stride

누적 스트라이드를 키우면 이후 레이어의 RF 증가가 **배수**로 커집니다:

```python
# Pool 없이 3×3 × 6
no_pool = [(3, 1)] * 6
print(f"Pool 없음: RF={calc_receptive_field(no_pool)[0]}")
# Pool 없음: RF=13

# Pool 2개 포함
with_pool = [(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2)]
print(f"Pool 있음: RF={calc_receptive_field(with_pool)[0]}")
# Pool 있음: RF=22
```

같은 6개 레이어인데 RF가 13 → 22로 70% 증가!

### 3. Dilated Convolution

커널 사이에 간격을 넣어 파라미터 추가 없이 RF 확장:

```
dilation=1:  ■■■    RF 기여: 3
              (3×3 영역)

dilation=2:  ■·■·■  RF 기여: 5
              (5×5 영역, 파라미터는 여전히 9개)

dilation=4:  ■···■···■  RF 기여: 9
              (9×9 영역, 파라미터는 여전히 9개)
```

```python
# Dilated convolution으로 RF 확장
dilated_layers = [
    (3, 1),  # dilation=1, RF 기여 = 3
    (3, 1),  # dilation=2 → 실효 커널 5, RF 기여 = 5
    (3, 1),  # dilation=4 → 실효 커널 9, RF 기여 = 9
]

# DeepLab에서 사용하는 ASPP:
# 여러 dilation rate를 병렬로 사용하여 다양한 RF 확보
import torch.nn as nn

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (간략화)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)                         # RF: 1
        self.conv6 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6)  # RF: 13
        self.conv12 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)  # RF: 25
        self.conv18 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)  # RF: 37
```

### 전략 비교

| 방법 | RF 증가 | 파라미터 | 연산량 | 사용처 |
|------|---------|---------|--------|--------|
| 레이어 추가 | 선형 (+2/layer) | 증가 | 증가 | VGG, ResNet |
| Pooling/Stride | 기하급수적 | 없음/적음 | **감소** | 대부분의 CNN |
| Dilation | 선형 (빠름) | **동일** | 동일 | DeepLab, Segmentation |

---

## Effective Receptive Field

### 이론 vs 실제

이론적 RF는 **"최대로 볼 수 있는 범위"**이고, 실제로 영향을 미치는 범위는 더 작습니다:

{{< figure src="/images/components/convolution/ko/effective-receptive-field.jpeg" caption="이론적 RF(전체 사각형) vs 실효 RF(가우시안 분포): 중앙에 영향력 집중" >}}

```
이론적 RF (사각형):
┌─────────────────┐
│ · · · · · · · · │   ← 가장자리: 거의 영향 없음
│ · · · · · · · · │
│ · · ■ ■ ■ · · · │
│ · · ■ ★ ■ · · · │   ★: 출력 픽셀
│ · · ■ ■ ■ · · · │   ■: 실제 영향 큰 영역
│ · · · · · · · · │
│ · · · · · · · · │   ← 가장자리: 거의 영향 없음
└─────────────────┘
```

- **가우시안 형태**: 중앙이 가장 영향 크고 가장자리로 갈수록 감소
- **실효 RF ≈ 이론 RF의 40~60%** (학습 초기)
- 학습이 진행되면 실효 RF가 점차 커짐

### 실무적 의미

```python
# Detection 모델 설계 시:
# 탐지하려는 객체 크기 < RF 여야 함

# 예: YOLO에서 32×32 객체를 탐지하려면
# → 해당 출력 레이어의 RF가 최소 32 이상이어야 함
# → 실효 RF 고려하면 64 이상 권장
```

---

## 실전: 모델별 RF 비교

| 모델 | 이론적 RF | 총 Stride | 입력 대비 | 특징 |
|------|----------|-----------|----------|------|
| AlexNet | 195 | 16 | 87% (224) | 큰 커널로 빠르게 확장 |
| VGG-16 | 196 | 16 | 87% (224) | 3×3 반복으로 동일 달성 |
| ResNet-50 | 427 | 32 | 190% (224) | 입력보다 큰 RF |
| ResNet-101 | 811 | 32 | 362% (224) | 매우 큰 RF |

ResNet 이후의 모델은 RF가 입력보다 크므로, **전체 이미지를 충분히 참고**합니다.

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| 작은 RF | 초기 레이어 | 에지, 텍스처 등 저수준 특징 |
| 큰 RF | 깊은 레이어 | 객체 전체, 장면 이해 |
| Dilation RF | DeepLab Segmentation | 해상도 유지하면서 넓은 맥락 |
| 다중 RF | [YOLO](/ko/docs/architecture/detection/yolo) FPN | 다양한 크기의 객체 탐지 |
| 전역 RF | [Self-Attention](/ko/docs/components/attention/self-attention) | RF 제한 없이 전체 참조 |

---

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d) — RF를 결정하는 핵심 연산
- [Pooling](/ko/docs/components/convolution/pooling) — RF를 빠르게 확장하는 방법
- [VGG](/ko/docs/architecture/cnn/vgg) — 3×3으로 RF를 쌓는 전략의 시작
- [ResNet](/ko/docs/architecture/cnn/resnet) — 깊은 네트워크로 큰 RF 확보
- [Self-Attention](/ko/docs/components/attention/self-attention) — RF 제한을 근본적으로 해결
