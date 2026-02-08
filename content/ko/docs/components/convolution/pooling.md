---
title: "Pooling"
weight: 2
math: true
---

# Pooling (풀링)

{{% hint info %}}
**선수지식**: [Conv2D](/ko/docs/components/convolution/conv2d)
{{% /hint %}}

> **한 줄 요약**: Pooling은 특징 맵을 **축소**하여 중요한 정보만 남기고, 작은 위치 변화에 강건하게 만드는 연산입니다.

## 왜 Pooling이 필요한가?

### 문제 상황 1: "특징 맵이 너무 큽니다"

Conv2d를 여러 번 거치면 채널 수가 계속 늘어나면서 연산량이 폭발합니다:

```python
# ResNet 중간 단계
# 입력: (B, 256, 56, 56) → 하나의 값에 대해 3×3×256 = 2,304번 곱셈
# 전체: 2,304 × 56 × 56 = 7,225,344번 곱셈 (1개 필터당!)
```

공간 크기를 **절반으로 줄이면** 연산량이 **1/4**로 줄어듭니다.

### 문제 상황 2: "고양이가 살짝 움직이면 다른 결과가 나옵니다"

사진에서 고양이가 1픽셀 오른쪽으로 이동했다고 "다른 이미지"가 되면 안 됩니다.

```
원본:            1픽셀 이동:
[0][5][3][1]     [0][0][5][3]
 → 다른 특징 맵이 나옴!
```

Pooling은 작은 영역의 **대표값**만 취하므로, 미세한 이동에 강건해집니다.

{{< figure src="/images/components/convolution/ko/pooling-overview.jpeg" caption="Pooling의 두 가지 역할: 공간 축소(연산 감소)와 위치 불변성 확보" >}}

---

## Max Pooling

### 동작 원리

영역 내에서 **가장 큰 값**만 선택합니다:

$$
y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}
$$

- $R_{i,j}$ : (i, j) 위치에 해당하는 입력 영역 (보통 2×2)
- 가장 강한 활성화 = 가장 두드러진 특징 → 그것만 유지

### 직관적 이해

"이 영역에 **에지가 있었나?**" → 가장 강한 반응만 기억:

```
입력 (4×4):                    Max Pool 2×2, stride 2:

┌────┬────┬────┬────┐          ┌─────┬─────┐
│  1 │  3 │  2 │  1 │          │     │     │
├────┼────┤────┼────┤   →      │  6  │  8  │
│  5 │ [6]│  7 │ [8]│          │     │     │
├────┼────┼────┼────┤          ├─────┼─────┤
│  4 │  2 │  1 │  3 │          │     │     │
├────┼────┤────┼────┤   →      │ 14  │ 16  │
│ 10 │[14]│ 12 │[16]│          │     │     │
└────┴────┴────┴────┘          └─────┴─────┘
 4×4                            2×2 (75% 축소!)
```

### 코드

```python
import torch
import torch.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.tensor([[[[1.,  3.,  2.,  1.],
                     [5.,  6.,  7.,  8.],
                     [4.,  2.,  1.,  3.],
                     [10., 14., 12., 16.]]]])

y = pool(x)
print(y)
# tensor([[[[ 6.,  8.],
#           [14., 16.]]]])

print(f"입력: {x.shape} → 출력: {y.shape}")
# 입력: torch.Size([1, 1, 4, 4]) → 출력: torch.Size([1, 1, 2, 2])
```

### 특징

- **학습 파라미터 없음** — 그냥 max 연산
- **강한 특징만 보존** — 잡음에 강건
- **위치 정보 손실** — "어디에 있었는지"는 잊어버림 (단점이자 장점)

---

## Average Pooling

### 동작 원리

영역 내 모든 값의 **평균**을 구합니다:

$$
y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} x_{m,n}
$$

### Max vs Average 비교

```python
x = torch.tensor([[[[1., 3., 2., 1.],
                     [5., 6., 7., 8.],
                     [4., 2., 1., 3.],
                     [10., 14., 12., 16.]]]])

max_pool = nn.MaxPool2d(2, 2)
avg_pool = nn.AvgPool2d(2, 2)

print("Max:", max_pool(x))
# Max: tensor([[[[ 6.,  8.],
#                [14., 16.]]]])

print("Avg:", avg_pool(x))
# Avg: tensor([[[[ 3.75,  4.5 ],
#                [ 7.5 ,  8.  ]]]])
```

{{< figure src="/images/components/convolution/ko/pooling-max-avg.png" caption="Max Pooling은 가장 강한 특징, Average Pooling은 전체 분포를 유지" >}}

| 비교 | Max Pooling | Average Pooling |
|------|------------|-----------------|
| 동작 | 최댓값 선택 | 평균값 계산 |
| 특징 | 가장 강한 활성화 유지 | 전체 정보의 평균 유지 |
| 장점 | 에지, 텍스처에 강함 | 부드러운 특징 |
| 주 사용처 | CNN 중간 레이어 | Global Pooling, 마지막 레이어 |

---

## Global Average Pooling (GAP)

### 왜 필요한가?

CNN의 마지막에서 분류를 하려면 공간 차원을 제거해야 합니다. 전통적 방법은 FC layer:

```python
# 전통적 방법 (AlexNet, VGG):
# (B, 512, 7, 7) → flatten → (B, 25088) → FC(25088, 4096)
# 파라미터: 25,088 × 4,096 = 102,760,448 (약 1억 개!)
```

**Global Average Pooling**은 각 채널을 **하나의 숫자**로 요약합니다:

```python
# GAP 방법 (ResNet 이후 표준):
# (B, 512, 7, 7) → GAP → (B, 512) → FC(512, 1000)
# 파라미터: 512 × 1,000 = 512,000 (200배 감소!)
```

### 직관적 이해

```
채널 1 (고양이 귀 감지기):     채널 2 (바퀴 감지기):
┌───────────────┐              ┌───────────────┐
│ 0.1  0.0  0.0 │              │ 0.0  0.0  0.0 │
│ 0.0  0.9  0.8 │  → 평균 0.3  │ 0.0  0.0  0.0 │  → 평균 0.0
│ 0.0  0.7  0.2 │              │ 0.0  0.0  0.0 │
└───────────────┘              └───────────────┘

"고양이 귀가 좀 있다(0.3)" vs "바퀴는 없다(0.0)" → 고양이!
```

### 코드

```python
import torch
import torch.nn as nn

# Global Average Pooling
gap = nn.AdaptiveAvgPool2d(1)  # 출력을 (1, 1)로

x = torch.randn(32, 512, 7, 7)  # 어떤 크기든 OK
y = gap(x)                       # (32, 512, 1, 1)
y = y.flatten(1)                 # (32, 512)

print(f"입력: {x.shape}")  # 입력: torch.Size([32, 512, 7, 7])
print(f"출력: {y.shape}")  # 출력: torch.Size([32, 512])

# 분류 헤드로 직접 연결
classifier = nn.Linear(512, 1000)
logits = classifier(y)  # (32, 1000)
```

### GAP의 장점

1. **파라미터 극감** — FC 대비 수백 배 적음
2. **입력 크기 자유** — 224×224든 320×320이든 동작
3. **과적합 방지** — 파라미터가 적으므로
4. **해석 가능** — 각 채널이 하나의 "특징 존재 점수"

---

## Stride Convolution vs Pooling

최근에는 Max Pooling 대신 **stride=2 convolution**을 사용하는 추세입니다:

```python
# 방법 1: Pooling (전통적)
traditional = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2)           # 학습 불가, 정보 버림
)

# 방법 2: Stride Conv (현대적)
modern = nn.Sequential(
    nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 학습 가능한 다운샘플링
    nn.ReLU(),
)

x = torch.randn(1, 64, 56, 56)
print(traditional(x).shape)  # torch.Size([1, 64, 28, 28])
print(modern(x).shape)       # torch.Size([1, 64, 28, 28])
```

| 비교 | Max Pooling | Stride Conv |
|------|------------|-------------|
| 파라미터 | 없음 | 있음 (커널) |
| 학습 | 불가 | 가능 |
| 정보 보존 | 최댓값만 | 학습된 조합 |
| 사용 모델 | VGG, 초기 ResNet | ResNet-D, EfficientNet |

**결론**: Max Pooling은 여전히 유효하지만, 성능이 중요하면 stride convolution을 고려하세요.

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| Max Pooling 2×2 | [AlexNet](/ko/docs/architecture/cnn/alexnet), [VGG](/ko/docs/architecture/cnn/vgg) | 초기 CNN의 표준 다운샘플링 |
| Stride Conv | [ResNet](/ko/docs/architecture/cnn/resnet) 이후 | Pooling 대체 |
| GAP | [ResNet](/ko/docs/architecture/cnn/resnet) 이후 모든 CNN | FC layer 대체, 분류 헤드 |
| Adaptive Pooling | Detection ([YOLO](/ko/docs/architecture/detection/yolo)) | 다양한 입력 크기 대응 |

---

## 관련 콘텐츠

- [Conv2D](/ko/docs/components/convolution/conv2d) — Pooling과 함께 사용되는 핵심 연산
- [Receptive Field](/ko/docs/components/convolution/receptive-field) — Pooling이 RF를 확장하는 원리
- [AlexNet](/ko/docs/architecture/cnn/alexnet) — Max Pooling을 사용한 초기 CNN
- [ResNet](/ko/docs/architecture/cnn/resnet) — GAP를 도입한 모델
