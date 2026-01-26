---
title: "StyleGAN"
weight: 7
math: true
---

# StyleGAN

{{% hint info %}}
**선수지식**: [GAN](/ko/docs/architecture/generative/gan) | [CNN 기초](/ko/docs/architecture/cnn)
{{% /hint %}}

## 한 줄 요약

> **"스타일을 층층이 주입해서 고품질 얼굴을 생성한다"**

---

## 왜 StyleGAN인가?

### 기존 GAN의 문제

```
기존 GAN:
노이즈 z → [Generator] → 이미지

문제: z 하나로 모든 걸 제어? 너무 어려움!
- 얼굴형, 머리색, 표정을 각각 제어 불가능
```

### StyleGAN의 해결책

```
StyleGAN:
노이즈 z → [Mapping Network] → w (스타일 벡터)
                                ↓
                         각 층에 스타일 주입!
                                ↓
                            고품질 이미지
```

> **비유**: 기존 GAN은 "대충 그려줘"라고 하는 것이고, StyleGAN은 "머리는 이렇게, 눈은 이렇게, 피부는 이렇게"라고 **세부 지시**를 하는 것입니다!

---

## 핵심 구조

```
┌─────────────────────────────────────────────────┐
│                   StyleGAN                       │
│                                                  │
│  z (512) ──→ [Mapping Network] ──→ w (512)      │
│              (8개 FC층)                          │
│                                                  │
│  Constant ──→ [Synthesis Network] ──→ 이미지    │
│   4×4        각 층에 w 주입 (AdaIN)              │
│              4×4 → 8×8 → ... → 1024×1024        │
└─────────────────────────────────────────────────┘
```

### 구성 요소

| 컴포넌트 | 역할 | 비유 |
|----------|------|------|
| **Mapping Network** | z → w 변환 | "스타일 번역가" |
| **Synthesis Network** | 이미지 생성 | "실제로 그리는 화가" |
| **AdaIN** | 스타일 주입 | "붓 터치 스타일 적용" |

---

## Mapping Network

### 왜 필요한가?

노이즈 $z$는 **구형(spherical)** 분포:
- 얽혀있음 (entangled)
- 얼굴형 바꾸면 머리색도 같이 바뀜

$w$ 공간은 **풀려있음 (disentangled)**:
- 각 차원이 독립적 의미
- 얼굴형만 바꾸기 가능!

```
z 공간: 모든 게 섞여있음 (구슬 뭉치)
    ↓ Mapping Network
w 공간: 정리정돈됨 (서랍장)
```

### 수식

$$
w = f(z)
$$

- $f$: 8개의 FC (Fully Connected) 층
- 입력 $z$: 512차원 가우시안 노이즈
- 출력 $w$: 512차원 스타일 벡터

---

## AdaIN (Adaptive Instance Normalization)

### 핵심 아이디어

각 층에서 스타일을 **정규화 후 재조정**합니다.

$$
\text{AdaIN}(x, y) = y_s \cdot \frac{x - \mu(x)}{\sigma(x)} + y_b
$$

**기호 설명:**
- $x$: 입력 feature map
- $\mu(x), \sigma(x)$: 평균, 표준편차 (정규화용)
- $y_s, y_b$: 스타일 벡터 $w$에서 유도된 스케일, 바이어스

**해석:**
1. 기존 스타일 제거 (정규화)
2. 새 스타일 주입 ($y_s$, $y_b$)

---

## Progressive Growing

### 아이디어

작은 해상도에서 시작해 점점 키웁니다.

```
4×4 학습 완료
   ↓ 층 추가
8×8 학습
   ↓ 층 추가
16×16 학습
   ↓ ...
1024×1024 학습 완료!
```

**장점:**
- 안정적 학습
- 고해상도 도달 가능
- 학습 시간 단축

---

## StyleGAN 버전 비교

| 버전 | 연도 | 핵심 개선 |
|------|------|----------|
| **StyleGAN** | 2018 | Mapping Network, AdaIN |
| **StyleGAN2** | 2019 | 물방울 아티팩트 제거, Weight Demodulation |
| **StyleGAN3** | 2021 | Alias-free, 애니메이션 개선 |

### StyleGAN2 개선점

```
StyleGAN 문제:
- 물방울 무늬 아티팩트 발생
- AdaIN이 원인

StyleGAN2 해결:
- AdaIN → Weight Demodulation
- 더 깨끗한 결과!
```

---

## 스타일 믹싱

두 이미지의 스타일을 섞을 수 있습니다!

```python
# w1: 사람 A의 스타일
# w2: 사람 B의 스타일

# 낮은 층 (4×4 ~ 8×8): 전체 구조 (얼굴형)
# 중간 층 (16×16 ~ 32×32): 얼굴 특징 (눈, 코, 입)
# 높은 층 (64×64 ~): 색상, 질감

# A의 얼굴형 + B의 색상
w_mix = [w1] * 4 + [w2] * 14
```

| 층 범위 | 제어하는 것 |
|---------|-------------|
| 낮은 층 (Coarse) | 포즈, 얼굴형, 안경 |
| 중간 층 (Middle) | 얼굴 특징, 헤어스타일 |
| 높은 층 (Fine) | 색상, 미세한 특징 |

---

## 코드 예시

```python
# StyleGAN2 사용 (stylegan2-ada-pytorch)
import torch
import dnnlib
import legacy

# 모델 로드
with dnnlib.util.open_url('ffhq.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].cuda()

# 노이즈 생성
z = torch.randn(1, G.z_dim).cuda()

# 이미지 생성
img = G(z, None)  # [1, 3, 1024, 1024]

# w 공간에서 직접 생성 (더 좋은 제어)
w = G.mapping(z, None)  # [1, 18, 512]
img = G.synthesis(w)
```

---

## 응용 분야

| 응용 | 설명 |
|------|------|
| **얼굴 생성** | 가상 인물 생성 |
| **얼굴 편집** | 나이, 표정, 스타일 변경 |
| **도메인 변환** | 사진→그림, 사람→동물 |
| **데이터 증강** | 학습 데이터 생성 |

---

## 한계와 발전

### 한계

- 얼굴 외 도메인에서 품질 저하
- 학습에 많은 데이터 필요
- Diffusion 모델 대비 다양성 부족

### 발전

```
StyleGAN (2018)
    ↓
StyleGAN2 (2019) - 품질 개선
    ↓
StyleGAN3 (2021) - 영상 생성 개선
    ↓
... Diffusion이 대세가 됨 (2022~)
```

---

## 요약

| 질문 | 답변 |
|------|------|
| StyleGAN이 뭔가요? | 스타일을 층별로 주입하는 GAN |
| w 공간이 왜 좋나요? | 각 특성을 독립적으로 제어 가능 |
| 스타일 믹싱이 뭔가요? | 두 이미지의 스타일을 층별로 조합 |
| 지금도 쓰이나요? | 얼굴 생성에는 여전히 강력함 |

---

## 관련 콘텐츠

- [GAN](/ko/docs/architecture/generative/gan) - 기본 GAN 구조
- [VAE](/ko/docs/architecture/generative/vae) - 다른 생성 모델
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - 현재 주류
- [Generation 태스크](/ko/docs/task/generation) - 평가 지표
