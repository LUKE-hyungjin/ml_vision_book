---
title: "Sampling"
weight: 3
math: true
---

# Diffusion Sampling (샘플링)

{{% hint info %}}
**선수지식**: [DDPM](/ko/docs/components/generative/ddpm) | [미분방정식 기초](/ko/docs/math/calculus)
{{% /hint %}}

## 한 줄 요약

> **샘플링 = "노이즈에서 이미지로 돌아가는 과정"**

---

## 왜 샘플링이 중요한가?

학습은 끝났습니다. 이제 **새로운 이미지를 만들어야** 합니다!

```
학습: 이미지 → 노이즈 (Forward)
생성: 노이즈 → 이미지 (Reverse) ← 이게 샘플링!
```

문제는... 1000번 반복하면 **너무 느립니다**. 더 빨리 할 수 없을까요?

---

## 기본: DDPM 샘플링 (1000 스텝)

### 원리

한 스텝씩 노이즈를 조금씩 제거합니다.

```
x_T (순수 노이즈)
 ↓ 노이즈 조금 제거
x_{T-1}
 ↓ 노이즈 조금 제거
x_{T-2}
 ↓ ... (1000번 반복)
x_0 (깨끗한 이미지)
```

### 수식 (쉽게 설명)

$$
x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}}_{\text{스케일 조정}} \left( x_t - \underbrace{\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}}_{\text{노이즈 비율}} \cdot \underbrace{\epsilon_\theta(x_t, t)}_{\text{예측된 노이즈}} \right) + \underbrace{\sigma_t z}_{\text{약간의 랜덤성}}
$$

**직관적 해석:**
1. 현재 이미지 $x_t$에서
2. **예측된 노이즈**를 빼고
3. **스케일을 조정**한 뒤
4. 약간의 **랜덤성**을 추가

> 왜 랜덤성을 추가하나요? → 다양한 결과를 만들기 위해!

---

## DDIM: 50스텝으로 줄이기

### 핵심 아이디어

> "1000스텝이 필요한가? 중간을 건너뛰면 안 될까?"

DDPM: 1000 → 999 → 998 → ... → 1 → 0
DDIM: 1000 → 980 → 960 → ... → 20 → 0 (50스텝만!)

### 작동 원리

DDIM은 **결정론적(deterministic)**입니다. 같은 노이즈로 시작하면 같은 이미지가 나옵니다.

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \underbrace{\hat{x}_0}_{\text{원본 예측}} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \underbrace{\epsilon_\theta}_{\text{노이즈 방향}}
$$

**핵심**: 매 스텝마다 "원본 이미지가 뭐였을까?"를 예측합니다.

### 원본 예측 ($\hat{x}_0$)

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}
$$

**해석**: 현재 이미지에서 노이즈를 빼면 원본이 나온다!

---

## 비유로 이해하기

### DDPM vs DDIM

| DDPM | DDIM |
|------|------|
| 계단을 한 칸씩 내려감 | 에스컬레이터로 빠르게 내려감 |
| 1000스텝 | 50스텝 |
| 매번 조금씩 랜덤 | 결정론적 (같은 결과) |
| 다양한 결과 | 동일한 결과 |

```
DDPM:  🚶 한 칸씩... 한 칸씩... (느림)
DDIM:  🛗 쭉~~ (빠름)
```

---

## DPM-Solver: 20스텝으로 더 빠르게

### 핵심 아이디어

> "미분방정식을 더 정밀하게 풀면 스텝을 더 줄일 수 있다!"

수학적으로, Diffusion은 **미분방정식(ODE)**입니다:

$$
dx = f(x, t) dt
$$

DPM-Solver는 이 ODE를 더 정확하게 풀어서 스텝을 줄입니다.

### 1차 vs 2차

| 차수 | 스텝 | 설명 |
|------|------|------|
| 1차 | ~50 | 직선으로 근사 |
| 2차 | ~20 | 곡선으로 근사 (더 정확) |

```
실제 경로:     ⌒⌒⌒
1차 근사:     ╱╱╱ (직선)
2차 근사:     ⌒⌒⌒ (곡선, 더 정확!)
```

---

## Classifier-Free Guidance (CFG)

### 문제: 프롬프트를 잘 안 따른다

"고양이"라고 했는데 대충 비슷한 동물이 나옴...

### 해결: 조건을 더 강하게!

$$
\epsilon_{\text{CFG}} = \underbrace{\epsilon_{\text{무조건}}}_{\text{아무거나}} + s \cdot (\underbrace{\epsilon_{\text{조건}}}_{\text{고양이}} - \underbrace{\epsilon_{\text{무조건}}}_{\text{아무거나}})
$$

**해석:**
- $\epsilon_{\text{조건}} - \epsilon_{\text{무조건}}$ = "고양이"와 "아무거나"의 **차이**
- 이 차이를 $s$배 **강화**
- $s=1$: 그냥 조건부 생성
- $s=7.5$: 고양이 특성 7.5배 강화! (권장)
- $s=20$: 너무 강해서 이상해짐

### 직관적 이해

```
s=1:   "대충 고양이 같은 것"
s=7.5: "확실한 고양이" ← 권장
s=15:  "과하게 고양이스러운 것"
s=30:  "고양이의 악몽" (품질 저하)
```

---

## 샘플링 방법 비교

| 방법 | 스텝 수 | 속도 | 품질 | 특징 |
|------|---------|------|------|------|
| **DDPM** | 1000 | 🐢 매우 느림 | ⭐⭐⭐ | 기본, 다양성 높음 |
| **DDIM** | 50 | 🚗 보통 | ⭐⭐⭐ | 결정론적 |
| **DPM-Solver** | 20 | 🚀 빠름 | ⭐⭐⭐ | ODE 기반 |
| **LCM** | 4-8 | ✈️ 매우 빠름 | ⭐⭐ | Latent Consistency |

---

## 실전 팁

### 1. 어떤 샘플러를 쓸까?

```python
# 품질 중시
scheduler = "DDPM"
steps = 1000

# 균형 (권장)
scheduler = "DPM++ 2M Karras"
steps = 20-30

# 속도 중시
scheduler = "LCM"
steps = 4-8
```

### 2. CFG 설정

```python
# 창의적인 결과 원할 때
guidance_scale = 3-5

# 프롬프트에 충실하게 (권장)
guidance_scale = 7-8

# 매우 충실하게 (과적합 주의)
guidance_scale = 10-15
```

### 3. 시드(Seed) 고정

```python
# 같은 결과 재현
torch.manual_seed(42)
```

---

## 요약

| 질문 | 답변 |
|------|------|
| 샘플링이 뭔가요? | 노이즈에서 이미지를 만드는 과정 |
| DDIM이 더 빠른 이유? | 스텝을 건너뛰어도 됨 |
| CFG가 뭔가요? | 조건을 강화해서 프롬프트를 잘 따르게 함 |
| 권장 설정은? | DPM++ 2M, 20스텝, CFG 7.5 |

---

## 관련 콘텐츠

- [DDPM](/ko/docs/components/generative/ddpm) - 기본 Diffusion 모델
- [Score Matching](/ko/docs/components/generative/score-matching) - Score 기반 관점
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - 실제 구현
