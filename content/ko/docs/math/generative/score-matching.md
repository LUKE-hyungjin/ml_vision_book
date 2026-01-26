---
title: "Score Matching"
weight: 2
math: true
---

# Score Matching

{{% hint info %}}
**선수지식**: [Gradient (미분)](/ko/docs/math/calculus) | [확률분포](/ko/docs/math/probability) | [DDPM](/ko/docs/math/generative/ddpm)
{{% /hint %}}

## 한 줄 요약

> **Score = "데이터가 많은 방향을 가리키는 화살표"**

---

## 왜 Score가 필요한가?

### 문제: 확률 분포를 직접 구하기 어렵다

고양이 이미지의 확률 분포 $p(\text{이미지})$를 구하고 싶다고 합시다.

$$
p(\text{이미지}) = \frac{e^{f(\text{이미지})}}{Z}
$$

**문제**: $Z$(정규화 상수)를 계산하려면 **모든 가능한 이미지**에 대해 합을 구해야 합니다.
- 256×256 RGB 이미지 = $256^{256 \times 256 \times 3}$개의 경우의 수
- 불가능!

---

## 해결책: 방향만 알면 된다

> **비유**: 산 정상에 가려면 "정상의 정확한 높이"를 몰라도 됩니다. **"어느 방향이 위인지"**만 알면 올라갈 수 있습니다!

### Score 함수 정의

$$
\text{Score}(x) = \nabla_x \log p(x)
$$

**각 기호의 의미:**
- $\nabla_x$ : $x$에 대한 기울기(gradient) - "어느 방향으로 가야 값이 커지나?"
- $\log p(x)$ : 확률의 로그값
- 전체 의미: **"이 점에서 어느 방향으로 가면 확률이 높아지나?"**

### 마법: Z가 사라진다!

$$
\nabla_x \log p(x) = \nabla_x \log \frac{e^{f(x)}}{Z} = \nabla_x f(x) - \nabla_x \log Z = \nabla_x f(x)
$$

$Z$는 $x$와 무관한 상수이므로 미분하면 0이 됩니다!

---

## Score의 직관적 이해

```
낮은 확률 영역        높은 확률 영역 (데이터 분포)
     ·                    ☆☆☆
      ·  → → →           ☆☆☆☆☆
       ·   → → →        ☆☆☆☆☆☆
        ·    ↗ ↗         ☆☆☆☆☆
         ·  ↗               ☆☆

→ = Score 방향 (데이터가 많은 쪽을 가리킴)
```

**Score를 따라가면** 노이즈에서 시작해서 **데이터가 있는 곳**으로 도달합니다!

---

## Score Matching: Score를 학습하기

### 문제: 진짜 Score를 모른다

$$
L = \mathbb{E}\left[ \| s_\theta(x) - \nabla_x \log p_{\text{data}}(x) \|^2 \right]
$$

진짜 데이터 분포 $p_{\text{data}}(x)$의 score를 모르는데 어떻게 학습하나?

### 해결: Denoising Score Matching

> **아이디어**: 노이즈를 추가한 데이터의 score는 계산할 수 있다!

1. 원본 데이터 $x$에 노이즈 $\epsilon$을 추가: $\tilde{x} = x + \sigma \epsilon$
2. 노이즈가 추가된 데이터의 score는 알 수 있음:

$$
\nabla_{\tilde{x}} \log p(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

**해석**: "노이즈가 추가됐으니, 원래 데이터 방향으로 돌아가라!"

---

## Score와 노이즈 예측의 관계

DDPM에서 배운 노이즈 예측 $\epsilon_\theta$와 Score $s_\theta$는 사실 같은 것입니다!

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sigma_t}
$$

| 관점 | 예측 대상 | 의미 |
|------|----------|------|
| DDPM | 노이즈 $\epsilon$ | "어떤 노이즈가 추가됐나?" |
| Score | Score $\nabla \log p$ | "어느 방향이 데이터인가?" |

**같은 것을 다르게 표현한 것!**

---

## Langevin Dynamics: Score로 샘플링하기

Score를 알면 샘플을 생성할 수 있습니다:

$$
x_{t+1} = x_t + \frac{\epsilon}{2} \cdot \underbrace{s(x_t)}_{\text{Score 방향}} + \sqrt{\epsilon} \cdot \underbrace{z_t}_{\text{랜덤 노이즈}}
$$

**해석**:
1. Score 방향(데이터가 많은 쪽)으로 조금 이동
2. 약간의 랜덤성 추가 (다양성을 위해)
3. 반복하면 데이터 분포에서 샘플 생성!

```
노이즈   →   →   →   →   데이터
  ·    Score 따라가기     ☆
```

---

## 코드로 이해하기

```python
def score_matching_loss(model, x, sigma):
    """
    대학교 1학년도 이해할 수 있는 Score Matching

    목표: 노이즈를 추가한 데이터에서 "원래 방향"을 예측
    """
    # 1. 랜덤 노이즈 생성
    noise = torch.randn_like(x)

    # 2. 노이즈 추가 (더럽히기)
    x_noisy = x + sigma * noise

    # 3. 모델이 Score 예측 (어느 방향이 원본인가?)
    score_pred = model(x_noisy, sigma)

    # 4. 정답 Score: -noise/sigma (원본 방향)
    score_true = -noise / sigma

    # 5. 예측과 정답의 차이 최소화
    loss = ((score_pred - score_true) ** 2).mean()

    return loss
```

---

## 요약: Score Matching의 핵심

| 질문 | 답변 |
|------|------|
| Score가 뭔가요? | 데이터가 많은 방향을 가리키는 화살표 |
| 왜 Score를 쓰나요? | 정규화 상수 $Z$ 없이 학습 가능 |
| 어떻게 학습하나요? | 노이즈 추가 → 원본 방향 예측 |
| DDPM과 뭐가 다른가요? | 같은 것의 다른 표현! |

---

## 관련 콘텐츠

- [DDPM](/ko/docs/math/generative/ddpm) - 노이즈 예측 관점
- [Sampling](/ko/docs/math/generative/sampling) - 다양한 샘플링 방법
- [확률 분포](/ko/docs/math/probability) - 기초 확률론
