---
title: "KL 발산"
weight: 7
math: true
---

# KL 발산 (Kullback-Leibler Divergence)

{{% hint info %}}
**선수지식**: [엔트로피](/ko/docs/math/probability/entropy), [확률분포](/ko/docs/math/probability/distribution)
{{% /hint %}}

## 한 줄 요약

> **"두 확률분포가 얼마나 다른가?"를 측정하는 방법**

---

## 왜 KL 발산을 배워야 하나요?

### 문제 상황 1: VAE Loss의 KL 항은 뭔가요?

```python
# VAE Loss
loss = reconstruction_loss + kl_weight * kl_divergence
#                            ↑ 이게 뭐야?
```

**정답**: 인코더의 출력이 **표준정규분포에 얼마나 가까운지** 측정합니다!
- KL이 크면 → 잠재 공간이 이상해짐
- KL이 작으면 → 잠재 공간이 N(0,1)에 가까움

### 문제 상황 2: Knowledge Distillation은 어떻게 작동하나요?

```python
# Teacher → Student 지식 전달
loss = kl_divergence(teacher_output, student_output)
```

**정답**: Student의 출력을 **Teacher의 출력 분포**에 가깝게 만듭니다!

### 문제 상황 3: Cross-Entropy와 KL 발산의 관계는?

```python
# 이 둘이 같은 걸 최소화한다고?
loss_ce = CrossEntropyLoss(pred, target)
loss_kl = KL_Divergence(target, pred)  # ???
```

**정답**: Cross-Entropy = Entropy + KL Divergence입니다!
- 타겟의 Entropy는 상수 → CE 최소화 = KL 최소화

---

## KL 발산이란?

### 정의

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

또는 기댓값 형태로:

$$
D_{KL}(P \| Q) = \mathbb{E}_P \left[ \log \frac{P(X)}{Q(X)} \right]
$$

**해석**:
- $P$: "진짜" 분포 (타겟, 정답)
- $Q$: "근사" 분포 (모델, 예측)
- $D_{KL}(P \| Q)$: "Q가 P를 얼마나 잘 설명하지 못하는가"

### 직관적 이해

```
KL 발산 = "Q가 P를 흉내낼 때의 비효율성"

예시: P = 실제 언어 분포, Q = 모델이 학습한 분포

P에서 자주 나오는 단어를 Q가 낮게 예측하면?
→ KL 발산 ↑ (큰 페널티)

P에서 드물게 나오는 단어를 Q가 잘못 예측해도?
→ KL 발산에 작은 영향 (드문 단어니까)
```

![KL Divergence 개념](/images/probability/ko/kl-divergence.svg)

### Cross-Entropy와의 관계

$$
H(P, Q) = H(P) + D_{KL}(P \| Q)
$$

```
Cross-Entropy = 타겟의 Entropy + KL 발산
     ↓                ↓              ↓
  최소화 대상       상수 (고정)     진짜 최소화 대상

따라서:
Cross-Entropy 최소화 = KL 발산 최소화 = 예측을 타겟에 가깝게!
```

![Cross-Entropy와 KL 발산 관계](/images/probability/ko/ce-kl-relationship.svg)

---

## KL 발산의 핵심 성질

### 1) 항상 0 이상

$$
D_{KL}(P \| Q) \geq 0
$$

등호 성립: $P = Q$일 때만

```python
# 같은 분포면 KL = 0
P = [0.3, 0.3, 0.4]
Q = [0.3, 0.3, 0.4]
# D_KL(P || Q) = 0
```

### 2) 비대칭! (매우 중요)

$$
D_{KL}(P \| Q) \neq D_{KL}(Q \| P)
$$

**KL 발산은 "거리"가 아닙니다!**

```python
P = [0.5, 0.5]      # 균등
Q = [0.9, 0.1]      # 편향

D_KL(P || Q) = 0.5 × log(0.5/0.9) + 0.5 × log(0.5/0.1)
             = 0.5 × (-0.85) + 0.5 × (2.32)
             ≈ 0.74

D_KL(Q || P) = 0.9 × log(0.9/0.5) + 0.1 × log(0.1/0.5)
             = 0.9 × (0.85) + 0.1 × (-2.32)
             ≈ 0.53

# 서로 다름!
```

![KL 발산의 비대칭성](/images/probability/ko/kl-asymmetry.svg)

---

## Forward KL vs Reverse KL (핵심!)

### 차이점

| | Forward KL: $D_{KL}(P \| Q)$ | Reverse KL: $D_{KL}(Q \| P)$ |
|---|---|---|
| **누가 타겟?** | P (진짜 분포) | P (진짜 분포) |
| **누구를 최적화?** | Q (모델) | Q (모델) |
| **P가 높고 Q가 낮으면** | 큰 페널티 | 작은 페널티 |
| **P가 낮고 Q가 높으면** | 작은 페널티 | 큰 페널티 |
| **결과** | Mode-covering (다 커버) | Mode-seeking (하나 집중) |

### 시각적 비교

```
P (실제: 두 봉우리)       Forward KL 결과        Reverse KL 결과

   ╭─╮   ╭─╮               ╭─────╮               ╭─╮
   │ │   │ │               │     │               │ │
───┴─┴───┴─┴───         ───┴─────┴───         ───┴─┴───────
    ↑       ↑              흐릿하지만             하나만
  mode1  mode2            둘 다 커버             선명하게

Forward: P가 있는 곳은 Q도 있어야 함 → 모든 모드 커버
Reverse: Q가 있는 곳은 P도 있어야 함 → 하나만 확실히
```

![Forward KL vs Reverse KL](/images/probability/ko/forward-reverse-kl.svg)

### 딥러닝에서의 선택

| 상황 | 사용하는 KL | 이유 |
|------|-------------|------|
| **VAE** | $D_{KL}(q(z\|x) \| p(z))$ | q가 p를 따라야 함 (Reverse) |
| **분류 학습** | $D_{KL}(P_{target} \| Q_{model})$ | 정답 분포 커버 (Forward) |
| **Knowledge Distillation** | $D_{KL}(P_{teacher} \| Q_{student})$ | Teacher 따라하기 (Forward) |
| **GAN (f-GAN)** | Reverse 또는 Forward | 목적에 따라 다름 |

---

## 딥러닝 핵심 적용

### 1) VAE의 KL Loss

$$
\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{KL Regularization}}
$$

**KL 항의 역할**:
- $q(z|x)$: 인코더 출력 (데이터마다 다른 분포)
- $p(z) = \mathcal{N}(0, I)$: 사전 분포 (표준정규)
- KL 최소화 → 잠재 공간이 정규분포처럼 됨

```python
def vae_loss(x, x_recon, mu, log_var):
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL Divergence: N(mu, sigma^2) vs N(0, 1)
    # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss

# KL 해석:
# mu가 0에서 멀면 → KL 증가 (페널티)
# sigma가 1에서 벗어나면 → KL 증가 (페널티)
```

**KL 발산 공식 유도** ($q = \mathcal{N}(\mu, \sigma^2)$, $p = \mathcal{N}(0, 1)$):

$$
D_{KL} = -\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)
$$

### 2) Knowledge Distillation

```python
def distillation_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Teacher의 "soft knowledge"를 Student에게 전달

    Temperature가 높으면:
    - 확률이 더 균등해짐 (soft)
    - 틀린 클래스 정보도 전달됨 ("3은 아닌데 8과 비슷해" 같은)
    """
    # Temperature scaling
    soft_student = F.softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

    # KL Divergence (Teacher → Student)
    # Student가 Teacher를 따라하도록
    loss = F.kl_div(
        soft_student.log(),
        soft_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)

    return loss

# 왜 T^2를 곱하는가?
# Temperature로 나누면 gradient가 T^2만큼 작아지므로 보정
```

### 3) PPO (강화학습)

새 정책이 이전 정책에서 너무 벗어나지 않도록:

```python
def ppo_loss(old_policy, new_policy, advantages, clip_epsilon=0.2):
    """
    KL 발산을 직접 쓰지 않지만, 비슷한 효과
    정책이 급격히 바뀌는 것을 방지
    """
    ratio = new_policy / old_policy

    # Clipping으로 급격한 변화 방지
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    return -torch.min(ratio * advantages, clipped * advantages).mean()
```

### 4) 정규화 (Regularization)

Label Smoothing도 KL 관점에서 이해 가능:

```python
# Hard label: [0, 0, 1, 0] → 균등분포와 KL이 매우 큼
# Smooth label: [0.025, 0.025, 0.925, 0.025] → 균등분포와 KL이 작아짐

# 효과: 모델이 너무 확신하지 않도록 (과적합 방지)
```

---

## 다른 분포 거리와 비교

### 왜 KL만 쓰지 않나요?

```
KL의 문제점:

1. 비대칭: D_KL(P||Q) ≠ D_KL(Q||P)

2. 무한대 가능: Q(x)=0이고 P(x)>0이면 → ∞
   (분포가 겹치지 않으면 정의 안 됨)

3. 삼각 부등식 불만족: 거리 공리 X
```

### 대안들

| 거리/발산 | 수식 | 특징 | 사용처 |
|-----------|------|------|--------|
| **KL Divergence** | $\sum p \log(p/q)$ | 비대칭, 정보이론 | VAE, Classification |
| **Jensen-Shannon** | $\frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$ | 대칭, 유한 | GAN (원래) |
| **Wasserstein** | $\inf_\gamma \mathbb{E}[\|x-y\|]$ | 대칭, 기하학적 | WGAN |
| **Total Variation** | $\frac{1}{2}\sum|p-q|$ | 대칭, 단순 | 이론 분석 |

### Jensen-Shannon Divergence

KL의 대칭 버전:

$$
D_{JS}(P \| Q) = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)
$$

여기서 $M = \frac{1}{2}(P + Q)$

```python
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
```

---

## 코드로 확인하기

```python
import numpy as np
import torch
import torch.nn.functional as F

# === 이산 분포 KL 발산 ===
def kl_divergence(p, q, eps=1e-10):
    """D_KL(P || Q)"""
    p = np.array(p) + eps
    q = np.array(q) + eps
    return np.sum(p * np.log(p / q))

print("=== KL 발산 기본 ===")
P = [0.5, 0.5]      # 균등
Q = [0.9, 0.1]      # 편향

print(f"D_KL(P || Q) = {kl_divergence(P, Q):.4f}")
print(f"D_KL(Q || P) = {kl_divergence(Q, P):.4f}")
print("→ 비대칭! 서로 다른 값")

# 같은 분포
print(f"\nD_KL(P || P) = {kl_divergence(P, P):.4f}")
print("→ 같으면 0")

# === VAE KL Loss ===
print("\n=== VAE KL Loss ===")

def vae_kl_loss(mu, log_var):
    """D_KL(N(mu, sigma^2) || N(0, 1))"""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# mu=0, sigma=1이면 KL=0
mu_0 = torch.zeros(1, 64)
logvar_0 = torch.zeros(1, 64)  # log(1) = 0
print(f"mu=0, sigma=1: KL = {vae_kl_loss(mu_0, logvar_0).item():.4f}")

# mu가 0에서 벗어나면
mu_shift = torch.ones(1, 64) * 2
logvar_0 = torch.zeros(1, 64)
print(f"mu=2, sigma=1: KL = {vae_kl_loss(mu_shift, logvar_0).item():.4f}")

# sigma가 1에서 벗어나면
mu_0 = torch.zeros(1, 64)
logvar_big = torch.ones(1, 64) * 2  # log(e^2) ≈ 2
print(f"mu=0, sigma=e: KL = {vae_kl_loss(mu_0, logvar_big).item():.4f}")

# === PyTorch KL Divergence ===
print("\n=== PyTorch KL Divergence ===")

# F.kl_div는 log_prob를 첫 번째 인자로 받음
p = torch.tensor([0.4, 0.3, 0.3])
q = torch.tensor([0.5, 0.25, 0.25])

# D_KL(P || Q) using PyTorch
# 주의: F.kl_div(input, target)에서 input은 log_prob!
kl_pt = F.kl_div(q.log(), p, reduction='sum')
print(f"D_KL(P || Q) = {kl_pt.item():.4f}")

# 직접 계산과 비교
kl_manual = kl_divergence(p.numpy(), q.numpy())
print(f"수동 계산: {kl_manual:.4f}")

# === Knowledge Distillation ===
print("\n=== Knowledge Distillation ===")

teacher_logits = torch.tensor([[3.0, 1.0, 0.5]])
student_logits = torch.tensor([[2.0, 1.5, 0.5]])

T = 4.0  # Temperature

soft_teacher = F.softmax(teacher_logits / T, dim=-1)
soft_student = F.softmax(student_logits / T, dim=-1)

print(f"Teacher (soft): {soft_teacher.tolist()[0]}")
print(f"Student (soft): {soft_student.tolist()[0]}")

kd_loss = F.kl_div(soft_student.log(), soft_teacher, reduction='batchmean') * (T ** 2)
print(f"Distillation Loss: {kd_loss.item():.4f}")
```

---

## 핵심 정리

| 개념 | 수식 | 핵심 |
|------|------|------|
| **KL 발산** | $D_{KL}(P\|Q) = \sum p \log(p/q)$ | P를 Q로 표현할 때의 추가 비트 |
| **비대칭성** | $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ | Forward vs Reverse |
| **비음수성** | $D_{KL} \geq 0$ | P=Q일 때만 0 |
| **CE 관계** | $H(P,Q) = H(P) + D_{KL}(P\|Q)$ | CE 최소화 = KL 최소화 |

---

## 핵심 통찰

```
1. KL 발산 = 두 분포의 "차이"
   - 0이면 같은 분포
   - 클수록 많이 다름

2. 비대칭이 중요!
   - Forward: P가 있으면 Q도 있어야 (mode-covering)
   - Reverse: Q가 있으면 P도 있어야 (mode-seeking)

3. 딥러닝 핵심 적용
   - VAE: 잠재 공간 정규화
   - Distillation: Teacher 따라하기
   - PPO: 정책 안정화

4. Cross-Entropy와의 관계
   - CE 최소화 = KL 최소화 (타겟 엔트로피가 상수이므로)
```

---

## 다음 단계

**파라미터를 데이터에 맞추는** 방법이 궁금하다면?
→ [최대 우도 추정](/ko/docs/math/probability/mle)으로!

분포에서 **값을 뽑는** 다양한 방법이 궁금하다면?
→ [샘플링](/ko/docs/math/probability/sampling)으로!

---

## 관련 콘텐츠

- [엔트로피](/ko/docs/math/probability/entropy) - KL 발산의 기반
- [확률분포](/ko/docs/math/probability/distribution) - 분포 기초
- [베이즈 정리](/ko/docs/math/probability/bayes) - 사전/사후 분포
- [VAE](/ko/docs/architecture/generative/vae) - KL Loss 실전 적용
- [Cross-Entropy Loss](/ko/docs/math/training/loss/cross-entropy) - 분류 Loss
