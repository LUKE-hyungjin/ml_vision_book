---
title: "상호 정보량"
weight: 13
math: true
---

# 상호 정보량 (Mutual Information)

{{% hint info %}}
**선수지식**: [엔트로피](/ko/docs/math/probability/entropy) | [KL 발산](/ko/docs/math/probability/kl-divergence) | [결합/조건부 분포](/ko/docs/math/probability/joint-conditional)
{{% /hint %}}

## 한 줄 요약

> **"X를 알면 Y에 대한 불확실성이 얼마나 줄어드나?" = 두 변수가 공유하는 정보량**

---

## 왜 상호 정보량을 배워야 하나요?

### 문제 상황 1: Contrastive Learning의 목표는 뭔가요?

```python
# SimCLR: 같은 이미지의 두 augmentation
z_i = encoder(augment_1(image))
z_j = encoder(augment_2(image))

# InfoNCE Loss = 상호 정보량의 하한!
loss = info_nce_loss(z_i, z_j)
```

**정답**: 같은 이미지의 두 뷰 사이 **상호 정보량을 최대화**합니다!
- $z_i$를 알면 $z_j$에 대해 많은 것을 알 수 있도록

### 문제 상황 2: 특성 선택(Feature Selection)을 어떻게 하나요?

```python
# 어떤 특성이 레이블과 가장 관련있나?
features = extract_features(images)  # 512차원
labels = [...]

# 512개 중 가장 유용한 특성은?
```

**정답**: 특성과 레이블의 **상호 정보량**이 높은 것을 선택!
- $I(\text{feature}; \text{label})$이 크면 → 유용한 특성

### 문제 상황 3: 상관계수와 뭐가 다른가요?

```python
# 상관계수: 선형 관계만 측정
corr = np.corrcoef(X, Y)[0, 1]  # X = sin(t), Y = cos(t) → ≈ 0

# 하지만 X와 Y는 강하게 관련되어 있음!
```

**정답**: 상호 정보량은 **비선형 관계까지** 포착합니다!
- 상관계수 = 0이어도 상호 정보량 > 0일 수 있음

---

## 1. 정의

### 수학적 정의

$$
I(X; Y) = \sum_{x,y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)}
$$

연속일 때:

$$
I(X; Y) = \int \int f(x, y) \log \frac{f(x, y)}{f(x) f(y)} \, dx \, dy
$$

### KL 발산으로 표현

$$
I(X; Y) = D_{KL}(P(X, Y) \| P(X) P(Y))
$$

**해석**: "결합 분포가 독립 가정과 얼마나 다른가?"
- $I = 0$ → $P(X,Y) = P(X)P(Y)$ → **독립**
- $I > 0$ → 독립이 아님 → 정보를 공유함

### 엔트로피로 표현

$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$

**해석**:
- $H(X)$: X의 총 불확실성
- $H(X|Y)$: Y를 알 때 X의 남은 불확실성
- $I(X;Y)$: Y가 X에 대해 **줄여준** 불확실성

![엔트로피와 상호 정보량 벤 다이어그램](/images/probability/ko/mutual-information-venn.png)

---

## 2. 핵심 성질

### 기본 성질

| 성질 | 수식 | 의미 |
|------|------|------|
| **비음수** | $I(X;Y) \geq 0$ | 항상 0 이상 |
| **대칭** | $I(X;Y) = I(Y;X)$ | X→Y든 Y→X든 같은 정보량 |
| **독립이면 0** | $I(X;Y) = 0 \iff$ X,Y 독립 | 독립 = 정보 공유 없음 |
| **상한** | $I(X;Y) \leq \min(H(X), H(Y))$ | 더 적은 엔트로피를 초과 불가 |

### 상관계수와의 비교

| | 상관계수 $\rho$ | 상호 정보량 $I(X;Y)$ |
|---|---|---|
| **측정 대상** | 선형 관계 | 모든 종류의 의존성 |
| **범위** | $[-1, 1]$ | $[0, \infty)$ |
| **$\rho = 0$이면** | 선형 관계 없음 | $I > 0$일 수 있음 |
| **$I = 0$이면** | $\rho = 0$ | 완전 독립 |

### 예시: 비선형 관계

![상관계수 vs 상호 정보량](/images/probability/ko/correlation-vs-mi.png)

상관계수는 비선형 관계를 놓치지만, 상호 정보량은 잡아냅니다!

---

## 3. 가우시안의 상호 정보량

### 2차원 가우시안

$X, Y$가 상관계수 $\rho$인 이변량 가우시안이면:

$$
I(X; Y) = -\frac{1}{2} \log(1 - \rho^2)
$$

```
ρ = 0    → I = 0      (독립)
ρ = 0.5  → I = 0.14
ρ = 0.9  → I = 0.83
ρ = 1.0  → I = ∞      (완전 종속)
```

### 다변량 가우시안

$$
I(\mathbf{X}_1; \mathbf{X}_2) = \frac{1}{2} \log \frac{|\Sigma_{11}| \cdot |\Sigma_{22}|}{|\Sigma|}
$$

---

## 4. 딥러닝에서의 활용

### 1) InfoNCE Loss (Contrastive Learning)

**핵심 아이디어**: InfoNCE Loss는 상호 정보량의 **하한**!

$$
I(X; Y) \geq \log N - \mathcal{L}_{\text{InfoNCE}}
$$

```python
def info_nce_loss(z_i, z_j, temperature=0.5):
    """
    InfoNCE Loss (SimCLR)

    z_i, z_j: 같은 이미지의 두 augmentation 표현
    목표: I(z_i; z_j) 최대화 = InfoNCE Loss 최소화
    """
    batch_size = z_i.shape[0]

    # 정규화
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # 유사도 행렬
    sim = torch.mm(z_i, z_j.T) / temperature  # (N, N)

    # 대각선 = positive pairs (같은 이미지)
    # 비대각선 = negative pairs (다른 이미지)
    labels = torch.arange(batch_size, device=z_i.device)

    # Cross-Entropy: 대각선이 가장 높도록
    loss = F.cross_entropy(sim, labels)
    return loss
```

### 2) MINE (Mutual Information Neural Estimation)

신경망으로 상호 정보량을 **직접 추정**:

$$
I(X; Y) \geq \mathbb{E}_{P(X,Y)}[T_\theta(x,y)] - \log \mathbb{E}_{P(X)P(Y)}[e^{T_\theta(x,y)}]
$$

```python
class MINE(nn.Module):
    """Mutual Information Neural Estimator"""
    def __init__(self, x_dim, y_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # Joint: (x, y) 쌍
        joint = self.net(torch.cat([x, y], dim=1))

        # Marginal: x와 셔플된 y 쌍
        y_shuffled = y[torch.randperm(y.shape[0])]
        marginal = self.net(torch.cat([x, y_shuffled], dim=1))

        # MINE 하한
        mi = joint.mean() - torch.log(torch.exp(marginal).mean())
        return mi
```

### 3) Information Bottleneck

**목표**: 입력 X에서 라벨 Y에 관한 정보만 남기기

$$
\min_{Z} I(X; Z) - \beta \cdot I(Z; Y)
$$

```
X (입력) ──→ Z (표현) ──→ Y (출력)

I(X; Z): 최소화 → 불필요한 정보 버리기 (압축)
I(Z; Y): 최대화 → 유용한 정보 보존 (예측)

β: 압축과 예측의 균형
```

```python
# 실전: VIB (Variational Information Bottleneck)
class VIB(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(...)    # X → Z
        self.classifier = nn.Sequential(...) # Z → Y

    def forward(self, x, y, beta=0.01):
        mu, log_var = self.encoder(x)

        # 재매개변수화
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

        # I(X; Z) ≈ KL(q(z|x) || p(z)) → 최소화
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # I(Z; Y) ≈ -CE(classifier(z), y) → 최대화
        pred = self.classifier(z)
        ce = F.cross_entropy(pred, y)

        return ce + beta * kl
```

### 4) Barlow Twins

상관 행렬로 **중복 정보**를 줄이기:

```python
# Barlow Twins: 특성 간 상호 정보를 줄임
# 대각선 = 1 (같은 특성은 완전 상관)
# 비대각선 = 0 (다른 특성은 독립 = 상호 정보량 ≈ 0)

# 이는 표현의 "중복 축소" (Redundancy Reduction)
# → 각 차원이 독립적인 정보를 담도록 유도
```

---

## 코드로 확인하기

```python
import numpy as np

# === 이산 분포에서 상호 정보량 ===
print("=== 이산 상호 정보량 ===")

def mutual_information(joint_prob):
    """결합 분포에서 상호 정보량 계산"""
    # 주변 분포
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)

    mi = 0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(
                    joint_prob[i, j] / (p_x[i] * p_y[j])
                )
    return mi

# 예시 1: 독립인 경우
joint_independent = np.array([
    [0.15, 0.10, 0.05],
    [0.30, 0.20, 0.10],
    [0.06, 0.04, 0.00],
])
# 정확한 독립 분포로 수정
p_x = np.array([0.3, 0.5, 0.2])
p_y = np.array([0.4, 0.4, 0.2])
joint_ind = np.outer(p_x, p_y)
print(f"독립: I(X;Y) = {mutual_information(joint_ind):.6f}")  # ≈ 0

# 예시 2: 종속인 경우 (날씨-아이스크림)
joint_dep = np.array([
    [0.30, 0.10, 0.02],
    [0.10, 0.15, 0.08],
    [0.05, 0.10, 0.10],
])
print(f"종속: I(X;Y) = {mutual_information(joint_dep):.4f}")

# 예시 3: 완전 종속 (Y = X)
joint_perfect = np.array([
    [0.33, 0.00, 0.00],
    [0.00, 0.33, 0.00],
    [0.00, 0.00, 0.34],
])
print(f"완전종속: I(X;Y) = {mutual_information(joint_perfect):.4f}")

# === 가우시안 상호 정보량 ===
print("\n=== 가우시안 상호 정보량 ===")

for rho in [0, 0.3, 0.5, 0.7, 0.9, 0.99]:
    mi = -0.5 * np.log(1 - rho**2)
    print(f"ρ = {rho:.2f} → I(X;Y) = {mi:.4f}")

# === 상관계수 vs 상호 정보량 ===
print("\n=== 상관계수 vs 상호 정보량 ===")

# 비선형 관계: Y = X²
N = 10000
X = np.random.uniform(-2, 2, N)
Y = X ** 2 + 0.1 * np.random.randn(N)

corr = np.corrcoef(X, Y)[0, 1]
print(f"Y = X² + noise")
print(f"  상관계수: {corr:.4f}")
print(f"  → 상관계수는 비선형 관계를 놓침!")

# 엔트로피 기반 MI 추정 (binning)
def mi_binned(x, y, bins=30):
    """히스토그램 기반 MI 추정"""
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()
    joint_prob = joint_prob[joint_prob > 0]

    marginal_x = np.histogram(x, bins=bins)[0].astype(float)
    marginal_x /= marginal_x.sum()
    marginal_y = np.histogram(y, bins=bins)[0].astype(float)
    marginal_y /= marginal_y.sum()

    h_x = -np.sum(marginal_x[marginal_x > 0] * np.log(marginal_x[marginal_x > 0]))
    h_y = -np.sum(marginal_y[marginal_y > 0] * np.log(marginal_y[marginal_y > 0]))
    h_xy = -np.sum(joint_prob * np.log(joint_prob))

    return h_x + h_y - h_xy

mi_est = mi_binned(X, Y)
print(f"  상호 정보량 (binning): {mi_est:.4f}")
print(f"  → 상호 정보량은 비선형 관계를 잡아냄!")

# === InfoNCE 시뮬레이션 ===
print("\n=== InfoNCE Loss ===")
import torch
import torch.nn.functional as F

batch_size = 256
dim = 128

# Positive pair: 유사한 벡터
z_i = F.normalize(torch.randn(batch_size, dim), dim=1)
noise = 0.1 * torch.randn(batch_size, dim)
z_j = F.normalize(z_i + noise, dim=1)  # 약간의 노이즈 추가

temperature = 0.5
sim = torch.mm(z_i, z_j.T) / temperature
labels = torch.arange(batch_size)

loss = F.cross_entropy(sim, labels)
mi_lower_bound = np.log(batch_size) - loss.item()

print(f"InfoNCE Loss: {loss.item():.4f}")
print(f"I(X;Y) >= log(N) - Loss = {np.log(batch_size):.2f} - {loss.item():.2f} = {mi_lower_bound:.2f}")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 활용 |
|------|------|-------------|
| **상호 정보량** | $I(X;Y) = H(X) - H(X\|Y)$ | Contrastive Learning 목표 |
| **KL 표현** | $D_{KL}(P(X,Y) \| P(X)P(Y))$ | 독립과의 거리 |
| **InfoNCE** | $I(X;Y) \geq \log N - \mathcal{L}$ | SimCLR, MoCo |
| **Information Bottleneck** | $\min I(X;Z) - \beta I(Z;Y)$ | VIB, 표현 학습 |

---

## 핵심 통찰

```
1. 상호 정보량 = 공유 정보의 크기
   - I(X;Y) = 0 → 완전 독립
   - I(X;Y) = H(X) → Y가 X를 완전 결정

2. 상관계수보다 강력
   - 비선형 관계까지 포착
   - 0이면 진짜 독립 (상관계수 0 ≠ 독립)

3. 대칭이다!
   - I(X;Y) = I(Y;X)
   - KL 발산과 달리 순서 무관

4. Contrastive Learning의 핵심
   - InfoNCE ≈ 상호 정보량 최대화
   - 같은 것끼리 정보 공유 높이기
   - 다른 것끼리 정보 공유 낮추기
```

---

## 다음 단계

**마르코프 체인**과 순차적 의존 관계가 궁금하다면?
→ [마르코프 체인](/ko/docs/math/probability/markov-chains)으로!

상호 정보량의 실전 활용인 **Contrastive Loss**를 배우고 싶다면?
→ [Contrastive Loss](/ko/docs/components/training/loss/contrastive-loss)로!

---

## 관련 콘텐츠

- [엔트로피](/ko/docs/math/probability/entropy) - 상호 정보량의 기반
- [KL 발산](/ko/docs/math/probability/kl-divergence) - 상호 정보량 = 특별한 KL
- [결합/조건부 분포](/ko/docs/math/probability/joint-conditional) - 결합 분포 기초
- [Contrastive Loss](/ko/docs/components/training/loss/contrastive-loss) - InfoNCE Loss
- [CLIP](/ko/docs/architecture/multimodal/clip) - 멀티모달 상호 정보량
