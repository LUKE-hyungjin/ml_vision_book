---
title: "샘플링"
weight: 9
math: true
---

# 샘플링 (Sampling)

{{% hint info %}}
**선수지식**: [확률분포](/ko/docs/math/probability/distribution), [기댓값](/ko/docs/math/probability/expectation)
{{% /hint %}}

## 한 줄 요약

> **"확률분포에서 값을 뽑는 방법"**

---

## 왜 샘플링을 배워야 하나요?

### 문제 상황 1: Diffusion은 어떻게 이미지를 "생성"하나요?

```python
# Diffusion 이미지 생성
noise = torch.randn(1, 3, 512, 512)  # 노이즈 "샘플링"
image = model.denoise(noise)          # 노이즈 → 이미지
```

**정답**: 가우시안 분포에서 **샘플링**해서 시작합니다!
- 샘플링 없이는 생성 모델이 작동하지 않음

### 문제 상황 2: VAE는 왜 "Reparameterization"이 필요한가요?

```python
# VAE Encoder
mu, log_var = encoder(x)

# 이렇게 하면 역전파가 안 됨!
z = torch.normal(mu, std)  # 직접 샘플링

# 이렇게 해야 함
z = mu + std * torch.randn_like(std)  # Reparameterization
```

**정답**: 샘플링은 미분이 안 되기 때문입니다!
- **Reparameterization Trick**으로 우회

### 문제 상황 3: GPT는 어떻게 "창의적인" 텍스트를 만드나요?

```python
# Temperature가 높으면 창의적
probs = softmax(logits / temperature)
next_token = sample(probs)  # 확률적 샘플링
```

**정답**: **Temperature Scaling**과 **Top-p/Top-k 샘플링**!
- 결정적 argmax가 아닌 확률적 샘플링으로 다양성 확보

---

## 1. 기본 샘플링

### 균등 분포 샘플링

> "[0, 1) 사이에서 랜덤하게 값 뽑기"

```python
import torch

# [0, 1) 균등 분포
samples = torch.rand(1000)

# [a, b) 균등 분포로 변환
a, b = 2, 5
samples_scaled = a + (b - a) * torch.rand(1000)
# 2 ~ 5 사이의 균등 분포

print(f"평균: {samples.mean():.3f} (기대값: 0.5)")
print(f"스케일 후 평균: {samples_scaled.mean():.3f} (기대값: 3.5)")
```

**딥러닝 적용**:
- Data Augmentation (랜덤 크롭, 회전 등)
- Dropout 마스크
- 배치 셔플링

### 가우시안 샘플링

> "평균 μ, 분산 σ²인 정규분포에서 값 뽑기"

```python
# 표준정규분포 N(0, 1)
samples = torch.randn(1000)

# N(μ, σ²)로 변환
mu, sigma = 5, 2
samples_scaled = mu + sigma * torch.randn(1000)

print(f"표준정규 - 평균: {samples.mean():.3f}, 표준편차: {samples.std():.3f}")
print(f"변환 후 - 평균: {samples_scaled.mean():.3f}, 표준편차: {samples_scaled.std():.3f}")
```

**딥러닝 적용**:
- VAE 잠재 공간
- Diffusion 노이즈
- 가중치 초기화
- Dropout 노이즈 (가우시안 버전)

---

## 2. Reparameterization Trick (핵심!)

### 문제: 샘플링은 미분이 안 된다

```python
# 직접 샘플링
z = torch.normal(mu, std)

# z에 대한 gradient?
# z는 확률적 연산 → gradient가 정의 안 됨!
```

**왜 문제인가?**:
- VAE에서 z = encoder(x)를 샘플링으로 얻음
- z → decoder(z) → reconstruction
- backprop하려면 d(loss)/d(mu), d(loss)/d(std)가 필요
- 그런데 샘플링 과정이 끊김!

### 해결: Reparameterization Trick

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

**핵심 아이디어**:
- 확률적 부분(ε)을 **입력**으로 분리
- μ, σ에 대한 gradient가 흐를 수 있음!

```
직접 샘플링 (gradient 끊김):
    mu, sigma → [샘플링] → z
                    ↑
               gradient X

Reparameterization (gradient 흐름):
    mu, sigma → [×, +] → z
                  ↑
    epsilon ─────╯    ← 상수처럼 취급
               gradient O
```

![Reparameterization Trick](/images/probability/ko/reparameterization-trick.svg)

### VAE에서의 구현

```python
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(...)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(...)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)  # log(σ²) 사용 (수치 안정성)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """핵심: Reparameterization Trick"""
        std = torch.exp(0.5 * log_var)  # σ = exp(log(σ²)/2)
        eps = torch.randn_like(std)      # ε ~ N(0, 1), gradient와 무관
        return mu + std * eps            # z ~ N(μ, σ²), gradient 흐름!

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)  # 샘플링 with gradient
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
```

---

## 3. 카테고리 샘플링

### Argmax (결정적)

```python
logits = model(x)  # [batch, num_classes]
prediction = logits.argmax(dim=-1)  # 항상 같은 결과

# 예: logits = [2.0, 1.0, 0.5]
# argmax → 항상 0 (첫 번째 클래스)
```

**문제**: 다양성이 없음!

### Multinomial (확률적)

```python
probs = torch.softmax(logits, dim=-1)
sample = torch.multinomial(probs, num_samples=1)

# 예: probs = [0.7, 0.2, 0.1]
# 70% 확률로 0, 20% 확률로 1, 10% 확률로 2
```

**장점**: 다양한 결과 가능
**단점**: 낮은 확률도 선택될 수 있음 (노이즈)

### Gumbel-Softmax (미분 가능 샘플링)

**문제**: multinomial도 미분 불가능
**해결**: Gumbel-Softmax Trick

$$
y_i = \frac{\exp((g_i + \log \pi_i) / \tau)}{\sum_j \exp((g_j + \log \pi_j) / \tau)}
$$

여기서 $g_i \sim \text{Gumbel}(0, 1)$

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    """미분 가능한 카테고리 샘플링"""
    # Gumbel noise
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

    # Gumbel-Softmax
    y_soft = torch.softmax((logits + gumbels) / tau, dim=-1)

    if hard:
        # Forward: one-hot (discrete)
        # Backward: soft (continuous gradient)
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft  # Straight-through

    return y_soft

# 사용
logits = torch.tensor([2.0, 1.0, 0.5])
sample = gumbel_softmax(logits, tau=0.5, hard=True)
# sample ≈ [1, 0, 0] (one-hot이지만 gradient 흐름!)
```

**딥러닝 적용**:
- VQ-VAE의 코드북 선택
- Neural Architecture Search
- 이산 변수가 필요한 모든 곳

---

## 4. Temperature Scaling

### 개념

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

**Temperature T의 효과**:
- $T \rightarrow 0$: argmax처럼 (가장 높은 것만)
- $T = 1$: 원래 분포
- $T \rightarrow \infty$: 균등 분포 (완전 랜덤)

### 시각화

```
logits = [3.0, 1.0, 0.5]

T = 0.1 (낮음)         T = 1.0 (기본)         T = 5.0 (높음)
  │█                     │ █                    │█ █ █
  │█                     │ █ █                  │█ █ █
  │█                     │ █ █ █                │█ █ █
  └┴─┴─                  └─┴─┴─┴─               └─┴─┴─┴─
[0.99, 0.01, 0.00]     [0.66, 0.24, 0.10]     [0.40, 0.33, 0.27]
  확신 (argmax)           보통                   균등에 가까움
```

![Temperature Scaling](/images/probability/ko/sampling-temperature.svg)

### 구현

```python
def sample_with_temperature(logits, temperature=1.0):
    """Temperature 조절 샘플링"""
    if temperature == 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# 텍스트 생성에서
# T < 1: 예측 가능한, 보수적인 텍스트
# T > 1: 창의적인, 다양한 텍스트
```

**딥러닝 적용**:
- GPT 텍스트 생성 (창의성 조절)
- Knowledge Distillation (soft label)
- Calibration

---

## 5. Top-k & Top-p 샘플링

### Top-k Sampling

> "상위 k개 토큰만 고려"

```python
def top_k_sampling(logits, k=50):
    """상위 k개만 샘플링 대상"""
    # 상위 k개 추출
    top_k_logits, top_k_indices = logits.topk(k, dim=-1)

    # 상위 k개에서 샘플링
    probs = torch.softmax(top_k_logits, dim=-1)
    sample_idx = torch.multinomial(probs, num_samples=1)

    # 원래 인덱스로 변환
    return top_k_indices.gather(-1, sample_idx)

# 예: k=3이면 상위 3개 토큰 중에서만 선택
# 너무 이상한 토큰이 선택되는 것 방지
```

**문제**: k가 고정이라 상황에 안 맞을 수 있음

### Top-p (Nucleus) Sampling

> "누적 확률이 p가 될 때까지의 토큰만 고려"

```python
def top_p_sampling(logits, p=0.9):
    """누적 확률 p까지의 토큰만 샘플링"""
    # 확률 내림차순 정렬
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # 누적 확률 계산
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # p를 초과하는 토큰 제거 (첫 번째는 유지)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # 제거할 토큰의 logit을 -inf로
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # 샘플링
    probs = torch.softmax(sorted_logits, dim=-1)
    sample_idx = torch.multinomial(probs, num_samples=1)

    return sorted_indices.gather(-1, sample_idx)

# 예: p=0.9면 상위 90% 확률을 차지하는 토큰들만 고려
# 분포가 뾰족하면 적은 토큰, 평평하면 많은 토큰 자동 선택
```

### 비교

```
확률 분포: [0.5, 0.25, 0.15, 0.05, 0.03, 0.02]

Top-k (k=3):
  고려: [0.5, 0.25, 0.15] (상위 3개)
  제외: [0.05, 0.03, 0.02]

Top-p (p=0.9):
  고려: [0.5, 0.25, 0.15] (누적 0.9)
  제외: [0.05, 0.03, 0.02]

다른 상황:
확률: [0.3, 0.3, 0.2, 0.1, 0.1]

Top-k (k=3):
  고려: [0.3, 0.3, 0.2] (상위 3개)

Top-p (p=0.9):
  고려: [0.3, 0.3, 0.2, 0.1] (누적 0.9까지)
  → 상황에 따라 유동적!
```

![Top-k vs Top-p 샘플링](/images/probability/ko/top-k-top-p.svg)

---

## 6. Diffusion의 샘플링

### DDPM 샘플링

```python
def ddpm_sample(model, shape, timesteps=1000):
    """DDPM 이미지 생성"""
    # 순수 노이즈로 시작
    x = torch.randn(shape)  # x_T ~ N(0, I)

    for t in reversed(range(timesteps)):
        # 노이즈 예측
        predicted_noise = model(x, t)

        # 점진적 디노이징
        alpha = alpha_schedule[t]
        alpha_bar = alpha_bar_schedule[t]

        # x_{t-1} 계산 (샘플링 수식)
        mean = (1 / sqrt(alpha)) * (x - (1-alpha)/sqrt(1-alpha_bar) * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x)  # 샘플링!
            sigma = sqrt(beta[t])
            x = mean + sigma * noise
        else:
            x = mean

    return x
```

### DDIM 샘플링 (결정적)

```python
def ddim_sample(model, shape, timesteps=50):
    """DDIM 빠른 샘플링 (결정적)"""
    x = torch.randn(shape)

    # 더 적은 스텝 (50 vs 1000)
    time_steps = torch.linspace(1000, 0, timesteps+1).long()

    for i in range(len(time_steps) - 1):
        t = time_steps[i]
        t_next = time_steps[i + 1]

        predicted_noise = model(x, t)

        # DDIM은 노이즈를 안 더함 → 결정적
        x = ddim_step(x, predicted_noise, t, t_next)

    return x
```

---

## 코드로 확인하기

```python
import torch
import torch.nn.functional as F

# === 기본 샘플링 ===
print("=== 기본 샘플링 ===")

# 균등 분포
uniform_samples = torch.rand(1000)
print(f"균등 분포 평균: {uniform_samples.mean():.3f} (기대값: 0.5)")

# 가우시안
gaussian_samples = torch.randn(1000)
print(f"가우시안 평균: {gaussian_samples.mean():.3f}, 표준편차: {gaussian_samples.std():.3f}")

# === Reparameterization ===
print("\n=== Reparameterization Trick ===")

mu = torch.tensor([1.0, 2.0], requires_grad=True)
log_var = torch.tensor([0.0, 0.5], requires_grad=True)

# Reparameterization
std = torch.exp(0.5 * log_var)
eps = torch.randn_like(std)
z = mu + std * eps

# Gradient 확인
loss = z.sum()
loss.backward()
print(f"mu gradient: {mu.grad}")  # gradient 있음!
print(f"log_var gradient: {log_var.grad}")  # gradient 있음!

# === Temperature ===
print("\n=== Temperature Scaling ===")

logits = torch.tensor([3.0, 1.0, 0.5])
for T in [0.5, 1.0, 2.0, 5.0]:
    probs = F.softmax(logits / T, dim=-1)
    print(f"T={T}: {[f'{p:.3f}' for p in probs.tolist()]}")

# === Top-k / Top-p ===
print("\n=== Top-k vs Top-p ===")

logits = torch.tensor([3.0, 2.0, 1.5, 1.0, 0.5, 0.1])
probs = F.softmax(logits, dim=-1)
print(f"원래 확률: {[f'{p:.3f}' for p in probs.tolist()]}")

# Top-k (k=3)
top_k_probs, top_k_idx = probs.topk(3)
print(f"Top-3: indices={top_k_idx.tolist()}, probs={[f'{p:.3f}' for p in top_k_probs.tolist()]}")

# Top-p (p=0.9)
sorted_probs, sorted_idx = probs.sort(descending=True)
cumsum = sorted_probs.cumsum(dim=-1)
mask = cumsum <= 0.9
mask[0] = True  # 최소 1개
print(f"Top-p (0.9): {mask.sum().item()}개 토큰 포함")

# === Gumbel-Softmax ===
print("\n=== Gumbel-Softmax ===")

def gumbel_softmax(logits, tau=1.0):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    return F.softmax((logits + gumbels) / tau, dim=-1)

logits = torch.tensor([2.0, 1.0, 0.5])
for _ in range(3):
    sample = gumbel_softmax(logits, tau=0.5)
    print(f"Gumbel-Softmax: {[f'{p:.3f}' for p in sample.tolist()]}")
```

---

## 핵심 정리

| 샘플링 방법 | 용도 | 특징 |
|-------------|------|------|
| **torch.rand** | 균등 분포 | [0, 1) 기본 |
| **torch.randn** | 가우시안 | VAE, Diffusion |
| **Reparameterization** | VAE | gradient 흐름 유지 |
| **Gumbel-Softmax** | 이산 변수 | 미분 가능 |
| **Temperature** | 다양성 조절 | T↓확신, T↑랜덤 |
| **Top-k** | 상위 k개 | 노이즈 제거 |
| **Top-p (Nucleus)** | 누적 확률 p | 적응적 |

---

## 핵심 통찰

```
1. 샘플링 = 생성 모델의 핵심
   - Diffusion: 가우시안 노이즈 → 이미지
   - VAE: 잠재 공간 샘플링 → 이미지
   - GPT: 다음 토큰 샘플링 → 텍스트

2. Reparameterization Trick
   - 샘플링 자체는 미분 불가
   - z = μ + σε로 분리하면 gradient 흐름!

3. Temperature & Top-k/p
   - 다양성 vs 품질 트레이드오프
   - T 낮고 k 작으면 → 안전하고 예측 가능
   - T 높고 p 크면 → 창의적이지만 불안정

4. 상황별 선택
   - VAE/Diffusion → Reparameterization
   - 텍스트 생성 → Temperature + Top-p
   - 이산 변수 → Gumbel-Softmax
```

---

## 관련 콘텐츠

- [확률분포](/ko/docs/math/probability/distribution) - 샘플링할 분포들
- [기댓값](/ko/docs/math/probability/expectation) - 샘플 평균
- [VAE](/ko/docs/architecture/generative/vae) - Reparameterization 활용
- [Diffusion](/ko/docs/math/generative/ddpm) - 반복적 샘플링
- [VLM](/ko/docs/architecture/multimodal/vlm) - 텍스트 생성 샘플링
