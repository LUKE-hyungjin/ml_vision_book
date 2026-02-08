---
title: "DDPM"
weight: 1
math: true
---

# DDPM (Denoising Diffusion Probabilistic Models)

{{% hint info %}}
**선수지식**: [DDPM 수학](/ko/docs/components/generative/ddpm) | [U-Net](/ko/docs/architecture/cnn)
{{% /hint %}}

## 한 줄 요약

> **"U-Net으로 노이즈를 예측하여 고품질 이미지 생성"**

---

## 왜 DDPM인가?

2020년 Ho et al.이 발표한 논문으로, Diffusion 모델의 실용성을 증명했습니다.

```
이전 (2015-2019):
- Diffusion 개념은 있었지만 품질 낮음
- GAN이 주류

DDPM (2020):
- 단순한 구조 + 단순한 Loss
- GAN에 필적하는 품질
- 이후 Stable Diffusion의 기반이 됨
```

---

{{< figure src="/images/math/generative/ddpm/ko/ddpm-training.svg" caption="DDPM 학습: 노이즈 예측하기" >}}

## 핵심 기여

| 기여 | 설명 |
|------|------|
| **Simplified Loss** | VLB 대신 단순 MSE로 학습 |
| **U-Net 아키텍처** | PixelCNN++ 스타일 U-Net 사용 |
| **노이즈 예측** | $x_0$ 대신 $\epsilon$ 예측 |
| **실험적 검증** | CIFAR-10, LSUN에서 SOTA 달성 |

---

## 아키텍처: U-Net

### 전체 구조

```
입력: x_t (노이즈 이미지), t (타임스텝)
        ↓
┌────────────────────────────────────────────┐
│                   U-Net                     │
│                                             │
│   [Time Embedding]                          │
│        ↓                                    │
│   ┌─────────┐                               │
│   │Encoder  │ 32×32 → 16×16 → 8×8          │
│   │(Down)   │ 채널: 128 → 256 → 512         │
│   └────┬────┘                               │
│        │ skip connections                    │
│   ┌────┴────┐                               │
│   │Decoder  │ 8×8 → 16×16 → 32×32          │
│   │(Up)     │ 채널: 512 → 256 → 128         │
│   └─────────┘                               │
│        ↓                                    │
│   Output: ε_θ (예측된 노이즈)               │
└────────────────────────────────────────────┘
```

### 핵심 컴포넌트

| 컴포넌트 | 설명 |
|----------|------|
| **ResBlock** | Residual block + Group Normalization |
| **Time Embedding** | Sinusoidal → MLP → 각 블록에 주입 |
| **Attention** | 16×16 해상도에서 Self-Attention |
| **Skip Connection** | Encoder → Decoder 연결 |

---

## Time Embedding

### Sinusoidal Positional Encoding

Transformer에서 사용하는 것과 동일한 방식:

$$
\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right)
$$

$$
\text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$

### 구현

```python
import torch
import math

def sinusoidal_embedding(timesteps, dim):
    """
    timesteps: [B] 텐서 (0~T-1 사이 정수)
    dim: 임베딩 차원

    반환: [B, dim] 텐서
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

# 사용
t = torch.randint(0, 1000, (batch_size,))
time_emb = sinusoidal_embedding(t, dim=256)  # [B, 256]
time_emb = mlp(time_emb)  # [B, 512]
```

### 블록에 주입

```python
class ResBlock(nn.Module):
    def forward(self, x, t_emb):
        # t_emb를 scale, shift로 변환
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)

        # Group Norm 후 적용
        h = self.norm(x)
        h = h * (1 + scale[:, :, None, None])
        h = h + shift[:, :, None, None]

        return self.conv(h) + x
```

---

## 학습 알고리즘 (Algorithm 1)

```
Algorithm 1: Training
────────────────────────────────────────
repeat
    x_0 ~ q(x_0)                  # 데이터에서 샘플
    t ~ Uniform({1, ..., T})      # 랜덤 타임스텝
    ε ~ N(0, I)                   # 랜덤 노이즈

    # Gradient 계산
    ∇_θ || ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t) ||²

until converged
────────────────────────────────────────
```

### 코드 구현

```python
def train_step(model, x_0, optimizer):
    batch_size = x_0.shape[0]

    # 1. 랜덤 타임스텝 (1 ~ T)
    t = torch.randint(1, T + 1, (batch_size,), device=x_0.device)

    # 2. 노이즈 생성
    epsilon = torch.randn_like(x_0)

    # 3. Forward process: x_t 계산
    sqrt_alpha_bar = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * epsilon

    # 4. 노이즈 예측
    epsilon_pred = model(x_t, t)

    # 5. Simple Loss
    loss = F.mse_loss(epsilon_pred, epsilon)

    # 6. 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## 샘플링 알고리즘 (Algorithm 2)

```
Algorithm 2: Sampling
────────────────────────────────────────
x_T ~ N(0, I)                     # 순수 노이즈에서 시작

for t = T, T-1, ..., 1 do
    z ~ N(0, I) if t > 1, else z = 0

    x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε_θ(x_t, t)) + σ_t z

end for

return x_0
────────────────────────────────────────
```

### 코드 구현

```python
@torch.no_grad()
def sample(model, shape, device):
    """
    shape: (batch_size, channels, height, width)
    """
    # 순수 노이즈에서 시작
    x = torch.randn(shape, device=device)

    # T → 1 역순으로
    for t in reversed(range(1, T + 1)):
        t_batch = torch.full((shape[0],), t, device=device)

        # 노이즈 예측
        epsilon_pred = model(x, t_batch)

        # 파라미터
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]

        # 평균 계산
        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * epsilon_pred
        )

        # 분산 (t > 1일 때만 노이즈 추가)
        if t > 1:
            sigma = torch.sqrt(beta)
            z = torch.randn_like(x)
            x = mean + sigma * z
        else:
            x = mean

    return x
```

---

## 하이퍼파라미터

### 논문 설정

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **T** | 1000 | 총 타임스텝 |
| **β_1** | 0.0001 | 시작 노이즈 |
| **β_T** | 0.02 | 끝 노이즈 |
| **Schedule** | Linear | $\beta_t = \beta_1 + (t-1)/(T-1) \cdot (\beta_T - \beta_1)$ |

### 모델 설정 (CIFAR-10)

| 파라미터 | 값 |
|----------|-----|
| **채널** | 128 |
| **채널 배수** | [1, 2, 2, 2] |
| **Attention 해상도** | 16 |
| **ResBlock 개수** | 2 per resolution |
| **Dropout** | 0.1 |

### 학습 설정

| 파라미터 | 값 |
|----------|-----|
| **Optimizer** | Adam |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 128 |
| **EMA decay** | 0.9999 |
| **학습 스텝** | 800K |

---

## ResBlock 구조

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()

        # 첫 번째 컨볼루션
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),  # scale + shift
        )

        # 두 번째 컨볼루션
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        # Skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)

        # Time conditioning
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(h)

        return h + self.skip(x)
```

---

## Self-Attention

16×16 해상도에서 Self-Attention 적용:

```python
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # Attention
        attn = torch.softmax(q.transpose(-1, -2) @ k / (C ** 0.5), dim=-1)
        h = (v @ attn.transpose(-1, -2)).view(B, C, H, W)

        return x + self.proj(h)
```

---

## 실험 결과

### CIFAR-10 (32×32)

| 모델 | FID ↓ | IS ↑ |
|------|-------|------|
| BigGAN | 14.73 | - |
| StyleGAN2 + ADA | 2.92 | - |
| **DDPM** | **3.17** | **9.46** |

### LSUN Bedroom (256×256)

| 모델 | FID ↓ |
|------|-------|
| StyleGAN | 2.65 |
| **DDPM** | **4.90** |

---

## DDPM의 한계와 후속 연구

| 한계 | 해결책 |
|------|--------|
| **느린 샘플링** (1000 스텝) | [DDIM](/ko/docs/components/generative/sampling) - 50 스텝 |
| **고해상도 어려움** | [Latent Diffusion](/ko/docs/architecture/generative/stable-diffusion) |
| **조건부 생성 미흡** | [Classifier-Free Guidance](/ko/docs/components/generative/sampling) |
| **고정된 분산** | Improved DDPM (학습된 분산) |

---

## 요약

| 질문 | 답변 |
|------|------|
| DDPM이 뭔가요? | 노이즈 예측 U-Net 기반 Diffusion 모델 |
| 왜 중요한가요? | Diffusion의 실용성을 처음 증명 |
| Loss가 뭔가요? | Simple MSE: $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ |
| 샘플링은 어떻게? | 1000 스텝 반복 denoising |

---

## 관련 콘텐츠

- [DDPM 수학](/ko/docs/components/generative/ddpm) - 수학적 유도
- [Sampling](/ko/docs/components/generative/sampling) - DDIM, DPM-Solver
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - Latent Diffusion
- [DiT](/ko/docs/architecture/generative/dit) - Transformer 기반 Diffusion
