---
title: "샘플링"
weight: 3
math: true
---

# 샘플링 (Sampling)

## 개요

> 💡 **샘플링**: 확률분포에서 값을 추출하는 방법

생성 모델과 확률적 학습의 핵심입니다.

### Temperature Scaling 시각화

![Temperature Scaling](/images/probability/ko/sampling-temperature.svg)

## 기본 샘플링

### 균등 분포 샘플링

```python
import torch

# [0, 1) 균등 분포
samples = torch.rand(1000)

# [a, b) 균등 분포
a, b = 2, 5
samples = a + (b - a) * torch.rand(1000)
```

### 가우시안 샘플링

```python
# N(0, 1) 표준정규분포
samples = torch.randn(1000)

# N(mu, sigma^2)
mu, sigma = 5, 2
samples = mu + sigma * torch.randn(1000)
```

## Reparameterization Trick

확률적 노드를 통한 역전파를 가능하게 하는 기법:

**문제**: z ~ N(μ, σ²) 에서 샘플링하면 미분 불가능

**해결**: z = μ + σ * ε, ε ~ N(0, 1)

```python
class VAE(nn.Module):
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 미분과 무관
        return mu + std * eps        # 미분 가능

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
```

## 카테고리 샘플링

### Argmax (결정적)

```python
logits = model(x)
prediction = logits.argmax(dim=-1)  # 항상 같은 결과
```

### 확률적 샘플링

```python
probs = torch.softmax(logits, dim=-1)
samples = torch.multinomial(probs, num_samples=1)
```

### Gumbel-Softmax (미분 가능 샘플링)

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
    y_soft = torch.softmax((logits + gumbels) / tau, dim=-1)

    if hard:
        # Forward: one-hot, Backward: soft
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft

    return y_soft
```

## Temperature Scaling

샘플링의 다양성 조절:

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- T → 0: Argmax (결정적)
- T = 1: 원래 분포
- T → ∞: 균등 분포 (랜덤)

```python
def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## Top-k, Top-p 샘플링

### Top-k

상위 k개 토큰만 고려:

```python
def top_k_sampling(logits, k=50):
    top_k_logits, top_k_indices = logits.topk(k)
    probs = torch.softmax(top_k_logits, dim=-1)
    sample_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, sample_idx)
```

### Top-p (Nucleus)

누적 확률 p까지의 토큰만 고려:

```python
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # 누적 확률 p 초과 토큰 제거
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = torch.softmax(sorted_logits, dim=-1)
    return sorted_indices.gather(-1, torch.multinomial(probs, num_samples=1))
```

## 관련 콘텐츠

- [확률분포](/ko/docs/math/probability/distribution)
- [VAE](/ko/docs/architecture/generative/vae) - Reparameterization 활용
- [Diffusion](/ko/docs/math/diffusion) - 반복적 샘플링
- [VLM](/ko/docs/architecture/multimodal/vlm) - 텍스트 생성 시 샘플링
