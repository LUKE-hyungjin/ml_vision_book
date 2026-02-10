---
title: "Flash Attention"
weight: 6
math: true
---

# Flash Attention

{{% hint info %}}
**선수지식**: [Multi-Head Attention](/ko/docs/components/attention/multi-head-attention)
{{% /hint %}}

## 한 줄 요약
> **수학적 결과는 동일하면서, GPU 메모리 계층을 활용해 Attention을 2-4배 빠르고 메모리 효율적으로 계산하는 알고리즘**

## 왜 필요한가?

### 문제: Attention은 메모리 병목이다

Attention의 핵심 연산을 다시 봅시다:

```python
# 표준 Attention
scores = Q @ K.T          # (N, N) 행렬 생성 — O(N²) 메모리!
attn = softmax(scores)    # (N, N) 행렬 유지
output = attn @ V          # 최종 출력
```

$N \times N$ 크기의 Attention 행렬을 **통째로 GPU 메모리에 저장**해야 합니다.

| 시퀀스 길이 N | Attention 행렬 크기 | FP16 메모리 |
|--------------|-------------------|-----------|
| 1K | 1M | 2 MB |
| 4K | 16M | 32 MB |
| 16K | 256M | 512 MB |
| 64K | 4B | **8 GB** |
| 128K | 16B | **32 GB** |

GPT-4나 Claude 같은 긴 컨텍스트 모델은 Flash Attention 없이는 불가능합니다.

### GPU 메모리 계층

{{< figure src="/images/components/attention/ko/flash-attention-memory-hierarchy.jpeg" caption="GPU 메모리 계층과 표준/Flash Attention의 메모리 접근 패턴 비교" >}}

Flash Attention을 이해하려면 GPU 메모리 구조를 알아야 합니다:

```
┌─────────────────────────┐
│  SRAM (온칩 메모리)        │  ← 매우 빠름 (19TB/s), 매우 작음 (~20MB)
│  • 레지스터, 공유 메모리     │
├─────────────────────────┤
│  HBM (GPU 메모리)         │  ← 느림 (1.5TB/s), 큼 (40-80GB)
│  • A100: 40GB or 80GB    │
├─────────────────────────┤
│  CPU 메모리               │  ← 매우 느림, 매우 큼
└─────────────────────────┘

HBM → SRAM: ~12배 느림!
```

**표준 Attention의 문제**: N×N 행렬을 HBM에 쓰고, 다시 읽고, 또 쓰고... **메모리 I/O가 병목**!

```
표준 Attention의 메모리 접근:
1. Q, K를 HBM에서 읽기
2. QK^T를 HBM에 쓰기        ← N×N 쓰기!
3. QK^T를 HBM에서 읽기      ← N×N 읽기!
4. softmax 결과를 HBM에 쓰기 ← N×N 쓰기!
5. softmax 결과를 HBM에서 읽기 ← N×N 읽기!
6. attn @ V 계산 후 쓰기

→ N×N 행렬을 4번이나 HBM과 주고받음!
```

---

## Flash Attention의 핵심 아이디어

{{< figure src="/images/components/attention/ko/flash-attention-tiling.png" caption="Flash Attention의 Tiling과 Online Softmax — 수학적으로 정확히 같은 결과, 계산 순서만 변경" >}}

### Tiling: 작은 블록으로 나누어 계산

N×N 전체를 한 번에 계산하지 않고, **SRAM에 들어가는 크기의 블록**으로 나눠서 계산합니다.

```
표준 Attention:
┌─────────────────┐
│                 │
│   N×N 전체를     │  ← HBM에 통째로 저장
│   한 번에 계산    │
│                 │
└─────────────────┘

Flash Attention:
┌──┬──┬──┬──┐
│B1│B2│B3│B4│  ← 작은 블록을 SRAM에서 계산
├──┼──┼──┼──┤     HBM에는 최종 결과만 저장
│B5│B6│B7│B8│
├──┼──┼──┼──┤
│B9│..│..│..│
└──┴──┴──┴──┘
```

### 문제: Softmax는 전체를 봐야 하지 않나?

Softmax의 정의:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

분모 $\sum_j e^{x_j}$는 **모든 원소의 합**입니다. 블록 단위로 나누면 이걸 어떻게 계산하죠?

### 해결: Online Softmax

블록별로 계산하면서 **점진적으로 softmax를 업데이트**합니다:

```
블록 1 처리:
  max₁ = max(scores_block1)
  sum₁ = Σ exp(scores_block1 - max₁)
  output₁ = softmax_block1 × V_block1

블록 2 처리:
  max₂ = max(max₁, max(scores_block2))

  # 이전 블록의 결과를 새 max에 맞춰 보정
  correction = exp(max₁ - max₂)
  sum₂ = sum₁ × correction + Σ exp(scores_block2 - max₂)
  output₂ = output₁ × correction + softmax_block2 × V_block2

... (모든 블록 처리 후)

최종 output = output_n / sum_n
```

**핵심**: 이전 블록의 결과를 **보정 계수(correction)**를 곱해 업데이트 → 수학적으로 정확히 같은 결과!

---

## 알고리즘 의사코드

```
FlashAttention(Q, K, V):
    블록 크기 B 설정 (SRAM 크기에 맞춤)
    output = zeros, row_max = -inf, row_sum = zeros

    for j in range(0, N, B):        # K, V 블록 순회
        K_block = K[j:j+B]          # SRAM으로 로드
        V_block = V[j:j+B]          # SRAM으로 로드

        for i in range(0, N, B):    # Q 블록 순회
            Q_block = Q[i:i+B]      # SRAM으로 로드

            # SRAM 내에서 계산 (HBM 접근 없음!)
            scores = Q_block @ K_block.T / sqrt(d)

            new_max = max(row_max[i:i+B], row_max(scores))
            correction = exp(row_max[i:i+B] - new_max)

            row_sum[i:i+B] = row_sum[i:i+B] * correction + sum(exp(scores - new_max))
            output[i:i+B] = output[i:i+B] * correction + exp(scores - new_max) @ V_block
            row_max[i:i+B] = new_max

    output = output / row_sum       # 최종 정규화
    return output
```

---

## 메모리 및 속도 비교

### 메모리

| 방법 | Attention 행렬 메모리 | 추가 메모리 |
|------|---------------------|-----------|
| 표준 Attention | $O(N^2)$ | - |
| Flash Attention | **$O(N)$** | 블록 크기만큼 |

**$O(N^2)$에서 $O(N)$으로!** — N=16K일 때 256M → 16K 수준으로 감소.

### 속도 (A100 GPU 기준)

| 시퀀스 길이 | 표준 Attention | Flash Attention v2 | 배율 |
|-----------|---------------|-------------------|------|
| 1K | 0.5ms | 0.2ms | 2.5× |
| 4K | 7.6ms | 2.8ms | 2.7× |
| 16K | 120ms | 40ms | 3.0× |

Flash Attention v2는 v1 대비 약 2배 더 빠릅니다 (더 나은 병렬화).

---

## 사용법

### PyTorch 2.0+ (가장 간단)

```python
import torch
import torch.nn.functional as F

# scaled_dot_product_attention이 자동으로 Flash Attention 사용
q = torch.randn(32, 12, 196, 64, device='cuda', dtype=torch.float16)
k = torch.randn(32, 12, 196, 64, device='cuda', dtype=torch.float16)
v = torch.randn(32, 12, 196, 64, device='cuda', dtype=torch.float16)

# GPU에서 자동으로 Flash Attention 적용!
output = F.scaled_dot_product_attention(q, k, v)
print(output.shape)  # torch.Size([32, 12, 196, 64])
```

### Causal Mask와 함께 사용

```python
# 자기회귀 모델 (GPT 등)에서 causal mask
output = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True  # 미래 토큰 마스킹을 효율적으로 처리
)
```

### 기존 코드에서 교체

```python
# Before (표준 Attention)
class OldAttention(nn.Module):
    def forward(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        return attn @ v

# After (Flash Attention — 한 줄 교체!)
class NewAttention(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)
```

**수학적 결과는 완전히 동일**합니다. 단순히 계산 순서를 바꾼 것뿐입니다.

---

## Flash Attention의 한계

| 한계 | 설명 | 대안 |
|------|------|------|
| **Attention 가중치 추출 어려움** | 중간 N×N 행렬을 저장하지 않으므로 | 시각화 필요 시 표준 Attention 사용 |
| **GPU 전용** | CUDA 커널 기반 | CPU에서는 효과 없음 |
| **커스텀 마스크 제한** | 일부 복잡한 마스크 패턴 미지원 | PyTorch 버전에 따라 개선 중 |
| **FP16/BF16 권장** | FP32에서는 효과 감소 | Mixed Precision 함께 사용 |

---

## Flash Attention v1 vs v2 vs v3

| | v1 (2022) | v2 (2023) | v3 (2024) |
|---|---|---|---|
| **핵심 개선** | Tiling + Online Softmax | 더 나은 병렬화, 워프 분할 | Hopper GPU 활용 (H100) |
| **속도** | 기준 | v1 대비 ~2× | v2 대비 ~1.5-2× |
| **GPU 활용률** | ~30-50% | ~50-70% | ~75%+ |
| **PyTorch 지원** | torch 2.0+ | torch 2.0+ (자동 선택) | flash-attn 라이브러리 |

```python
# PyTorch는 자동으로 최적의 구현을 선택
# Flash Attention이 가능하면 Flash를, 아니면 Memory-Efficient를, 아니면 Math를 사용
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(q, k, v)
```

---

## Flash Attention이 왜 중요한가?

Flash Attention 이전에는 긴 시퀀스를 처리하려면:
1. 시퀀스를 잘라서 처리 (정보 손실)
2. 근사 Attention 사용 (정확도 손실)
3. 매우 큰 GPU 메모리 필요 (비용 문제)

Flash Attention 이후:
- **정확도 손실 없이** 긴 시퀀스 처리
- GPT-4, Claude 등 128K+ 컨텍스트 모델 가능
- ViT 학습 시간 15-20% 단축
- 실질적으로 **모든 Transformer 모델**이 Flash Attention 사용

---

## 핵심 정리

| 질문 | 답 |
|------|---|
| 무엇이 달라지나? | 계산 **순서**만 변경 — 수학적 결과는 동일 |
| 왜 빠른가? | GPU 메모리 I/O를 최소화 (HBM ↔ SRAM 왕복 감소) |
| 메모리는? | $O(N^2) \to O(N)$ — 긴 시퀀스에서 극적 감소 |
| 어떻게 쓰나? | `F.scaled_dot_product_attention()` — 한 줄 교체 |
| 한계는? | Attention Map 시각화 어려움, GPU 전용 |

## 관련 콘텐츠

- [Multi-Head Attention](/ko/docs/components/attention/multi-head-attention) — Flash Attention이 최적화하는 대상
- [Window Attention](/ko/docs/components/attention/window-attention) — 또 다른 효율적 Attention 기법
- [Self-Attention](/ko/docs/components/attention/self-attention) — 표준 Attention의 수학
- [ViT](/ko/docs/architecture/transformer/vit) — Flash Attention의 대표적 수혜 모델
