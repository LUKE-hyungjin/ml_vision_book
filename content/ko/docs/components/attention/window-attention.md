---
title: "Window Attention"
weight: 5
math: true
---

# Window Attention

{{% hint info %}}
**선수지식**: [Multi-Head Attention](/ko/docs/components/attention/multi-head-attention)
{{% /hint %}}

## 한 줄 요약
> **이미지를 작은 윈도우로 나누어 각 윈도우 내에서만 Attention을 수행하고, Shifted Window로 윈도우 간 정보를 교환하는 기법**

## 왜 필요한가?

### 문제: $O(N^2)$이 고해상도에서 폭발

Self-Attention은 모든 위치 쌍을 비교하므로 $O(N^2)$ 복잡도입니다.

| 해상도 | 패치 수 (N) | $N^2$ | Attention 메모리 (FP16) |
|--------|------------|-------|------------------------|
| 224×224 | 196 (14²) | 38K | ~76 KB |
| 384×384 | 576 (24²) | 331K | ~662 KB |
| 512×512 | 1,024 (32²) | 1M | ~2 MB |
| 1024×1024 | 4,096 (64²) | 16.7M | ~33 MB |
| 2048×2048 | 16,384 (128²) | 268M | **~536 MB** |

고해상도 이미지(의료, 위성 등)에서는 메모리가 폭발합니다!

### 핵심 관찰: 이미지는 대부분 로컬 관계가 중요

```
고양이 사진에서:
- 귀 픽셀 ↔ 눈 픽셀 : 매우 관련 있음 (가까움)
- 귀 픽셀 ↔ 배경 구석 : 거의 관련 없음 (멀음)

→ 모든 쌍을 계산하는 것은 낭비!
→ "가까운 것끼리만" 계산하면 되지 않을까?
```

이것이 **Window Attention**의 아이디어입니다. CNN의 로컬 수용장(receptive field)과 Attention의 동적 가중치를 결합한 것입니다.

---

## Window Attention 기본 원리

### 1단계: 이미지를 윈도우로 나누기

```
8×8 특징 맵을 윈도우 크기 M=4로 나누면:

┌───────┬───────┐
│ 윈도우1 │ 윈도우2 │
│  4×4   │  4×4   │
├───────┼───────┤
│ 윈도우3 │ 윈도우4 │
│  4×4   │  4×4   │
└───────┴───────┘

→ 4개 윈도우, 각 윈도우 안에 16개 패치
```

### 2단계: 각 윈도우 내에서 독립적으로 Attention

```
윈도우1 (16개 패치):
  Attention 행렬: 16×16 = 256 연산

전체:
  4개 윈도우 × 256 = 1,024 연산

비교 (Global Attention):
  64×64 = 4,096 연산  → 4배 절약!
```

### 복잡도 비교

{{< figure src="/images/components/attention/ko/window-vs-global-complexity.jpeg" caption="Global vs Window Attention 복잡도 비교 — N이 커질수록 Window의 이점이 폭발적으로 증가" >}}

| 방법 | 복잡도 | 8×8 예시 | 64×64 예시 |
|------|--------|---------|-----------|
| Global Attention | $O(N^2)$ | 4,096 | 16,777,216 |
| Window Attention (M=7) | $O(N \cdot M^2)$ | 3,136 | 200,704 |
| **배율** | | 1.3× | **83×** |

$N$이 커질수록 Window Attention의 이점이 **폭발적으로** 증가합니다.

---

## Shifted Window: 윈도우 간 정보 교환

{{< figure src="/images/components/attention/ko/window-attention-shifted.jpeg" caption="Window Attention과 Shifted Window — 경계에서 끊기는 문제를 윈도우 시프트로 해결" >}}

### 문제: 윈도우 경계에서 정보가 끊긴다

```
일반 Window Attention:
┌───────┬───────┐
│ A  A  │ B  B  │
│ A  A  │ B  B  │   ← A와 B 사이에 정보 교환 없음!
├───────┼───────┤
│ C  C  │ D  D  │
│ C  C  │ D  D  │
└───────┴───────┘

고양이 얼굴이 윈도우 경계에 걸리면?
→ 왼쪽 눈과 오른쪽 눈이 서로를 볼 수 없음!
```

### 해결: 윈도우를 반 칸 이동 (Shifted Window)

```
Layer L: 일반 윈도우          Layer L+1: 시프트된 윈도우
┌───────┬───────┐            ┌──┬────────┬──┐
│       │       │            │  │        │  │
│  W1   │  W2   │     →      ├──┼────────┼──┤
│       │       │            │  │        │  │
├───────┼───────┤            │  │  W1'   │  │
│       │       │            │  │        │  │
│  W3   │  W4   │            ├──┼────────┼──┤
│       │       │            │  │        │  │
└───────┴───────┘            └──┴────────┴──┘

시프트 양: (M//2, M//2) = (3, 3) for M=7

→ L+1에서는 이전 윈도우의 경계 부분이 같은 윈도우에 속하게 됨!
→ 2개 층을 거치면 모든 패치가 인접 윈도우의 정보를 간접적으로 받음
```

### Cyclic Shift 구현

시프트하면 윈도우 크기가 불규칙해집니다. 이를 **순환 시프트(cyclic shift)**로 해결합니다:

```python
import torch

def window_partition(x, window_size):
    """특징 맵을 윈도우로 나누기"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                  W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows  # (num_windows*B, M, M, C)

def window_reverse(windows, window_size, H, W):
    """윈도우를 다시 특징 맵으로 합치기"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# Shifted Window의 cyclic shift
shift_size = window_size // 2
shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
```

---

## 구현: Window Attention 모듈

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WindowAttention(nn.Module):
    """Swin Transformer 스타일 Window Attention"""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (M, M)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = math.sqrt(head_dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative Position Bias (윈도우 내 상대 위치 편향)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 상대 위치 인덱스 계산
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        """
        x: (num_windows*B, M*M, C)
        mask: (num_windows, M*M, M*M) — shifted window용 마스크
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        # Relative Position Bias 추가
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Shifted Window 마스크 적용
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)

# 사용 예시
win_attn = WindowAttention(dim=96, window_size=7, num_heads=3)
windows = torch.randn(64, 49, 96)  # 64개 윈도우, 7×7=49 패치, 96차원
out = win_attn(windows)
print(out.shape)  # torch.Size([64, 49, 96])
```

---

## Relative Position Bias

### 왜 필요한가?

윈도우 내에서 **패치 간 상대적 위치**가 중요합니다. 바로 옆 패치와 대각선 패치는 다른 관계를 가집니다.

```
7×7 윈도우에서 (3,3) 위치 기준 상대 위치:

(-3,-3) (-3,-2) (-3,-1) (-3,0) (-3,1) (-3,2) (-3,3)
(-2,-3) (-2,-2) (-2,-1) (-2,0) (-2,1) (-2,2) (-2,3)
(-1,-3) (-1,-2) (-1,-1) (-1,0) (-1,1) (-1,2) (-1,3)
( 0,-3) ( 0,-2) ( 0,-1) (0,0)  ( 0,1) ( 0,2) ( 0,3)
( 1,-3) ( 1,-2) ( 1,-1) ( 1,0) ( 1,1) ( 1,2) ( 1,3)
( 2,-3) ( 2,-2) ( 2,-1) ( 2,0) ( 2,1) ( 2,2) ( 2,3)
( 3,-3) ( 3,-2) ( 3,-1) ( 3,0) ( 3,1) ( 3,2) ( 3,3)

→ 범위: -(M-1) ~ +(M-1) = -6 ~ +6
→ 가능한 값: (2M-1)² = 13² = 169개
```

이 상대 위치마다 학습 가능한 bias를 Attention score에 더합니다. 이것이 Swin Transformer가 **절대 위치 인코딩 없이도** 위치 정보를 인코딩하는 방법입니다.

---

## Swin Transformer에서의 사용

Swin Transformer는 Window Attention을 **계층적으로** 사용합니다:

```
Stage 1: 56×56, 윈도우 7×7 → 64개 윈도우, C=96
  ├── Block: Window Attention (일반)
  └── Block: Shifted Window Attention

Patch Merging (2×2 → 1, 채널 2배)

Stage 2: 28×28, 윈도우 7×7 → 16개 윈도우, C=192
  ├── Block: Window Attention (일반)
  └── Block: Shifted Window Attention

Patch Merging

Stage 3: 14×14, 윈도우 7×7 → 4개 윈도우, C=384
  ├── 6 Blocks (교대 반복)
  ...

Patch Merging

Stage 4: 7×7, 윈도우 7×7 → 1개 윈도우 = Global Attention!, C=768
```

**핵심**: 마지막 단계에서는 윈도우 크기 = 특징 맵 크기이므로, **자연스럽게 Global Attention**이 됩니다!

---

## Window Attention vs Global Attention

| | Global Attention | Window Attention |
|---|---|---|
| **시야** | 전체 이미지 | 윈도우 내부 |
| **복잡도** | $O(N^2)$ | $O(N \cdot M^2)$ |
| **장거리 관계** | 직접 모델링 | Shifted Window로 간접 |
| **메모리** | 큼 | 작음 |
| **대표 모델** | ViT | Swin Transformer |
| **고해상도** | 어려움 | **적합** |
| **Detection** | 비효율 | **효율적** (다양한 해상도 처리) |

---

## 핵심 정리

| 질문 | 답 |
|------|---|
| 왜 Window? | $O(N^2) \to O(N \cdot M^2)$ — 고해상도에서 필수적 |
| 윈도우 간 정보 교환은? | Shifted Window — 윈도우를 반 칸 이동 |
| 위치 정보는? | Relative Position Bias — 상대 위치별 학습 가능한 편향 |
| CNN과의 관계? | 로컬 수용장 + 동적 가중치 = CNN의 장점 + Attention의 장점 |
| 어디서 쓰이나? | Swin Transformer, 고해상도 비전 태스크 |

## 관련 콘텐츠

- [Multi-Head Attention](/ko/docs/components/attention/multi-head-attention) — Window Attention의 기반
- [Flash Attention](/ko/docs/components/attention/flash-attention) — GPU 메모리 최적화
- [Self-Attention](/ko/docs/components/attention/self-attention) — Global Attention 비교 대상
- [Swin Transformer](/ko/docs/architecture/transformer/swin-transformer) — Window Attention의 대표 모델
