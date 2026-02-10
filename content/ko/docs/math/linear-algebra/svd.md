---
title: "SVD"
weight: 7
math: true
---

# 특이값 분해 (Singular Value Decomposition)

{{% hint info %}}
**선수지식**: [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue)
{{% /hint %}}

> **한 줄 요약**: SVD는 **모든 행렬을 해부하는 방법**입니다. LoRA, 모델 압축, 추천 시스템의 핵심입니다.

## 왜 SVD를 배워야 하나요?

### 문제 상황 1: "LoRA가 왜 적은 파라미터로 작동하나요?"

```python
# GPT-3: 175B 파라미터
# LoRA 파인튜닝: 0.1% 파라미터만 학습

from peft import LoraConfig, get_peft_model
config = LoraConfig(r=8, lora_alpha=16)  # r이 뭐지?
```

→ LoRA는 **저랭크 행렬 분해(SVD의 핵심 아이디어)**를 활용합니다.

### 문제 상황 2: "모델 압축을 어떻게 하나요?"

```python
# 4096×4096 = 16M 파라미터를 줄이고 싶음
# 어떻게?
```

→ **SVD로 저랭크 근사**하면 파라미터 수를 획기적으로 줄일 수 있습니다.

### 문제 상황 3: "추천 시스템의 행렬 분해가 뭔가요?"

```python
# Netflix Prize: 사용자-영화 평점 행렬
# 빈 칸을 어떻게 예측?
```

→ **SVD**로 사용자/영화를 저차원 벡터로 표현합니다.

---

## SVD란 무엇인가?

### 핵심 정의

**모든** m × n 행렬 A는 다음과 같이 분해됩니다:

$$
A = U \Sigma V^T
$$

- $U$: m × m **직교 행렬** (좌특이벡터)
- $\Sigma$: m × n **대각 행렬** (특이값: $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$)
- $V$: n × n **직교 행렬** (우특이벡터)

```python
import numpy as np
import torch

# 예시: 3×2 행렬
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U: {U.shape}")   # (3, 2) - 왼쪽 특이벡터
print(f"S: {S.shape}")   # (2,) - 특이값
print(f"Vt: {Vt.shape}") # (2, 2) - 오른쪽 특이벡터의 전치

# 복원 확인
A_reconstructed = U @ np.diag(S) @ Vt
print(f"복원 오차: {np.linalg.norm(A - A_reconstructed):.10f}")  # ≈ 0
```

### 고유값 분해와의 차이

| 특성 | 고유값 분해 | SVD |
|------|-----------|-----|
| 적용 가능 행렬 | 정사각 행렬만 | **모든 행렬** |
| 분해 형태 | $A = P\Lambda P^{-1}$ | $A = U\Sigma V^T$ |
| 값의 특성 | 고유값 (음수 가능) | 특이값 (항상 ≥ 0) |
| 벡터의 특성 | 직교 보장 안 됨 | **항상 직교** |

```python
# 정사각이 아닌 행렬도 OK!
A = np.random.randn(100, 50)  # 100×50
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"SVD 성공: U{U.shape}, S{S.shape}, Vt{Vt.shape}")
# U(100,50), S(50,), Vt(50,50)
```

### 특이값과 고유값의 관계

**$A^T A$의 고유값** = **A의 특이값**의 제곱

$$
A^T A = V\Sigma^T U^T U\Sigma V^T = V\Sigma^2 V^T
$$

```python
A = np.random.randn(5, 3)

# SVD로 특이값
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# A^T A의 고유값
AtA = A.T @ A
eigenvalues = np.linalg.eigvalsh(AtA)

print(f"특이값²: {np.sort(S**2)[::-1]}")
print(f"고유값: {np.sort(eigenvalues)[::-1]}")
# 같음!
```

---

## 기하학적 의미

### SVD = 회전 + 스케일 + 회전

행렬 A의 작용을 세 단계로 분해:

$$
A\mathbf{x} = U(\Sigma(V^T\mathbf{x}))
$$

1. **$V^T$**: 입력 공간에서 회전 (직교 변환)
2. **$\Sigma$**: 각 축 방향으로 늘리거나 줄임 (스케일링)
3. **$U$**: 출력 공간에서 회전 (직교 변환)

{{< figure src="/images/math/linear-algebra/ko/svd-decomposition.png" caption="SVD는 모든 행렬을 '회전(V^T) → 스케일(Σ) → 회전(U)' 세 단계로 분해한다" >}}

```python
import matplotlib.pyplot as plt
import numpy as np

# 행렬 정의
A = np.array([[3, 1],
              [1, 2]])

# SVD 분해
U, S, Vt = np.linalg.svd(A)

# 단위원
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# 각 단계 시각화
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 원본
axes[0].plot(circle[0], circle[1])
axes[0].set_title('1. 원본')
axes[0].axis('equal')

# V^T 적용 (회전)
step1 = Vt @ circle
axes[1].plot(step1[0], step1[1])
axes[1].set_title('2. V^T: 회전')
axes[1].axis('equal')

# Σ 적용 (스케일)
step2 = np.diag(S) @ step1
axes[2].plot(step2[0], step2[1])
axes[2].set_title(f'3. Σ: 스케일 ({S[0]:.1f}, {S[1]:.1f})')
axes[2].axis('equal')

# U 적용 (회전)
step3 = U @ step2
axes[3].plot(step3[0], step3[1])
axes[3].set_title('4. U: 회전 (최종)')
axes[3].axis('equal')

plt.tight_layout()
plt.show()
```

### 특이값의 의미

- **$\sigma_1$**: 행렬이 가장 많이 늘리는 방향의 배율
- **$\sigma_2$**: 두 번째로 많이 늘리는 방향의 배율
- ...
- 특이값이 0에 가까우면 그 방향 정보는 "사라짐"

---

## 저랭크 근사 (Low-Rank Approximation)

### 핵심 아이디어

상위 r개의 특이값만 사용하면 원래 행렬을 **근사**할 수 있습니다:

$$
A \approx A_r = U_r \Sigma_r V_r^T
$$

- $U_r$: U의 첫 r개 열
- $\Sigma_r$: 상위 r개 특이값
- $V_r^T$: $V^T$의 첫 r개 행

{{< figure src="/images/math/linear-algebra/ko/svd-low-rank.png" caption="저랭크 근사: 상위 r개 특이값만 사용하여 원래 행렬을 근사 — 작은 특이값은 버려도 정보 손실이 적다" >}}

### Eckart-Young 정리

**저랭크 근사의 최적성**: 주어진 랭크 r에 대해, SVD 기반 근사가 **Frobenius 노름 관점에서 최적**입니다.

$$
A_r = \arg\min_{\text{rank}(B) = r} \|A - B\|_F
$$

### 파라미터 절약

| 원본 | 저랭크 근사 | 절약 |
|------|-----------|------|
| m × n | (m × r) + (r) + (r × n) | 큼 |
| 4096 × 4096 = 16M | (4096 × 64) + (64) + (64 × 4096) ≈ 524K | **97%** |

```python
def low_rank_approximation(A, r):
    """
    A를 랭크 r로 근사
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # 상위 r개만 선택
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    # 근사 행렬
    A_r = U_r @ np.diag(S_r) @ Vt_r

    # 오차 계산
    error = np.linalg.norm(A - A_r, 'fro') / np.linalg.norm(A, 'fro')

    return A_r, error

# 테스트
A = np.random.randn(1000, 500)

for r in [10, 50, 100, 200]:
    A_r, error = low_rank_approximation(A, r)
    original_params = A.shape[0] * A.shape[1]
    approx_params = A.shape[0] * r + r + r * A.shape[1]
    savings = 1 - approx_params / original_params

    print(f"랭크 {r:3d}: 오차 {error:.4f}, 파라미터 절약 {savings:.1%}")
```

### 설명된 분산 (Explained Variance)

특이값의 비율로 "얼마나 정보를 보존했는지" 측정:

$$
\text{Explained Variance Ratio} = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{n} \sigma_i^2}
$$

```python
def explained_variance(S, r):
    """상위 r개 특이값이 설명하는 분산 비율"""
    total_var = np.sum(S**2)
    explained = np.sum(S[:r]**2)
    return explained / total_var

A = np.random.randn(100, 50)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

for r in [5, 10, 20, 30]:
    ev = explained_variance(S, r)
    print(f"랭크 {r}: 설명된 분산 {ev:.1%}")
```

---

## 딥러닝에서의 적용

### 1. LoRA (Low-Rank Adaptation)

**문제**: GPT-3 같은 거대 모델을 파인튜닝하려면 모든 파라미터를 저장/학습해야 함

**해결**: 가중치 변화 $\Delta W$를 저랭크로 분해

$$
W' = W + \Delta W = W + BA
$$

- $W$: 원본 가중치 (고정)
- $B$: (d × r) 행렬
- $A$: (r × k) 행렬
- $r \ll \min(d, k)$

**원리**: 대부분의 파인튜닝은 저랭크 변화로 충분!

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 원본 가중치 (고정)
        self.W = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA 파라미터 (학습)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        # 스케일링 팩터
        self.scaling = alpha / rank

    def forward(self, x):
        # 원본 출력 + LoRA 출력
        return x @ self.W.T + (x @ self.A.T @ self.B.T) * self.scaling

# 파라미터 비교
in_dim, out_dim, rank = 4096, 4096, 8
lora = LoRALinear(in_dim, out_dim, rank)

original_params = in_dim * out_dim
lora_params = rank * in_dim + out_dim * rank

print(f"원본 파라미터: {original_params:,}")      # 16,777,216
print(f"LoRA 파라미터: {lora_params:,}")          # 65,536
print(f"절약: {1 - lora_params/original_params:.2%}")  # 99.61%
```

### 2. 모델 압축 (Weight Compression)

학습된 모델의 가중치를 SVD로 압축:

```python
def compress_layer(weight, rank):
    """
    가중치 행렬을 저랭크로 압축
    """
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    # 상위 r개만 사용
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    # 압축된 표현: (U * sqrt(S)) @ (sqrt(S) * Vt)
    # 두 행렬로 분해
    sqrt_S = torch.sqrt(S_r)
    A = U_r * sqrt_S  # (out, rank)
    B = sqrt_S.unsqueeze(1) * Vt_r  # (rank, in)

    return A, B

class CompressedLinear(nn.Module):
    """SVD로 압축된 Linear Layer"""
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)  # (out, rank)
        self.B = nn.Parameter(B, requires_grad=False)  # (rank, in)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        # x @ B^T @ A^T = x @ (AB)^T
        out = x @ self.B.T @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out

# 사용 예시
original = nn.Linear(1024, 1024)
A, B = compress_layer(original.weight.data, rank=64)
compressed = CompressedLinear(A, B, original.bias.data)

# 오차 확인
x = torch.randn(32, 1024)
original_out = original(x)
compressed_out = compressed(x)
error = torch.norm(original_out - compressed_out) / torch.norm(original_out)
print(f"상대 오차: {error.item():.4f}")
```

### 3. 추천 시스템 (Matrix Factorization)

**문제**: 사용자-아이템 평점 행렬의 빈 칸 채우기

**해결**: SVD로 사용자/아이템을 저차원 임베딩으로 표현

```python
import numpy as np

# 사용자-영화 평점 행렬 (0 = 미평가)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
# 행: 사용자, 열: 영화

# 평균으로 결측값 임시 채우기
filled = ratings.copy().astype(float)
filled[filled == 0] = np.nan
col_means = np.nanmean(filled, axis=0)
for i in range(filled.shape[1]):
    filled[np.isnan(filled[:, i]), i] = col_means[i]

# SVD
U, S, Vt = np.linalg.svd(filled, full_matrices=False)

# 저랭크 근사 (랭크 2)
r = 2
U_r = U[:, :r]
S_r = S[:r]
Vt_r = Vt[:r, :]

# 예측 평점
predicted = U_r @ np.diag(S_r) @ Vt_r
print("예측 평점 행렬:")
print(np.round(predicted, 1))

# 사용자 0이 영화 2에 줄 예상 평점
print(f"\n사용자 0 → 영화 2 예측: {predicted[0, 2]:.1f}")
```

### 4. 이미지 압축

```python
from PIL import Image
import numpy as np

def compress_image_svd(image_path, ranks=[5, 20, 50, 100]):
    """이미지를 SVD로 압축"""
    # 이미지 로드 (그레이스케일)
    img = np.array(Image.open(image_path).convert('L'), dtype=float)
    print(f"원본 크기: {img.shape}")

    # SVD
    U, S, Vt = np.linalg.svd(img, full_matrices=False)

    results = []
    for r in ranks:
        # 저랭크 근사
        img_r = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]

        # 압축률 계산
        original_size = img.shape[0] * img.shape[1]
        compressed_size = img.shape[0] * r + r + r * img.shape[1]
        compression_ratio = original_size / compressed_size

        # PSNR 계산
        mse = np.mean((img - img_r) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

        results.append({
            'rank': r,
            'compression': compression_ratio,
            'psnr': psnr,
            'image': img_r
        })

        print(f"랭크 {r:3d}: 압축비 {compression_ratio:.1f}x, PSNR {psnr:.1f}dB")

    return results

# 테스트 (실제 이미지 경로 필요)
# results = compress_image_svd('test_image.jpg')
```

### 5. Truncated SVD (대규모 행렬용)

전체 SVD는 O(mn²) 복잡도. 큰 행렬에서는 **Truncated SVD**로 상위 k개만 계산:

```python
from scipy.sparse.linalg import svds
import numpy as np

# 대규모 희소 행렬
A = np.random.randn(10000, 5000)

# 전체 SVD (느림)
# U, S, Vt = np.linalg.svd(A)  # 메모리 부족 가능

# Truncated SVD (빠름) - 상위 100개만
k = 100
U_k, S_k, Vt_k = svds(A, k=k)

print(f"U_k: {U_k.shape}")   # (10000, 100)
print(f"S_k: {S_k.shape}")   # (100,)
print(f"Vt_k: {Vt_k.shape}") # (100, 5000)
```

---

## Truncated SVD vs PCA

둘 다 차원 축소에 사용되지만:

| 특성 | PCA | Truncated SVD |
|------|-----|---------------|
| 중심화 | 평균 빼야 함 | 안 빼도 됨 |
| 희소 행렬 | 밀집화됨 | 희소성 유지 |
| 적용 | 일반 데이터 | 텍스트, 희소 데이터 |

```python
from sklearn.decomposition import TruncatedSVD, PCA

# 희소 데이터 (예: TF-IDF)
X = np.random.randn(1000, 500)
X[X < 1.5] = 0  # 희소하게 만듦

# Truncated SVD: 희소성 유지
svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X)

# PCA: 중심화로 밀집화됨
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

print(f"SVD 출력: {X_svd.shape}")
print(f"PCA 출력: {X_pca.shape}")
```

---

## 코드로 확인: 전체 파이프라인

```python
import numpy as np
import torch

print("=== SVD 기본 ===")
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"A: {A.shape}")
print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

# 복원
A_reconstructed = U @ np.diag(S) @ Vt
print(f"복원 오차: {np.linalg.norm(A - A_reconstructed):.10f}")

print("\n=== 저랭크 근사 ===")
# 랭크 1 근사
r = 1
A_r1 = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
error_r1 = np.linalg.norm(A - A_r1, 'fro') / np.linalg.norm(A, 'fro')
print(f"랭크 1 근사 오차: {error_r1:.4f}")

# 랭크 2 근사
r = 2
A_r2 = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
error_r2 = np.linalg.norm(A - A_r2, 'fro') / np.linalg.norm(A, 'fro')
print(f"랭크 2 근사 오차: {error_r2:.4f}")

print(f"\n특이값: {S}")
print("특이값이 빠르게 감소하면 저랭크 근사가 잘 작동!")

print("\n=== LoRA 시뮬레이션 ===")
# 원본 가중치
W = torch.randn(256, 256)

# LoRA 업데이트 (랭크 8)
r = 8
A = torch.randn(r, 256) * 0.01
B = torch.zeros(256, r)

# 파인튜닝 후 (시뮬레이션)
B = torch.randn(256, r) * 0.01
delta_W = B @ A  # 저랭크 업데이트

W_new = W + delta_W

print(f"원본 W: {W.shape}")
print(f"delta_W 랭크: {torch.linalg.matrix_rank(delta_W).item()}")  # 8
print(f"LoRA 파라미터: {r * 256 + 256 * r}")  # 4096
print(f"원본 파라미터: {256 * 256}")  # 65536

print("\n=== 특이값과 고유값 관계 ===")
A = np.random.randn(4, 3)

# A의 특이값
_, S, _ = np.linalg.svd(A)

# A^T A의 고유값
AtA = A.T @ A
eigenvalues = np.linalg.eigvalsh(AtA)

print(f"특이값²:  {np.sort(S**2)[::-1]}")
print(f"고유값:   {np.sort(eigenvalues)[::-1]}")
print("일치함!")
```

---

## 핵심 정리

| 개념 | 수식 | 딥러닝 적용 |
|------|------|------------|
| SVD | $A = U\Sigma V^T$ | 행렬 분해의 기본 |
| 저랭크 근사 | $A \approx U_r \Sigma_r V_r^T$ | 모델 압축 |
| LoRA | $W' = W + BA$ | 효율적 파인튜닝 |
| Truncated SVD | 상위 k개만 계산 | 대규모 행렬 |

## 핵심 통찰

1. **SVD = 회전 + 스케일 + 회전**: 모든 행렬의 작용을 이렇게 분해 가능
2. **특이값 = 중요도**: 큰 특이값이 중요한 정보를 담음
3. **저랭크 근사가 가능한 이유**: 실제 데이터는 redundancy가 많음
4. **LoRA의 핵심**: 파인튜닝은 저랭크 변화로 충분함

---

## 선형대수 학습 완료!

벡터 → 행렬 → 고유값 → SVD를 모두 배웠습니다.

**이제 할 수 있는 것**:
- Linear Layer가 하는 일 이해
- Shape 에러 즉시 해결
- PCA 직접 구현
- LoRA가 왜 작동하는지 이해
- 모델 압축 기법 설계

## 관련 콘텐츠

- [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue) - SVD의 기반
- [행렬 연산](/ko/docs/math/linear-algebra/matrix) - 기본 연산
- [LoRA](/ko/docs/components/training/peft/lora) - SVD 기반 효율적 학습
- [Attention](/ko/docs/components/attention) - 행렬 연산의 집약체
