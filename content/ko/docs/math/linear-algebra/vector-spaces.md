---
title: "벡터 공간"
weight: 3
math: true
---

# 벡터 공간 (Vector Spaces)

{{% hint info %}}
**선수지식**: [벡터 기초](/ko/docs/math/linear-algebra/vector)
{{% /hint %}}

> **한 줄 요약**: 벡터 공간은 벡터들이 사는 "세계의 규칙"이며, **기저(basis)**를 이해하면 딥러닝의 차원 축소, 임베딩 공간, LoRA가 왜 작동하는지 알 수 있습니다.

## 왜 벡터 공간을 배워야 하나요?

### 문제 상황 1: "왜 768차원 임베딩이 50차원으로 줄여도 되나요?"

```python
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=50).fit_transform(X_768d)
# 768차원 → 50차원인데 정보 95%가 보존된다고?
```

→ 데이터가 실제로 사용하는 **부분공간의 차원**이 50에 가깝기 때문입니다.

### 문제 상황 2: "LoRA에서 rank=8이면 왜 충분한가요?"

```python
config = LoraConfig(r=8)  # 4096×4096 행렬을 rank 8로?
```

→ 가중치 변화 ΔW가 **저차원 부분공간**에 집중되기 때문입니다.

### 문제 상황 3: "선형 독립이 뭔가요?"

```python
# Feature가 3개인데 rank가 2라고?
A = torch.tensor([[1., 2., 3.],
                  [2., 4., 6.],  # = 2 × 첫 번째 행
                  [0., 1., 0.]])
print(torch.linalg.matrix_rank(A))  # 2
```

→ 하나의 행이 다른 행들로 **표현 가능**하면 "독립"이 아닙니다.

---

## 벡터 공간이란?

### 직관: 벡터들이 사는 세계의 규칙

벡터 공간은 **"덧셈과 스칼라 곱이 자유롭게 가능한 벡터들의 집합"**입니다.

어떤 집합 V가 벡터 공간이 되려면:
1. **덧셈 닫힘**: V의 벡터 두 개를 더해도 V 안에 있다
2. **스칼라 곱 닫힘**: V의 벡터에 상수를 곱해도 V 안에 있다
3. **영벡터 존재**: 아무것도 안 하는 벡터 **0**이 있다

```python
import torch

# ℝ³은 벡터 공간
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])

# 덧셈 닫힘: 결과도 ℝ³
print(v1 + v2)  # [5, 7, 9] ∈ ℝ³ ✓

# 스칼라 곱 닫힘: 결과도 ℝ³
print(3.0 * v1)  # [3, 6, 9] ∈ ℝ³ ✓

# 영벡터 존재
print(torch.zeros(3))  # [0, 0, 0] ∈ ℝ³ ✓
```

### 딥러닝에서의 벡터 공간

| 벡터 공간 | 차원 | 예시 |
|----------|------|------|
| 픽셀 공간 | 28×28 = 784 | MNIST 이미지 |
| 임베딩 공간 | 768 | BERT 출력 |
| 잠재 공간 | 4×64×64 | Stable Diffusion latent |
| 가중치 공간 | 수백만~수십억 | 모델 파라미터 전체 |

{{< figure src="/images/math/linear-algebra/ko/span-and-basis.jpeg" caption="Span과 기저: 벡터들의 선형 결합으로 도달 가능한 공간과, 같은 점을 다른 기저로 표현하기" >}}

---

## 선형 결합 (Linear Combination)

### 정의

벡터들에 각각 상수를 곱해서 더한 것:

$$
\mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k
$$

```python
v1 = torch.tensor([1.0, 0.0])
v2 = torch.tensor([0.0, 1.0])

# v1과 v2의 선형 결합으로 2D 평면의 모든 점 표현 가능
point = 3.0 * v1 + 2.0 * v2  # [3, 2]
```

### 딥러닝 적용: Linear Layer의 본질

```python
# y = W @ x + b에서
# y의 각 원소 = W의 행 벡터들과 x의 선형 결합

W = torch.tensor([[1., 2.],
                  [3., 4.],
                  [5., 6.]])  # 3×2
x = torch.tensor([0.5, 1.5])

# y[0] = 0.5 * [1,2]의 1번째 + 1.5 * [1,2]의 2번째 = 0.5*1 + 1.5*2 = 3.5
y = W @ x  # x[0]*W[:,0] + x[1]*W[:,1] = W의 열벡터들의 선형 결합!
print(y)   # [3.5, 7.5, 11.5]
```

**핵심 통찰**: `W @ x`는 **W의 열 벡터들을 x의 값으로 선형 결합**한 것입니다.

---

## 생성 (Span)

### 정의

벡터들의 모든 가능한 선형 결합이 만드는 공간:

$$
\text{span}(\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k \mid c_i \in \mathbb{R}\}
$$

### 비유: 물감 섞기

- 빨강, 파랑 물감 → 섞으면 **보라 계열** 전부 가능
- 빨강, 파랑, 노랑 물감 → **거의 모든 색** 가능
- 빨강, 진빨강(=빨강×2) → 여전히 **빨강 계열만** 가능 (중복!)

```python
# 두 벡터의 span
v1 = torch.tensor([1.0, 0.0, 0.0])  # x축
v2 = torch.tensor([0.0, 1.0, 0.0])  # y축

# span(v1, v2) = xy 평면 (z=0인 모든 점)
# 3D 공간의 벡터인데, 실제로는 2D 평면만 커버

# 같은 방향의 벡터를 추가해도 span은 변하지 않음
v3 = torch.tensor([2.0, 0.0, 0.0])  # = 2*v1
# span(v1, v2, v3) = span(v1, v2) = 여전히 xy 평면
```

---

## 선형 독립 (Linear Independence)

### 정의

어떤 벡터도 **나머지 벡터들의 선형 결합으로 표현할 수 없을 때** 선형 독립:

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k = \mathbf{0} \quad \Rightarrow \quad c_1 = c_2 = \ldots = c_k = 0
$$

→ 영벡터를 만드는 유일한 방법이 "모든 계수가 0"일 때

### 직관: 새로운 정보를 주는가?

```python
# 선형 독립 (각각 새로운 방향)
v1 = torch.tensor([1.0, 0.0, 0.0])
v2 = torch.tensor([0.0, 1.0, 0.0])
v3 = torch.tensor([0.0, 0.0, 1.0])
# v3는 v1, v2로 표현 불가능 → 독립!

# 선형 종속 (중복된 정보)
u1 = torch.tensor([1.0, 0.0, 0.0])
u2 = torch.tensor([0.0, 1.0, 0.0])
u3 = torch.tensor([1.0, 1.0, 0.0])  # = u1 + u2
# u3는 u1 + u2로 표현 가능 → 종속!
```

### Rank와의 관계

**행렬의 Rank = 선형 독립인 열(또는 행)의 수**

```python
# 3개 열 벡터를 가진 행렬
A = torch.tensor([[1., 0., 1.],
                  [0., 1., 1.],
                  [0., 0., 0.]])

print(torch.linalg.matrix_rank(A))  # 2
# 세 번째 열 = 첫 번째 열 + 두 번째 열 → 독립인 열은 2개
```

### 딥러닝 적용: Feature 중복 감지

```python
# 모델의 Feature가 중복되면 비효율적
features = model.extract_features(images)  # (batch, 512)

# 특이값으로 "실질적 차원" 확인
U, S, V = torch.linalg.svd(features)
effective_dim = (S > 0.01 * S[0]).sum()  # 의미있는 특이값 수
print(f"512차원 중 실질적 차원: {effective_dim}")
# 만약 50이면 → 462개 차원은 중복된 정보!
```

---

## 기저 (Basis)

### 정의

벡터 공간 V의 **기저(basis)**는 두 조건을 만족하는 벡터 집합:

1. **선형 독립**: 벡터들 사이에 중복 없음
2. **V를 생성(span)**: 이 벡터들로 V의 모든 벡터 표현 가능

$$
V = \text{span}(\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n) \quad \text{where } \mathbf{b}_i\text{들이 선형 독립}
$$

### 비유: 좌표계

기저는 **"공간을 설명하는 좌표축"**입니다.

```python
# 표준 기저 (Standard Basis) - 가장 자연스러운 좌표축
e1 = torch.tensor([1., 0., 0.])  # x축
e2 = torch.tensor([0., 1., 0.])  # y축
e3 = torch.tensor([0., 0., 1.])  # z축

# 모든 3D 벡터 = 표준 기저의 선형 결합
v = torch.tensor([3., 5., 2.])
# v = 3*e1 + 5*e2 + 2*e3

# 다른 기저도 가능! (독립이고 span하면 됨)
b1 = torch.tensor([1., 1., 0.])
b2 = torch.tensor([1., -1., 0.])
b3 = torch.tensor([0., 0., 1.])
# 이것도 ℝ³의 유효한 기저
```

### 차원 (Dimension)

**차원 = 기저 벡터의 개수**

어떤 기저를 선택해도 개수는 항상 같습니다.

```python
# ℝ³의 차원 = 3
# 어떤 기저를 골라도 항상 3개

# BERT 임베딩 공간의 차원 = 768
# → 768개의 독립적인 방향이 존재
```

---

## 부분공간 (Subspace)

### 정의

큰 벡터 공간 안에 있는 **작은 벡터 공간**:

```python
# ℝ³ 안의 부분공간들:
# - 원점을 지나는 직선 (1차원 부분공간)
# - 원점을 지나는 평면 (2차원 부분공간)
# - ℝ³ 자체 (3차원)
# - {0} (0차원)

# xy 평면은 ℝ³의 2차원 부분공간
# span([1,0,0], [0,1,0]) = {[x, y, 0] | x, y ∈ ℝ}
```

### 딥러닝에서의 부분공간

**1. LoRA: 파라미터 업데이트가 사는 부분공간**

```python
# 원본 가중치: W ∈ ℝ^(4096×4096) — 약 1,600만 파라미터
# LoRA: ΔW = B @ A where B ∈ ℝ^(4096×8), A ∈ ℝ^(8×4096)
# ΔW의 rank ≤ 8 → 8차원 부분공간에서만 변화!

# 왜 이게 되는가?
# 파인튜닝 시 가중치 변화가 실제로 저차원 부분공간에 집중되기 때문
d, r = 4096, 8
B = torch.randn(d, r)
A = torch.randn(r, d)
delta_W = B @ A  # rank ≤ 8인 행렬

print(f"원본 파라미터: {d*d:,}")          # 16,777,216
print(f"LoRA 파라미터: {d*r + r*d:,}")   # 65,536 (0.4%!)
```

**2. PCA: 데이터가 사는 부분공간 찾기**

```python
import numpy as np

# 100차원 데이터지만 실제로는 3차원 부분공간에 집중
np.random.seed(42)
# 3개의 기저 벡터로 데이터 생성
basis = np.random.randn(3, 100)  # 3개의 100차원 벡터
coefficients = np.random.randn(500, 3)  # 500개 데이터의 계수
X = coefficients @ basis + 0.01 * np.random.randn(500, 100)  # 약간의 노이즈

# 특이값 확인
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(f"상위 특이값: {S[:5].round(1)}")
# 상위 3개만 크고, 나머지는 거의 0
# → 데이터가 3차원 부분공간에 살고 있음!
```

**3. Null Space (영공간): Ax = 0의 해 공간**

```python
# 영공간 = 행렬에 의해 0으로 보내지는 벡터들의 공간
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])

# 이 행렬의 rank = 2 (3번째 행 = 행1과 행2의 선형 결합)
print(f"Rank: {torch.linalg.matrix_rank(A)}")  # 2

# 영공간의 차원 = 열 수 - rank = 3 - 2 = 1
# → 1차원 부분공간이 0으로 매핑됨 (정보 손실!)
```

---

## 기저 변환 (Change of Basis)

### 같은 벡터, 다른 좌표

같은 점도 **어떤 기저(좌표계)를 쓰느냐**에 따라 좌표가 달라집니다:

```python
# 표준 기저에서의 좌표
v_standard = torch.tensor([3.0, 1.0])

# 새로운 기저
b1 = torch.tensor([1.0, 1.0])   # 45도 방향
b2 = torch.tensor([-1.0, 1.0])  # 135도 방향

# 기저 변환 행렬 (새 기저 벡터를 열로)
P = torch.stack([b1, b2], dim=1)  # [[1,-1],[1,1]]

# 새 기저에서의 좌표
v_new = torch.linalg.solve(P, v_standard)
print(f"표준 기저: {v_standard}")
print(f"새 기저:   {v_new}")

# 검증: 새 좌표로 원래 벡터 복원
v_reconstructed = v_new[0] * b1 + v_new[1] * b2
print(f"복원:     {v_reconstructed}")
print(f"일치: {torch.allclose(v_standard, v_reconstructed)}")
```

### 딥러닝 적용: 임베딩 = 기저 변환

```python
# Embedding Layer는 기저 변환으로 볼 수 있다!
vocab_size, embed_dim = 10000, 768

# 단어 인덱스: 표준 기저의 좌표 (one-hot)
# 임베딩 행렬: 기저 변환 행렬
embedding = torch.nn.Embedding(vocab_size, embed_dim)

# one-hot [0, 0, ..., 1, ..., 0] → 768차원 임베딩 벡터
# = 표준 기저(10000차원, sparse) → 새 기저(768차원, dense)
word_idx = torch.tensor([42])
word_vec = embedding(word_idx)  # 768차원으로 기저 변환
print(f"임베딩: {word_vec.shape}")  # (1, 768)
```

---

## 직교 기저 (Orthogonal Basis)

### 정의

기저 벡터들이 모두 **서로 수직**:

$$
\mathbf{b}_i \cdot \mathbf{b}_j = 0 \quad (i \neq j)
$$

크기까지 1이면 **정규직교 기저 (Orthonormal Basis)**:

$$
\mathbf{b}_i \cdot \mathbf{b}_j = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}
$$

```python
# 정규직교 기저 확인
Q = torch.tensor([[1/2**0.5, -1/2**0.5],
                  [1/2**0.5,  1/2**0.5]])

# Q^T Q = I 이면 정규직교
print(Q.T @ Q)
# [[1, 0],
#  [0, 1]] ← 단위행렬!
```

### 왜 직교 기저가 좋은가?

1. **좌표 계산이 쉬움**: 내적만 하면 됨
2. **수치적 안정성**: 오차가 증폭되지 않음
3. **독립적**: 각 축이 서로 간섭하지 않음

```python
# 직교 기저에서의 좌표 = 그냥 내적!
Q = torch.tensor([[1/2**0.5, -1/2**0.5],
                  [1/2**0.5,  1/2**0.5]])
v = torch.tensor([3.0, 1.0])

# 비직교 기저: solve 필요 (비쌈)
# 직교 기저: 내적만 하면 됨! (쌈)
coords = Q.T @ v
print(f"좌표: {coords}")

# 복원
v_reconstructed = Q @ coords
print(f"복원: {v_reconstructed}")
```

### 딥러닝 적용

```python
# SVD가 주는 것 = 최적의 직교 기저
# A = U @ diag(S) @ V^T
# U의 열: 출력 공간의 직교 기저
# V의 열: 입력 공간의 직교 기저

A = torch.randn(5, 3)
U, S, Vt = torch.linalg.svd(A, full_matrices=False)

# U와 V는 직교 행렬
print(f"U^T U ≈ I: {torch.allclose(U.T @ U, torch.eye(3), atol=1e-5)}")
print(f"V V^T ≈ I: {torch.allclose(Vt @ Vt.T, torch.eye(3), atol=1e-5)}")
```

---

## 코드로 확인: 전체 파이프라인

```python
import torch
import numpy as np

print("=== 선형 독립 판별 ===")
# 세 벡터가 선형 독립인지 확인
V = torch.tensor([[1., 0., 2.],
                  [0., 1., 1.],
                  [1., 1., 3.]])  # 3번째 열 = 1번째 + 2번째

print(f"Rank: {torch.linalg.matrix_rank(V)}")  # 2 (독립인 벡터 2개)

print("\n=== 부분공간의 차원 ===")
# 데이터의 실질적 차원 확인
np.random.seed(0)
# 100차원이지만 5차원 부분공간에 존재하는 데이터
true_basis = np.random.randn(5, 100)
X = np.random.randn(200, 5) @ true_basis

U, S, Vt = np.linalg.svd(X, full_matrices=False)
# 상위 특이값만 의미있음
threshold = 0.01 * S[0]
effective_dim = (S > threshold).sum()
print(f"데이터 차원: {X.shape[1]}")
print(f"실질적 차원: {effective_dim}")  # ≈ 5
print(f"상위 10 특이값: {S[:10].round(2)}")

print("\n=== 기저 변환 ===")
# 표준 기저 → 고유벡터 기저
A = torch.tensor([[2., 1.],
                  [1., 2.]], dtype=torch.float32)

eigenvalues, eigenvectors = torch.linalg.eigh(A)
print(f"고유값: {eigenvalues}")
print(f"고유벡터(새 기저):\n{eigenvectors}")

# 원래 벡터를 고유벡터 기저로 표현
v = torch.tensor([3., 1.])
v_eigen_coords = eigenvectors.T @ v
print(f"표준 기저: {v}")
print(f"고유벡터 기저: {v_eigen_coords}")

# 복원
v_restored = eigenvectors @ v_eigen_coords
print(f"복원: {v_restored}")
print(f"일치: {torch.allclose(v, v_restored)}")

print("\n=== LoRA의 부분공간 시뮬레이션 ===")
# 큰 행렬의 변화가 저차원 부분공간에 집중되는지 확인
d = 256
W_pretrained = torch.randn(d, d)
W_finetuned = W_pretrained + 0.01 * torch.randn(d, d)  # 작은 변화

delta_W = W_finetuned - W_pretrained
U, S, Vt = torch.linalg.svd(delta_W)

total_energy = (S**2).sum()
cumsum = torch.cumsum(S**2, dim=0)
for r in [1, 4, 8, 16, 32]:
    explained = (cumsum[r-1] / total_energy * 100).item()
    print(f"rank {r:2d}: {explained:.1f}% 에너지 설명")
```

---

## 핵심 정리

| 개념 | 정의 | 딥러닝 적용 |
|------|------|------------|
| 선형 결합 | $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots$ | Linear Layer = 열벡터의 선형 결합 |
| Span | 모든 선형 결합의 집합 | 모델이 표현 가능한 함수의 범위 |
| 선형 독립 | 중복 없는 벡터 집합 | Feature 중복 감지, Rank |
| 기저 | 독립 + span = 좌표축 | Embedding 공간의 축 |
| 차원 | 기저 벡터의 수 | 임베딩 차원, latent 차원 |
| 부분공간 | 큰 공간 안의 작은 공간 | LoRA, PCA, 데이터 매니폴드 |

## 핵심 통찰

1. **차원 = 자유도**: 768차원이라도 데이터가 50차원 부분공간에 살면, 실질적 자유도는 50
2. **Rank = 실질적 차원**: 행렬의 rank는 출력이 사는 부분공간의 차원
3. **기저 = 좌표계**: 같은 데이터도 어떤 기저(PCA, 고유벡터, 표준)로 보느냐에 따라 다르게 표현
4. **LoRA가 작동하는 이유**: 파인튜닝의 변화는 저차원 부분공간에 집중

---

## 다음 단계

벡터 공간의 구조를 이해했습니다. 이제 **행렬이 공간을 어떻게 변환하는지** 체계적으로 알아봅니다.

→ [선형 변환](/ko/docs/math/linear-algebra/linear-transformations): 회전, 투영, 반사의 수학

## 관련 콘텐츠

- [벡터 기초](/ko/docs/math/linear-algebra/vector) - 벡터의 기본 연산
- [행렬 연산](/ko/docs/math/linear-algebra/matrix) - Rank, 공간 변환
- [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue) - PCA, 주성분 분석
- [SVD](/ko/docs/math/linear-algebra/svd) - 최적의 저랭크 근사
