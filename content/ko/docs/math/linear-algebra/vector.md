---
title: "벡터 기초"
weight: 1
math: true
---

# 벡터 기초 (Vector Basics)

> **한 줄 요약**: 벡터는 숫자들의 목록이며, 딥러닝에서 **모든 데이터는 벡터**로 표현됩니다.

## 왜 벡터를 배워야 하나요?

### 문제 상황 1: "Embedding이 뭔가요?"

```python
# BERT의 단어 임베딩
word = "apple"
embedding = model.embed(word)
print(embedding.shape)  # torch.Size([768])
```

→ 단어 "apple"이 **768개의 숫자 목록(벡터)**으로 변환되었습니다!

### 문제 상황 2: "유사도를 어떻게 계산하나요?"

```python
# 두 문장의 유사도
sim = cosine_similarity(sentence1_vec, sentence2_vec)
```

→ **벡터 연산(내적)**으로 유사도를 계산합니다.

### 문제 상황 3: "왜 이미지가 (3, 224, 224)인가요?"

```python
image = torchvision.io.read_image("cat.jpg")
print(image.shape)  # torch.Size([3, 224, 224])
```

→ 이미지도 결국 **큰 벡터(텐서)**입니다!

---

## 벡터란 무엇인가?

### 가장 간단한 정의: 숫자들의 목록

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = [x_1, x_2, \ldots, x_n]
$$

```python
import torch

# 3차원 벡터
x = torch.tensor([1.0, 2.0, 3.0])
print(f"벡터: {x}")
print(f"차원: {x.shape}")  # torch.Size([3])
```

### 딥러닝에서 벡터의 의미

**벡터 = 특징(feature)의 모음**

| 대상 | 벡터 표현 | 차원 |
|------|----------|------|
| 단어 | Word Embedding | 300~768 |
| 이미지 | 피처 벡터 | 512~2048 |
| 문장 | Sentence Embedding | 768~1024 |
| 사용자 | User Embedding | 64~256 |

```python
# 단어 → 벡터
word_vec = torch.randn(768)  # "apple" = 768차원 벡터

# 이미지 → 벡터
image_vec = torch.randn(2048)  # CNN 특징

# 문장 → 벡터
sentence_vec = torch.randn(768)  # BERT 출력
```

---

## 벡터의 기본 연산

### 1. 덧셈과 뺄셈

$$
\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \end{bmatrix}
$$

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"덧셈: {x + y}")  # [5, 7, 9]
print(f"뺄셈: {x - y}")  # [-3, -3, -3]
```

**딥러닝 적용**: Residual Connection!

```python
# ResNet의 Skip Connection
output = F.relu(conv(x) + x)  # 벡터 덧셈!
```

### 2. 스칼라 곱 (상수 곱)

$$
c \cdot \mathbf{x} = \begin{bmatrix} c \cdot x_1 \\ c \cdot x_2 \\ \vdots \end{bmatrix}
$$

```python
x = torch.tensor([1.0, 2.0, 3.0])
c = 2.0

print(f"스칼라 곱: {c * x}")  # [2, 4, 6]
```

**딥러닝 적용**: Learning Rate 적용!

```python
# Gradient Descent
weights = weights - learning_rate * gradients  # 스칼라 × 벡터
```

### 3. 요소별 곱 (Hadamard Product)

$$
\mathbf{x} \odot \mathbf{y} = \begin{bmatrix} x_1 \cdot y_1 \\ x_2 \cdot y_2 \\ \vdots \end{bmatrix}
$$

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"요소별 곱: {x * y}")  # [4, 10, 18]
```

**딥러닝 적용**: Gating 메커니즘!

```python
# LSTM의 Gate
output = forget_gate * cell_state  # 요소별 곱으로 정보 선택
```

---

## 내적 (Dot Product): 가장 중요한 연산

### 정의

$$
\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{n} x_i y_i = x_1 y_1 + x_2 y_2 + \ldots + x_n y_n
$$

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 내적 = 스칼라 (숫자 하나!)
dot_product = torch.dot(x, y)
print(f"내적: {dot_product}")  # 32.0 = 1*4 + 2*5 + 3*6
```

### 내적의 기하학적 의미

$$
\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta
$$

- **θ = 0°**: 같은 방향 → 내적 최대 (양수)
- **θ = 90°**: 수직 → 내적 = 0
- **θ = 180°**: 반대 방향 → 내적 최소 (음수)

```python
# 같은 방향
a = torch.tensor([1.0, 0.0])
b = torch.tensor([2.0, 0.0])
print(f"같은 방향: {torch.dot(a, b)}")  # 2.0 (양수)

# 수직
a = torch.tensor([1.0, 0.0])
b = torch.tensor([0.0, 1.0])
print(f"수직: {torch.dot(a, b)}")  # 0.0

# 반대 방향
a = torch.tensor([1.0, 0.0])
b = torch.tensor([-1.0, 0.0])
print(f"반대 방향: {torch.dot(a, b)}")  # -1.0 (음수)
```

{{< figure src="/images/math/linear-algebra/ko/dot-product-geometry.jpeg" caption="내적의 기하학적 의미: 같은 방향(양수), 수직(0), 반대 방향(음수)" >}}

### 딥러닝에서의 내적

**1. Attention Score**

$$
\text{score} = \mathbf{q} \cdot \mathbf{k}
$$

```python
# Query와 Key의 유사도 = 내적
query = torch.randn(64)   # Query 벡터
key = torch.randn(64)     # Key 벡터
attention_score = torch.dot(query, key)
```

**2. 유사도 계산**

```python
# 두 임베딩의 유사도
def similarity(vec1, vec2):
    return torch.dot(vec1, vec2)

# 더 유사할수록 내적이 큼
```

**3. Linear Layer의 본질**

```python
# Linear Layer = 여러 벡터와의 내적
W = torch.randn(256, 784)  # 256개의 784차원 벡터
x = torch.randn(784)

# 각 출력 = W의 한 행과 x의 내적
y = W @ x  # y[i] = W[i] · x
```

---

## 벡터의 크기 (Norm)

### L2 Norm (유클리드 거리)

$$
\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \ldots + x_n^2}
$$

```python
x = torch.tensor([3.0, 4.0])
norm = torch.norm(x)
print(f"L2 norm: {norm}")  # 5.0 = sqrt(9 + 16)
```

### L1 Norm (맨해튼 거리)

$$
\|\mathbf{x}\|_1 = |x_1| + |x_2| + \ldots + |x_n|
$$

```python
x = torch.tensor([3.0, -4.0])
l1_norm = torch.norm(x, p=1)
print(f"L1 norm: {l1_norm}")  # 7.0 = 3 + 4
```

### 딥러닝 적용: 정규화

**L2 정규화 (Weight Decay)**

```python
# Loss에 가중치 크기 페널티 추가
loss = ce_loss + lambda_ * torch.norm(weights, p=2)**2
```

**L1 정규화 (Sparsity)**

```python
# Sparse 해를 유도
loss = ce_loss + lambda_ * torch.norm(weights, p=1)
```

---

## 정규화된 벡터 (Unit Vector)

### 정의: 크기가 1인 벡터

$$
\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}
$$

```python
x = torch.tensor([3.0, 4.0])
x_normalized = x / torch.norm(x)
print(f"정규화: {x_normalized}")      # [0.6, 0.8]
print(f"크기: {torch.norm(x_normalized)}")  # 1.0
```

### Cosine Similarity

$$
\cos\theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \hat{\mathbf{x}} \cdot \hat{\mathbf{y}}
$$

```python
def cosine_similarity(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

# 또는 PyTorch 함수 사용
sim = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
```

### 딥러닝 적용

**1. Layer Normalization**

```python
# 벡터를 정규화하여 학습 안정화
x_norm = (x - x.mean()) / x.std()
```

**2. Embedding 정규화**

```python
# 임베딩 벡터 정규화 (검색에서 중요)
embeddings = F.normalize(embeddings, dim=-1)
```

**3. Contrastive Learning**

```python
# 정규화된 벡터로 유사도 계산
z_i = F.normalize(encoder(x_i), dim=1)
z_j = F.normalize(encoder(x_j), dim=1)
similarity = torch.mm(z_i, z_j.T)  # Cosine similarity matrix
```

---

## 벡터 공간과 차원

### 벡터 공간이란?

**n차원 벡터 공간**: n개의 숫자로 이루어진 모든 벡터의 집합

```python
# 2차원 공간: 평면
v_2d = torch.tensor([x, y])

# 3차원 공간: 공간
v_3d = torch.tensor([x, y, z])

# 768차원 공간: BERT 임베딩 공간
v_768d = torch.randn(768)
```

### 고차원 공간의 직관

**차원의 저주**: 차원이 높아질수록...
- 대부분의 공간이 "비어있음"
- 거리 개념이 희미해짐
- 하지만 ML에서는 오히려 유용!

```python
# 고차원에서 랜덤 벡터들은 거의 직교
v1 = torch.randn(1000)
v2 = torch.randn(1000)
cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
print(f"랜덤 벡터 유사도: {cos_sim.item():.4f}")  # 거의 0에 가까움
```

---

## 코드로 확인: 임베딩 공간 탐색

```python
import torch
import torch.nn.functional as F

# 가상의 단어 임베딩
embeddings = {
    "king": torch.tensor([0.8, 0.2, 0.9, 0.1]),
    "queen": torch.tensor([0.7, 0.8, 0.9, 0.1]),
    "man": torch.tensor([0.9, 0.1, 0.1, 0.1]),
    "woman": torch.tensor([0.8, 0.9, 0.1, 0.1]),
}

def similarity(word1, word2):
    v1, v2 = embeddings[word1], embeddings[word2]
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

print("=== 유사도 측정 ===")
print(f"king-queen: {similarity('king', 'queen'):.3f}")
print(f"king-man: {similarity('king', 'man'):.3f}")
print(f"man-woman: {similarity('man', 'woman'):.3f}")

# 유명한 벡터 연산: king - man + woman ≈ queen
result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
result_norm = F.normalize(result.unsqueeze(0), dim=1).squeeze()

print("\n=== king - man + woman = ? ===")
for word, vec in embeddings.items():
    vec_norm = F.normalize(vec.unsqueeze(0), dim=1).squeeze()
    sim = F.cosine_similarity(result_norm.unsqueeze(0), vec_norm.unsqueeze(0))
    print(f"{word}: {sim.item():.3f}")
```

---

## 핵심 정리

| 연산 | 수식 | 결과 | 딥러닝 적용 |
|------|------|------|------------|
| 덧셈 | $\mathbf{x} + \mathbf{y}$ | 벡터 | Residual Connection |
| 스칼라 곱 | $c \cdot \mathbf{x}$ | 벡터 | Learning Rate |
| 요소별 곱 | $\mathbf{x} \odot \mathbf{y}$ | 벡터 | Gating |
| 내적 | $\mathbf{x} \cdot \mathbf{y}$ | 스칼라 | Attention, 유사도 |
| Norm | $\|\mathbf{x}\|$ | 스칼라 | 정규화 |
| 정규화 | $\mathbf{x}/\|\mathbf{x}\|$ | 단위 벡터 | Cosine Similarity |

## 핵심 통찰

1. **벡터 = 특징의 모음**: 단어, 이미지, 문장 모두 벡터로 표현
2. **내적 = 유사도**: 같은 방향이면 크고, 수직이면 0
3. **정규화 = 방향만 비교**: 크기 무시하고 방향만 보려면 정규화
4. **고차원도 OK**: 768차원도 잘 작동함 (오히려 표현력 증가)

---

## 다음 단계

벡터 하나를 이해했습니다. 이제 **여러 벡터를 모으면** 행렬이 됩니다.

→ [행렬 연산](/ko/docs/math/linear-algebra/matrix): 벡터들의 집합과 변환

## 관련 콘텐츠

- [행렬 연산](/ko/docs/math/linear-algebra/matrix) - 벡터의 확장
- [Attention](/ko/docs/components/attention) - 내적 기반 메커니즘
- [Embedding](/ko/docs/architecture) - 벡터 표현 학습
