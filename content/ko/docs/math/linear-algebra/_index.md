---
title: "선형대수"
weight: 1
bookCollapseSection: true
math: true
---

# 선형대수 (Linear Algebra)

> **한 줄 요약**: 딥러닝의 모든 연산은 행렬 곱셈입니다. 선형대수 없이는 코드 한 줄도 이해할 수 없습니다.

## 왜 선형대수를 배워야 하나요?

### 질문 1: "왜 이미지가 (B, C, H, W) 형태인가요?"

```python
# ImageNet 배치
images = torch.randn(32, 3, 224, 224)  # ← 이 숫자들이 뭘 의미하지?
```

→ 이미지는 **3차원 텐서**, 배치는 **4차원 텐서**입니다. 텐서 = 다차원 배열 = 선형대수!

### 질문 2: "nn.Linear(784, 256)이 뭘 하는 건가요?"

```python
layer = nn.Linear(784, 256)
output = layer(input)  # ← 내부에서 뭐가 일어나지?
```

→ 내부적으로 **행렬 곱셈**: $y = Wx + b$ (W는 256×784 행렬)

### 질문 3: "왜 shape이 안 맞으면 에러가 나나요?"

```python
A = torch.randn(32, 784)
B = torch.randn(256, 784)
C = A @ B  # RuntimeError: size mismatch!
```

→ 행렬 곱셈의 **차원 규칙**을 모르면 항상 에러와 싸웁니다.

### 질문 4: "LoRA가 뭔가요? 왜 파라미터가 적은가요?"

```python
# 원본: 4096 × 4096 = 16M 파라미터
# LoRA: 4096 × 8 + 8 × 4096 = 65K 파라미터 (99.6% 감소!)
```

→ **저랭크 분해(SVD)**를 이해해야 LoRA가 왜 작동하는지 알 수 있습니다.

---

## 선형대수가 딥러닝에서 하는 역할

### 모든 것은 텐서다

```python
# 이미지 = 텐서
image = torch.randn(3, 224, 224)      # C × H × W

# 텍스트 = 텐서
embeddings = torch.randn(512, 768)    # Seq × Dim

# 모델 = 텐서들의 모음
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")   # 전부 텐서!
```

### 모든 연산은 행렬 곱셈이다

| 레이어 | 연산 | 행렬 표현 |
|--------|------|----------|
| Linear | FC Layer | $y = Wx + b$ |
| Conv2d | 합성곱 | im2col + 행렬 곱 |
| Attention | QKV | $\text{softmax}(QK^T)V$ |
| BatchNorm | 정규화 | 요소별 스케일링 |

---

## 학습 로드맵

| 순서 | 개념 | 핵심 질문 | 딥러닝 적용 |
|:----:|------|----------|------------|
| 1 | [벡터 기초](/ko/docs/math/linear-algebra/vector) | 벡터가 뭔가요? | Embedding, 특징 표현 |
| 2 | [행렬 연산](/ko/docs/math/linear-algebra/matrix) | 행렬 곱, Rank, 행렬식? | Linear Layer, LoRA, Flow |
| 3 | [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue) | 행렬의 본질은? | PCA, 학습 안정성 |
| 4 | [SVD](/ko/docs/math/linear-algebra/svd) | 행렬을 분해하면? | 압축, LoRA |

---

## 핵심 개념 미리보기

### 1. 벡터 = 숫자들의 목록

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
$$

**딥러닝에서**: 하나의 데이터 포인트 = 하나의 벡터

```python
# 단어 하나 = 768차원 벡터
word_embedding = torch.randn(768)
```

### 2. 행렬 = 벡터들의 모음 = 변환

$$
W = \begin{bmatrix} w_{11} & w_{12} & \cdots \\ w_{21} & w_{22} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

**딥러닝에서**: 레이어 = 행렬 = 공간 변환

```python
# Linear Layer = 행렬 곱
linear = nn.Linear(784, 256)  # 784차원 → 256차원 변환
```

### 3. 행렬 곱 = 차원 변환

$$
\mathbf{y} = W\mathbf{x}
$$

**핵심 규칙**: $(m \times k) \cdot (k \times n) = (m \times n)$

```python
W = torch.randn(256, 784)  # (256, 784)
x = torch.randn(784)       # (784,)
y = W @ x                  # (256,) ← 차원이 변했다!
```

### 3+. Rank = 행렬의 "실제 정보량"

$$
\text{rank}(A) \leq \min(m, n)
$$

**딥러닝에서**: LoRA가 작동하는 이유 - 파인튜닝은 저랭크로 충분!

### 3++. 행렬식 = 부피 변화율

$$
\det(A) = \text{공간이 얼마나 늘어나거나 줄어드는가}
$$

**딥러닝에서**: Flow 모델의 확률 변환에 필수

### 4. 고유값 = 행렬의 "성격"

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

**딥러닝에서**: 학습 안정성, 주성분 분석

### 5. SVD = 행렬의 "해부"

$$
A = U\Sigma V^T
$$

**딥러닝에서**: 압축, LoRA, 차원 축소

---

## 선형대수 없이 할 수 있는 것 vs 없는 것

| 상황 | 선형대수 없이 | 선형대수로 |
|------|-------------|-----------|
| 에러 디버깅 | shape 맞출 때까지 trial & error | 차원 규칙으로 즉시 해결 |
| 모델 이해 | 블랙박스로 사용 | 내부 동작 완전 이해 |
| 논문 읽기 | 수식 스킵 | QKV Attention 등 이해 |
| 최적화 | 남이 만든 것만 사용 | LoRA 등 직접 구현 |
| 커스텀 레이어 | 불가능 | 자유롭게 설계 |

---

## 관련 콘텐츠

- [Convolution](/ko/docs/components/convolution) - 특수한 행렬 연산
- [Attention](/ko/docs/components/attention) - QKV 행렬 연산
- [LoRA](/ko/docs/components/training/peft/lora) - SVD 기반 효율적 학습
- [BatchNorm](/ko/docs/components/normalization/batch-norm) - 통계 기반 정규화
