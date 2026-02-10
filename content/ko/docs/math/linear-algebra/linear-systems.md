---
title: "연립방정식과 최소제곱"
weight: 5
math: true
---

# 연립방정식과 최소제곱 (Linear Systems & Least Squares)

{{% hint info %}}
**선수지식**: [행렬 연산](/ko/docs/math/linear-algebra/matrix) | [벡터 공간](/ko/docs/math/linear-algebra/vector-spaces)
{{% /hint %}}

> **한 줄 요약**: $A\mathbf{x} = \mathbf{b}$를 푸는 것이 선형대수의 핵심이며, 정확한 해가 없을 때 **최소제곱법**으로 "가장 가까운 해"를 찾습니다. 이것이 바로 **선형 회귀와 카메라 캘리브레이션의 수학적 원리**입니다.

## 왜 연립방정식을 배워야 하나요?

### 문제 상황 1: "카메라 캘리브레이션이 어떻게 되는 건가요?"

```python
# 3D 점 → 2D 이미지 좌표 매핑에서 카메라 행렬을 구하려면?
# 여러 점으로부터 Ax = b를 세우고... 어떻게 풀지?
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None
)
```

→ 내부적으로 **연립방정식의 최소제곱 해**를 구합니다.

### 문제 상황 2: "Homography는 어떻게 계산하나요?"

```python
# 4개 점의 대응 관계로 변환 행렬을 구하려면?
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
# H는 3×3 행렬... 어떻게 8개 미지수를 구할까?
```

→ 대응점들로 **연립방정식을 세우고** 해를 구합니다.

### 문제 상황 3: "선형 회귀가 왜 '닫힌 해'가 있나요?"

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
# 경사하강법 없이 바로 답이 나온다고?
```

→ **정규방정식** $(A^T A)\mathbf{x} = A^T \mathbf{b}$를 직접 풀기 때문입니다.

---

## 연립방정식 $A\mathbf{x} = \mathbf{b}$

### 기본 형태

여러 방정식을 한 번에 표현합니다:

$$
\begin{cases} 2x_1 + 3x_2 = 7 \\ x_1 - x_2 = 1 \end{cases} \quad\Longleftrightarrow\quad \underbrace{\begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}}_{\mathbf{x}} = \underbrace{\begin{bmatrix} 7 \\ 1 \end{bmatrix}}_{\mathbf{b}}
$$

**각 기호의 의미:**
- $A$ : **계수 행렬** — 시스템의 구조 (m×n)
- $\mathbf{x}$ : **미지수 벡터** — 우리가 찾고 싶은 것 (n×1)
- $\mathbf{b}$ : **관측 벡터** — 알고 있는 결과 (m×1)

```python
import torch

A = torch.tensor([[2.0, 3.0],
                  [1.0, -1.0]])
b = torch.tensor([7.0, 1.0])

# PyTorch로 풀기
x = torch.linalg.solve(A, b)
print(f"해: {x}")  # tensor([2., 1.]) → x1=2, x2=1
```

### 해의 존재와 유일성

연립방정식의 해는 세 가지 경우 중 하나입니다:

| 경우 | 조건 | 기하학적 의미 | 예시 |
|------|------|--------------|------|
| **유일한 해** | $\text{rank}(A) = n$, $\mathbf{b} \in \text{Col}(A)$ | 직선이 한 점에서 만남 | 정방행렬, 역행렬 존재 |
| **무한히 많은 해** | $\text{rank}(A) < n$, $\mathbf{b} \in \text{Col}(A)$ | 직선이 겹침 | 미지수가 방정식보다 많음 |
| **해 없음** | $\mathbf{b} \notin \text{Col}(A)$ | 직선이 평행 | 데이터에 노이즈가 있을 때 |

```python
# 경우 1: 유일한 해
A1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
b1 = torch.tensor([3.0, 4.0])
print(torch.linalg.solve(A1, b1))  # [3, 4]

# 경우 2: 무한히 많은 해 (rank 부족)
A2 = torch.tensor([[1.0, 2.0], [2.0, 4.0]])  # 두 번째 행 = 2 × 첫 번째 행
print(torch.linalg.matrix_rank(A2))  # rank = 1 < 2

# 경우 3: 해 없음 → 최소제곱으로 해결!
```

### 열 공간의 관점

$A\mathbf{x} = \mathbf{b}$를 다른 시각으로 봅시다:

$$
A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n = \mathbf{b}
$$

**"$A$의 열벡터들을 조합해서 $\mathbf{b}$를 만들 수 있는가?"**

→ $\mathbf{b}$가 $A$의 **열 공간(Column Space)**에 있으면 해가 존재합니다.

---

## 가우스 소거법: 기본 풀이 방법

### 핵심 아이디어

방정식을 **행 사다리꼴(Row Echelon Form)**로 변환하여 위에서부터 역대입합니다.

$$
\begin{bmatrix} 2 & 1 & | & 5 \\ 4 & 3 & | & 11 \end{bmatrix} \xrightarrow{R_2 - 2R_1} \begin{bmatrix} 2 & 1 & | & 5 \\ 0 & 1 & | & 1 \end{bmatrix}
$$

→ $x_2 = 1$, $2x_1 + 1 = 5$ → $x_1 = 2$

```python
# 가우스 소거법 구현 (교육용)
def gaussian_elimination(A, b):
    """Ax = b를 가우스 소거법으로 풀기"""
    n = A.shape[0]
    # 확장 행렬 [A | b]
    Ab = torch.cat([A.float(), b.float().unsqueeze(1)], dim=1)

    # 전진 소거 (Forward Elimination)
    for i in range(n):
        # 피봇이 0이면 행 교환
        max_row = torch.argmax(torch.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # 아래 행들에서 피봇 열 제거
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] -= factor * Ab[i]

    # 후진 대입 (Back Substitution)
    x = torch.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x

A = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
b = torch.tensor([5.0, 11.0])
print(gaussian_elimination(A, b))  # tensor([2., 1.])
```

### 딥러닝 적용: 왜 가우스 소거법을 직접 안 쓸까?

| 방법 | 시간 복잡도 | 메모리 | 사용 장면 |
|------|-----------|--------|----------|
| 가우스 소거법 | $O(n^3)$ | $O(n^2)$ | 작은 시스템 (캘리브레이션) |
| 경사하강법 | $O(n \cdot \text{iter})$ | $O(n)$ | 큰 시스템 (딥러닝) |
| 분해 기반 (LU, QR) | $O(n^3)$ | $O(n^2)$ | 중간 크기 (통계 모델) |

→ 딥러닝에서는 파라미터가 수백만 개라서 **경사하강법**이 필수!

{{< figure src="/images/math/linear-algebra/ko/least-squares.png" caption="최소제곱법: b를 열 공간(Col(A))에 투영하여 가장 가까운 해를 찾는다" >}}

---

## 최소제곱법 (Least Squares): 해가 없을 때

### 현실의 문제: 데이터에는 항상 노이즈가 있다

카메라 캘리브레이션에서 100개의 대응점이 있으면:
- 방정식: 200개 (각 점에서 x, y 2개)
- 미지수: 11개 (카메라 파라미터)
- **방정식이 미지수보다 훨씬 많다** → 정확한 해가 존재하지 않음!

### 핵심 아이디어: 가장 가까운 해 찾기

정확히 $A\mathbf{x} = \mathbf{b}$를 만족하는 $\mathbf{x}$가 없으면, **오차를 최소화**하는 $\mathbf{x}$를 찾습니다:

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|^2
$$

**각 기호의 의미:**
- $\hat{\mathbf{x}}$ : 최소제곱 해 — "최선의 근사"
- $\|A\mathbf{x} - \mathbf{b}\|^2$ : **잔차(residual)**의 제곱합 — 오차 크기
- $\arg\min$ : 이 오차를 최소화하는 $\mathbf{x}$

### 직관적 이해: 투영

$\mathbf{b}$가 $A$의 열 공간 밖에 있으면, **열 공간 위로 투영한 점**이 가장 가까운 해입니다.

$$
\hat{\mathbf{b}} = A\hat{\mathbf{x}} = \text{proj}_{\text{Col}(A)} \mathbf{b}
$$

→ "완벽한 해가 없으면, 열 공간에서 $\mathbf{b}$에 가장 가까운 점을 찾는다"

### 정규방정식 (Normal Equation)

최소제곱 해는 다음 공식으로 구합니다:

$$
A^T A \hat{\mathbf{x}} = A^T \mathbf{b}
$$

$$
\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}
$$

**왜 이 공식이 나오나?** (미분으로 유도)

오차 함수 $f(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|^2$를 $\mathbf{x}$에 대해 미분하고 0으로 놓으면:

$$
\nabla_\mathbf{x} f = 2A^T(A\mathbf{x} - \mathbf{b}) = 0 \quad\Rightarrow\quad A^T A \mathbf{x} = A^T \mathbf{b}
$$

```python
import torch

# 노이즈가 있는 데이터: y = 2x + 1 + noise
torch.manual_seed(42)
x_data = torch.linspace(0, 10, 50)
y_data = 2 * x_data + 1 + torch.randn(50) * 0.5

# A 행렬 구성: [x, 1] (각 행)
A = torch.stack([x_data, torch.ones_like(x_data)], dim=1)  # (50, 2)
b = y_data  # (50,)

# 정규방정식으로 풀기: (A^T A)^{-1} A^T b
x_hat = torch.linalg.solve(A.T @ A, A.T @ b)
print(f"기울기: {x_hat[0]:.3f}, 절편: {x_hat[1]:.3f}")  # ≈ 2.0, ≈ 1.0

# PyTorch의 lstsq 사용
result = torch.linalg.lstsq(A, b)
print(f"lstsq 결과: 기울기={result.solution[0]:.3f}, 절편={result.solution[1]:.3f}")
```

### 딥러닝 적용: 선형 회귀의 닫힌 해

```python
# scikit-learn의 LinearRegression은 내부적으로 이것을 수행
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# 내부: theta = (X^T X)^{-1} X^T y  ← 정규방정식!
```

---

## 의사역행렬 (Pseudo-Inverse)

### 역행렬이 없을 때는?

정방행렬이 아니거나, 특이(singular)행렬이면 $A^{-1}$이 존재하지 않습니다.

**무어-펜로즈 의사역행렬** $A^+$는 모든 행렬에 대해 정의됩니다:

$$
A^+ = (A^T A)^{-1} A^T \quad \text{(열이 독립일 때)}
$$

일반적으로는 SVD를 이용해 계산합니다:

$$
A = U \Sigma V^T \quad\Rightarrow\quad A^+ = V \Sigma^+ U^T
$$

여기서 $\Sigma^+$는 $\Sigma$의 0이 아닌 특이값의 역수를 취한 것입니다.

```python
# 의사역행렬 계산
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])  # 3×2 행렬 (역행렬 불가)

A_pinv = torch.linalg.pinv(A)
print(f"A shape: {A.shape}")        # (3, 2)
print(f"A+ shape: {A_pinv.shape}")  # (2, 3)

# 최소제곱 해: x = A+ @ b
b = torch.tensor([1.0, 2.0, 3.0])
x_hat = A_pinv @ b
print(f"최소제곱 해: {x_hat}")

# 검증: A @ x_hat ≈ b (완벽하지 않지만 가장 가까움)
print(f"A @ x_hat = {A @ x_hat}")  # b에 가장 가까운 값
print(f"잔차: {torch.norm(A @ x_hat - b):.4f}")
```

### 과결정 vs 미결정 시스템

| 시스템 | 조건 | 해의 특성 | 의사역행렬의 역할 |
|--------|------|----------|------------------|
| **과결정** | $m > n$ (방정식 > 미지수) | 해 없음 → 최소제곱 해 | 잔차 최소화 |
| **정결정** | $m = n$ | 유일한 해 | 일반 역행렬과 동일 |
| **미결정** | $m < n$ (방정식 < 미지수) | 무한한 해 → 최소 노름 해 | 가장 작은 $\|\mathbf{x}\|$ 선택 |

```python
# 과결정: 100개 관측 → 3개 파라미터
A_over = torch.randn(100, 3)   # 방정식이 더 많음
b_over = torch.randn(100)
x_over = torch.linalg.lstsq(A_over, b_over).solution  # 최소제곱 해

# 미결정: 3개 관측 → 100개 파라미터
A_under = torch.randn(3, 100)  # 미지수가 더 많음
b_under = torch.randn(3)
x_under = torch.linalg.pinv(A_under) @ b_under  # 최소 노름 해
print(f"미결정 해의 노름: {torch.norm(x_under):.4f}")  # 가능한 해 중 가장 작음
```

---

## 비전에서의 핵심 응용

### 1. 호모그래피 추정

4개 이상의 대응점으로 원근 변환 행렬을 구합니다:

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_1 & h_2 & h_3 \\ h_4 & h_5 & h_6 \\ h_7 & h_8 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

각 대응점이 2개의 방정식을 만들고, 미지수는 8개:
- 4개 점 → 8개 방정식 → 정확한 해
- 5개 이상 → **과결정** → 최소제곱 해

```python
import cv2
import numpy as np

# 대응점 (노이즈 포함 가능)
src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100],
                  [50, 0], [100, 50]])  # 6개 점 (과결정)
dst = np.float32([[10, 10], [90, 5], [95, 95], [5, 90],
                  [52, 8], [92, 48]])

# 내부적으로 최소제곱 해를 구함
H, mask = cv2.findHomography(src, dst, method=0)
print(f"Homography:\n{H}")
```

### 2. 카메라 캘리브레이션 (DLT)

**DLT (Direct Linear Transform)**는 카메라 행렬을 구하는 대표적 방법입니다:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim P \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

$P$는 3×4 투영 행렬 (11개 미지수), 6개 이상의 3D-2D 대응점이 필요합니다.

```python
# 카메라 캘리브레이션 = 큰 연립방정식 풀기
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points,  # 3D 점들
    image_points,   # 대응하는 2D 점들
    image_size,
    None, None
)
# K: 카메라 내부 파라미터 행렬 (정규방정식으로 추정)
```

### 3. 선형 회귀 → 신경망의 기초

```python
import torch
import torch.nn as nn

# 선형 회귀: 닫힌 해
X = torch.randn(1000, 10)  # 1000개 샘플, 10개 특징
w_true = torch.randn(10)
y = X @ w_true + torch.randn(1000) * 0.1

# 방법 1: 정규방정식 (작은 데이터에 적합)
w_closed = torch.linalg.solve(X.T @ X, X.T @ y)
print(f"정규방정식 오차: {torch.norm(w_closed - w_true):.4f}")

# 방법 2: 경사하강법 (큰 데이터에 적합)
w_gd = torch.zeros(10, requires_grad=True)
optimizer = torch.optim.SGD([w_gd], lr=0.01)

for step in range(200):
    loss = ((X @ w_gd - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"경사하강법 오차: {torch.norm(w_gd.detach() - w_true):.4f}")
# 두 방법 모두 거의 같은 답에 도달!
```

---

## 수치적 안정성: 실전에서 중요한 것

### 조건수 (Condition Number)

$A^T A$의 **조건수**가 크면 해가 불안정합니다:

$$
\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

- $\kappa \approx 1$: 안정적
- $\kappa \gg 1$: 작은 노이즈에도 해가 크게 변함

```python
# 조건수가 나쁜 행렬
A_bad = torch.tensor([[1.0, 1.0],
                      [1.0, 1.0001]])
print(f"조건수: {torch.linalg.cond(A_bad):.0f}")  # 매우 큰 값!

# 해결: 정규화 (Regularization)
# (A^T A + λI)x = A^T b  ← Ridge Regression
lambda_reg = 0.01
A = torch.randn(100, 10)
b = torch.randn(100)
x_ridge = torch.linalg.solve(
    A.T @ A + lambda_reg * torch.eye(10),
    A.T @ b
)
```

### 정규화와 최소제곱의 관계

**Ridge Regression** = 정규화된 최소제곱:

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \left( \|A\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|^2 \right)
$$

$$
\hat{\mathbf{x}} = (A^T A + \lambda I)^{-1} A^T \mathbf{b}
$$

→ $\lambda$가 조건수를 개선하여 안정적인 해를 만듭니다.

```python
# Weight Decay = Ridge Regression의 경사하강법 버전!
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
# weight_decay가 바로 λ 역할
```

---

## 핵심 정리

| 개념 | 수식 | 의미 | 딥러닝/비전 적용 |
|------|------|------|-----------------|
| 연립방정식 | $A\mathbf{x} = \mathbf{b}$ | 정확한 해 찾기 | 기본 선형 시스템 |
| 최소제곱 | $\min\|A\mathbf{x}-\mathbf{b}\|^2$ | 가장 가까운 해 | 선형 회귀, 캘리브레이션 |
| 정규방정식 | $(A^TA)\mathbf{x}=A^T\mathbf{b}$ | 최소제곱의 닫힌 해 | LinearRegression |
| 의사역행렬 | $A^+ = (A^TA)^{-1}A^T$ | 일반화된 역행렬 | SVD 기반 풀이 |
| 정규화 | $(A^TA + \lambda I)\mathbf{x}=A^T\mathbf{b}$ | 안정적인 해 | Weight Decay |

## 핵심 통찰

1. **$A\mathbf{x} = \mathbf{b}$는 어디에나**: 카메라 캘리브레이션, 호모그래피, 선형 회귀 모두 같은 문제
2. **해가 없으면 최소제곱**: 현실 데이터에는 노이즈 → "가장 가까운 해"가 실용적
3. **정규화는 안정성**: 조건수가 나쁘면 $\lambda I$를 더해서 해결 → Weight Decay의 원리
4. **크기에 따라 방법 선택**: 작으면 정규방정식, 크면 경사하강법

---

## 다음 단계

행렬의 풀이 방법을 이해했으니, 이제 행렬을 **분해**하여 더 깊은 성질을 알아봅시다.

→ [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue): 행렬의 본질적 성질

## 관련 콘텐츠

- [행렬 연산](/ko/docs/math/linear-algebra/matrix) — $A$의 기본 연산
- [벡터 공간](/ko/docs/math/linear-algebra/vector-spaces) — 열 공간, 영 공간
- [고유값/고유벡터](/ko/docs/math/linear-algebra/eigenvalue) — 행렬 분해의 시작
- [SVD](/ko/docs/math/linear-algebra/svd) — 의사역행렬의 실제 계산
