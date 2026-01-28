---
title: "미적분"
weight: 2
bookCollapseSection: true
math: true
---

# 미적분 (Calculus)

> **한 줄 요약**: 미적분은 딥러닝이 "어떻게 학습하는가"의 수학적 핵심입니다.

## 왜 미적분을 배워야 하나요?

딥러닝 코드를 처음 접하면 이런 질문이 생깁니다:

### 질문 1: "loss.backward()가 뭘 하는 거죠?"

```python
loss = criterion(output, target)
loss.backward()  # ← 이게 정확히 뭘 하는 거지?
optimizer.step()
```

→ `backward()`는 **미분을 자동으로 계산**합니다. 미분이 뭔지 모르면 딥러닝 학습을 이해할 수 없습니다.

### 질문 2: "왜 Learning Rate가 중요한가요?"

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lr = ?
```

→ Learning Rate는 **기울기(gradient) 방향으로 얼마나 이동할지**를 결정합니다. 기울기가 뭔지 모르면 학습률 튜닝도 감으로만 합니다.

### 질문 3: "왜 깊은 네트워크가 학습이 안 되나요?"

```python
# 100층 네트워크 - 왜 학습이 안 될까?
model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(100)])
```

→ **Vanishing Gradient** 문제입니다. Chain Rule이 반복 적용되면서 gradient가 사라집니다.

---

## 미적분이 딥러닝에서 하는 역할

딥러닝 학습의 본질:

```
1. 예측 계산: ŷ = f(x; θ)
2. 손실 계산: L = Loss(ŷ, y)
3. ★ 기울기 계산: ∂L/∂θ  ← 미분!
4. 파라미터 업데이트: θ ← θ - η · ∂L/∂θ
```

**3번 단계가 미적분입니다.** 이것 없이는 모델이 학습할 수 없습니다.

### 비유: 눈을 감고 산을 내려가기

{{< mermaid >}}
graph LR
    A[현재 위치] -->|기울기 측정| B[가장 가파른 방향]
    B -->|반대로 이동| C[더 낮은 곳]
    C -->|반복| D[가장 낮은 곳]
{{< /mermaid >}}

- 눈을 감고 산 정상에 서 있다고 상상해보세요
- 발로 주변 기울기를 느껴봅니다 (= gradient 계산)
- 가장 가파른 **내리막** 방향으로 한 걸음 이동합니다 (= gradient descent)
- 이걸 반복하면 결국 골짜기(최저점)에 도달합니다

**미분**은 "이 방향으로 가면 얼마나 올라가는가/내려가는가?"를 알려주는 도구입니다.

---

## 학습 로드맵

이 섹션에서는 대학 미적분을 몰라도 딥러닝에 필요한 미적분을 배울 수 있습니다.

| 순서 | 개념 | 핵심 질문 | 딥러닝 적용 |
|:----:|------|----------|------------|
| 1 | [미분 기초](/ko/docs/math/calculus/basics) | 변화율이 뭔가요? | 손실 함수의 민감도 |
| 2 | [Gradient](/ko/docs/math/calculus/gradient) | 여러 변수가 있으면? | 파라미터 업데이트 방향 |
| 2+ | ↳ Jacobian, Hessian | 벡터 함수의 미분은? | 레이어 변환, 곡률 |
| 3 | [Chain Rule](/ko/docs/math/calculus/chain-rule) | 함수가 연결되면? | 깊은 네트워크의 기울기 |
| 4 | [Backpropagation](/ko/docs/math/calculus/backpropagation) | 효율적으로 계산하려면? | loss.backward()의 원리 |
| 5 | [최적화 수학](/ko/docs/math/calculus/optimization) | GD가 왜 작동하나요? | Taylor 전개, Adam의 원리 |

---

## 미적분 없이 딥러닝을 할 수 있을까?

**사용만 하려면**: 가능합니다. PyTorch가 자동으로 미분을 계산해줍니다.

**제대로 이해하려면**: 불가능합니다.

| 상황 | 미적분 없이 | 미적분으로 |
|------|-----------|-----------|
| 학습이 안 될 때 | "Learning rate 바꿔볼까?" | "Gradient가 왜 0인지 분석" |
| 새로운 Loss 설계 | 남이 만든 것만 사용 | 직접 설계하고 미분 유도 |
| 논문 읽기 | 수식 스킵 | 왜 이렇게 설계했는지 이해 |
| 커스텀 Layer 구현 | 불가능 | backward 함수 직접 구현 |

---

## 핵심 개념 미리보기

### 1. 미분 = 순간 변화율

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

**직관**: x가 아주 조금 변할 때, f(x)가 얼마나 변하는가?

### 2. Gradient = 각 방향의 미분을 모은 벡터

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots \right)
$$

**직관**: 여러 변수가 있을 때, 각각이 결과에 얼마나 영향을 주는가?

### 2+. Jacobian & Hessian = 고차원 미분

- **Jacobian**: 벡터→벡터 함수의 미분 (m×n 행렬)
- **Hessian**: 2차 미분, 곡률 정보 (n×n 행렬)

**직관**: 레이어가 입력을 어떻게 변형하는가? 학습이 쉬운가 어려운가?

### 3. Chain Rule = 연결된 함수의 미분

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

**직관**: A가 B에 영향을 주고, B가 C에 영향을 주면, A가 C에 주는 영향은?

### 4. Backpropagation = Chain Rule의 효율적 적용

**직관**: 수백만 파라미터의 gradient를 한 번에 계산하는 알고리즘

### 5. Taylor 전개 = 함수 근사

$$
f(x+\delta) \approx f(x) + f'(x)\delta + \frac{1}{2}f''(x)\delta^2
$$

**직관**: Gradient Descent가 왜 작동하는가? Learning Rate의 상한은?

---

## 관련 콘텐츠

- [최적화 수학](/ko/docs/math/calculus/optimization) - Taylor 전개, 2차 최적화
- [SGD](/ko/docs/math/training/optimizer/sgd) - Gradient를 활용한 최적화
- [Adam](/ko/docs/math/training/optimizer/adam) - 적응적 학습률 (Hessian 근사)
- [Loss Functions](/ko/docs/math/training/loss) - 미분 가능한 목적 함수
- [BatchNorm](/ko/docs/math/normalization/batch-norm) - Gradient 흐름 개선
