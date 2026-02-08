---
title: "학습"
weight: 6
bookCollapseSection: true
math: true
---

# 학습 (Training)

딥러닝 모델 학습에 필요한 핵심 개념들입니다.

## 핵심 개념

| 카테고리 | 설명 |
|----------|------|
| [손실 함수](/ko/docs/components/training/loss) | 예측과 정답 간 차이 측정 |
| [최적화 알고리즘](/ko/docs/components/training/optimizer) | 파라미터 업데이트 방법 |
| [정규화 기법](/ko/docs/components/training/regularization) | 과적합 방지 |
| [PEFT](/ko/docs/components/training/peft) | 효율적 미세조정 |

## 학습 루프

```python
for epoch in range(epochs):
    for batch in dataloader:
        # 순전파
        output = model(batch)
        loss = criterion(output, target)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
```

## 관련 콘텐츠

- [Backpropagation](/ko/docs/math/calculus/backpropagation)
- [Gradient](/ko/docs/math/calculus/gradient)
