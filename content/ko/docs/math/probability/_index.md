---
title: "확률/통계"
weight: 3
bookCollapseSection: true
math: true
---

# 확률과 통계 (Probability & Statistics)

딥러닝은 본질적으로 확률 모델입니다. 분류, 생성, 불확실성 추정 모두 확률에 기반합니다.

## 왜 확률인가?

- **분류**: P(class | image) 확률 출력
- **생성**: P(image) 분포에서 샘플링
- **Loss**: 확률분포 간 거리 (Cross-Entropy, KL Divergence)
- **Dropout**: 베르누이 분포로 랜덤 마스킹
- **Diffusion**: 노이즈 추가/제거의 확률 과정

---

## 🟢 기본 (Basic)

확률을 처음 배우거나 복습이 필요한 분들을 위한 기초 개념입니다.

| 개념 | 설명 | 선수 지식 |
|------|------|----------|
| [확률의 기초](/ko/docs/math/probability/basics) | 확률의 정의, 조건부 확률, 독립 | 없음 |
| [확률 변수](/ko/docs/math/probability/random-variable) | 이산/연속 확률 변수 | 확률의 기초 |
| [기댓값과 분산](/ko/docs/math/probability/expectation) | 평균, 분산, 공분산 | 확률 변수 |

### 학습 순서

```
확률의 기초 → 확률 변수 → 기댓값과 분산
```

---

## 🔵 핵심 (Core)

딥러닝에서 반드시 알아야 할 확률 개념입니다.

| 개념 | 설명 | 딥러닝 적용 |
|------|------|------------|
| [확률분포](/ko/docs/math/probability/distribution) | 확률의 함수적 표현 | Softmax, VAE, Diffusion |
| [베이즈 정리](/ko/docs/math/probability/bayes) | 조건부 확률의 역산 | 불확실성 추정, 사후 확률 |
| [샘플링](/ko/docs/math/probability/sampling) | 분포에서 값 추출 | 생성 모델, Dropout, MCMC |

### 딥러닝에서의 활용

```
입력 x → 모델 → P(y|x) 확률분포 출력 → 샘플링 또는 argmax
```

---

## 🔴 심화 (Advanced)

손실 함수와 정보 이론의 기반이 되는 개념들입니다.

| 개념 | 설명 | 딥러닝 적용 |
|------|------|------------|
| [엔트로피](/ko/docs/math/probability/entropy) | 불확실성의 측정 | Cross-Entropy Loss |
| [KL 발산](/ko/docs/math/probability/kl-divergence) | 분포 간 거리 | VAE, 지식 증류 |
| [최대 우도 추정](/ko/docs/math/probability/mle) | 파라미터 추정 | 모델 학습의 원리 |

### 손실 함수와의 관계

$$
\text{Cross-Entropy} = H(p) + D_{KL}(p \| q)
$$

- Cross-Entropy를 최소화 = KL Divergence 최소화 = 두 분포를 같게

---

## 전체 학습 로드맵

```
[기본]                    [핵심]                    [심화]
확률의 기초 ─────────────→ 확률분포 ─────────────→ 엔트로피
    │                        │                        │
    ↓                        ↓                        ↓
확률 변수 ───────────────→ 베이즈 정리 ──────────→ KL 발산
    │                        │                        │
    ↓                        ↓                        ↓
기댓값과 분산 ───────────→ 샘플링 ─────────────→ 최대 우도 추정
```

---

## 관련 콘텐츠

- [Cross-Entropy Loss](/ko/docs/math/training/loss/cross-entropy) - 확률분포 기반 손실
- [Diffusion](/ko/docs/math/diffusion) - 확률 과정 기반 생성
- [Label Smoothing](/ko/docs/math/training/regularization/label-smoothing) - 분포 정규화
- [Softmax](/ko/docs/math/probability/distribution) - 확률분포 변환
