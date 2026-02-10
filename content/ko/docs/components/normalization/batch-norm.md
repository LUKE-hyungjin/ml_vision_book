---
title: "Batch Normalization"
weight: 1
math: true
---

# Batch Normalization (배치 정규화)

{{% hint info %}}
**선수지식**: [기댓값](/ko/docs/math/probability/expectation)
{{% /hint %}}

> **한 줄 요약**: Batch Normalization은 미니배치 단위로 각 채널의 활성화를 **평균 0, 분산 1**로 정규화하여, 학습을 빠르고 안정적으로 만드는 기법입니다.

## 왜 Batch Normalization이 필요한가?

### 문제 상황: "레이어를 깊게 쌓으면 학습이 불안정합니다"

```python
# 깊은 네트워크에서 일어나는 일
x = input_image                    # 평균 ~0, 분산 ~1
x = layer1(x)                      # 평균 0.5, 분산 2.3
x = layer2(x)                      # 평균 3.1, 분산 15.7
x = layer3(x)                      # 평균 -12.4, 분산 89.2  ← 값이 폭발!
x = layer10(x)                     # NaN...  ← 학습 실패
```

**Internal Covariate Shift**: 레이어를 통과할수록 활성화 분포가 계속 변합니다.

```
Layer 1 출력: 평균 0, 분산 1    → Layer 2는 이 분포에 맞춰 학습
    ↓ (파라미터 업데이트)
Layer 1 출력: 평균 2, 분산 5    → Layer 2 입장에서는 "입력이 갑자기 바뀜!"
    ↓
Layer 2가 새 분포에 적응하느라 학습이 느려짐
```

### 해결: "각 레이어 입력을 매번 정규화하자!"

시험 점수에 비유하면:
- **정규화 전** = 과목마다 만점이 다름 (국어 100점, 수학 200점, 영어 50점)
- **정규화 후** = 모든 과목을 표준 점수로 변환 (평균 0, 분산 1)

이렇게 하면 어떤 레이어든 **일관된 범위의 입력**을 받게 됩니다.

![Batch Normalization 정규화 축](/images/components/normalization/ko/normalization-comparison.jpeg)

---

## 수식

### 3단계 과정

입력 텐서 $(B, C, H, W)$에서 **채널별로** 정규화합니다.

**Step 1. 배치 통계 계산** (채널 $c$마다):

$$
\mu_c = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} x_{b,c,h,w}
$$

$$
\sigma_c^2 = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} (x_{b,c,h,w} - \mu_c)^2
$$

**Step 2. 정규화** (평균 0, 분산 1로):

$$
\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
$$

**Step 3. 스케일/시프트** (학습 가능):

$$
y_{b,c,h,w} = \gamma_c \cdot \hat{x}_{b,c,h,w} + \beta_c
$$

**각 기호의 의미:**
- $\mu_c$ : 채널 $c$의 배치 평균 — $B \times H \times W$개 값의 평균
- $\sigma_c^2$ : 채널 $c$의 배치 분산
- $\epsilon$ : 0으로 나누기 방지 (보통 $10^{-5}$)
- $\gamma_c, \beta_c$ : **학습 가능한** 스케일/시프트 파라미터

### 왜 γ, β가 필요한가?

정규화만 하면 항상 평균 0, 분산 1로 강제합니다. 하지만 네트워크가 **다른 분포가 더 좋다고 판단**하면, $\gamma$와 $\beta$로 되돌릴 수 있습니다.

```
γ=σ, β=μ 로 설정하면 → 정규화 취소 (원래 분포 복원!)
→ "정규화할지 말지"를 네트워크가 학습으로 결정
```

### 정규화 축: 어디서 평균을 구하나?

![Batch Normalization 정규화 축](/images/components/normalization/ko/batch-norm-axis.png)

```
입력: (B, C, H, W)

    B (배치)
    ↑
    │  ┌─────┬─────┬─────┐
    │  │ c=1 │ c=2 │ c=3 │  ← 채널
    3  │     │     │     │
    │  │     │ ←이 │     │
    2  │     │  파란│     │
    │  │     │  영역│     │
    1  │     │     │     │
    │  └─────┴─────┴─────┘
    └──────→ C × H × W

BatchNorm: 파란 영역 = B × H × W 방향으로 평균
→ 같은 채널의 모든 위치, 모든 샘플에서 평균
→ 채널당 1개의 μ, σ² (총 C개)
```

---

## Training vs Inference

### 핵심 차이

| | Training | Inference |
|---|---|---|
| 통계 | **현재 배치**의 평균/분산 | **저장된** running 평균/분산 |
| 업데이트 | running stats를 EMA로 업데이트 | 업데이트 없음 |
| 배치 의존 | O (배치마다 통계 다름) | X (고정된 통계) |

### Running Statistics (EMA)

학습 중에 전체 데이터의 통계를 **지수이동평균(EMA)**으로 추적합니다:

$$
\hat{\mu} \leftarrow (1 - m) \cdot \hat{\mu} + m \cdot \mu_{\text{batch}}
$$

$$
\hat{\sigma}^2 \leftarrow (1 - m) \cdot \hat{\sigma}^2 + m \cdot \sigma^2_{\text{batch}}
$$

- $m$ : momentum (기본 0.1)
- $\hat{\mu}, \hat{\sigma}^2$ : running statistics (버퍼에 저장)

```python
# 반드시 모드 전환!
model.train()   # Training: 배치 통계 사용, running stats 업데이트
model.eval()    # Inference: running stats 사용, 업데이트 안 함

# 흔한 버그: eval() 안 하고 추론
# → 배치마다 결과가 달라지는 비결정적 동작!
```

---

## 구현

```python
import torch
import torch.nn as nn

# === PyTorch BatchNorm ===
bn = nn.BatchNorm2d(num_features=64)  # 채널 수

x = torch.randn(32, 64, 56, 56)  # (B, C, H, W)
y = bn(x)

# 학습 가능 파라미터: γ(weight), β(bias) — 각 C개
print(f"gamma: {bn.weight.shape}")          # [64]
print(f"beta: {bn.bias.shape}")             # [64]

# 버퍼: running_mean, running_var — 학습 불가, 저장용
print(f"running_mean: {bn.running_mean.shape}")  # [64]
print(f"running_var: {bn.running_var.shape}")     # [64]


# === 수동 구현 ===
class ManualBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # 학습 가능 파라미터
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # 버퍼 (저장되지만 학습 안 됨)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x: (B, C, H, W)
        if self.training:
            # 현재 배치 통계
            mean = x.mean(dim=(0, 2, 3))            # B, H, W 축 평균 → (C,)
            var = x.var(dim=(0, 2, 3), unbiased=False)  # (C,)
            # Running stats 업데이트 (EMA)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                     self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + \
                                    self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # 정규화 + 스케일/시프트
        x_norm = (x - mean[None, :, None, None]) / \
                 torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_norm + \
               self.beta[None, :, None, None]
```

---

## CNN에서의 위치

### Conv → BN → ReLU (표준 순서)

```python
# 현대 CNN의 기본 블록
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=kernel_size // 2,
                              bias=False)    # BN이 bias 역할 → Conv bias 불필요!
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ResNet의 모든 Conv 블록이 이 패턴
```

**`bias=False`인 이유**: BN의 $\beta$가 bias 역할을 하므로, Conv의 bias는 중복됩니다.

### 파라미터 수

```python
# BatchNorm2d(256)의 파라미터
bn = nn.BatchNorm2d(256)
params = sum(p.numel() for p in bn.parameters())  # γ + β
buffers = sum(b.numel() for b in bn.buffers())     # running_mean + running_var
print(f"학습 파라미터: {params}")   # 512 (256 + 256)
print(f"버퍼: {buffers}")          # 512 (256 + 256)
# Conv에 비하면 매우 적음 (Conv: 수만~수십만)
```

---

## 장점과 한계

### 장점

| 효과 | 설명 |
|------|------|
| **학습 안정화** | 활성화 분포가 일정하게 유지 |
| **큰 학습률 사용** | 분포가 안정적이므로 공격적 업데이트 가능 |
| **초기화 둔감** | 가중치 초기화에 덜 민감 |
| **약한 정규화** | 배치 내 노이즈가 Dropout과 유사한 효과 |

### 한계

| 문제 | 상황 | 대안 |
|------|------|------|
| **작은 배치** | Detection/Seg에서 배치 1~4 | [GroupNorm](/ko/docs/components/normalization/group-norm) |
| **Train/Eval 불일치** | running stats가 실제와 다를 때 | [LayerNorm](/ko/docs/components/normalization/layer-norm) |
| **Transformer 부적합** | 시퀀스 길이가 다를 때 | [LayerNorm](/ko/docs/components/normalization/layer-norm) |
| **Style Transfer** | 이미지별 통계 필요 | [InstanceNorm](/ko/docs/components/normalization/instance-norm) |

---

## 코드로 확인하기

```python
import torch
import torch.nn as nn

# === 정규화 효과 확인 ===
print("=== 정규화 전후 비교 ===")
bn = nn.BatchNorm2d(3)
bn.train()

# 분포가 편향된 입력
x = torch.randn(16, 3, 8, 8) * 10 + 50  # 평균 ~50, 분산 ~100

y = bn(x)

print(f"입력 — 평균: {x.mean():.1f}, 분산: {x.var():.1f}")
print(f"출력 — 평균: {y.mean():.4f}, 분산: {y.var():.4f}")
# 출력 ≈ 평균 0, 분산 1

# === Train vs Eval 동작 차이 ===
print("\n=== Train vs Eval ===")
bn = nn.BatchNorm2d(3)

# 학습 모드에서 여러 배치 처리
bn.train()
for _ in range(100):
    x = torch.randn(16, 3, 8, 8) * 2 + 3  # 평균 3, 분산 4
    _ = bn(x)

print(f"Running mean: {bn.running_mean}")    # ≈ [3, 3, 3]
print(f"Running var:  {bn.running_var}")      # ≈ [4, 4, 4]

# Eval 모드: running stats 사용
bn.eval()
x_test = torch.randn(1, 3, 8, 8) * 2 + 3
y_test = bn(x_test)

# 배치 1개도 일관된 결과!
print(f"Eval 출력 평균: {y_test.mean():.4f}")

# === bias=False 확인 ===
print("\n=== Conv bias=False ===")
conv_with_bias = nn.Conv2d(3, 64, 3, padding=1, bias=True)
conv_no_bias = nn.Conv2d(3, 64, 3, padding=1, bias=False)
bn64 = nn.BatchNorm2d(64)

p_with = sum(p.numel() for p in conv_with_bias.parameters())
p_without = sum(p.numel() for p in conv_no_bias.parameters())
print(f"Conv (bias=True):  {p_with:,} 파라미터")
print(f"Conv (bias=False): {p_without:,} 파라미터")
print(f"절약: {p_with - p_without}개 (BN의 β가 대체)")
```

---

## 핵심 정리

| 항목 | 내용 |
|------|------|
| **정규화 축** | Batch + H + W (채널별) |
| **통계** | 채널당 $\mu, \sigma^2$ — 총 $C$개씩 |
| **학습 파라미터** | $\gamma, \beta$ — 각 $C$개 |
| **배치 의존** | O — 배치 크기 ≥ 16 권장 |
| **Train/Eval 차이** | O — `model.eval()` 필수! |
| **주 사용처** | CNN (ResNet, VGG 등) |

---

## 딥러닝 연결고리

| 개념 | 어디서 쓰이나 | 왜 중요한가 |
|------|-------------|------------|
| Conv-BN-ReLU | [ResNet](/ko/docs/architecture/cnn/resnet), [VGG](/ko/docs/architecture/cnn/vgg) | CNN의 표준 블록 |
| BN + bias=False | 거의 모든 CNN | 파라미터 중복 제거 |
| BN → GN 교체 | Detection, Segmentation | 작은 배치 대응 |
| BN → LN 교체 | Transformer, LLM | 시퀀스 모델 대응 |

---

## 관련 콘텐츠

- [기댓값](/ko/docs/math/probability/expectation) — 선수 지식: 평균, 분산의 정의
- [Layer Normalization](/ko/docs/components/normalization/layer-norm) — Transformer의 정규화
- [Group Normalization](/ko/docs/components/normalization/group-norm) — 작은 배치에서의 대안
- [Instance Normalization](/ko/docs/components/normalization/instance-norm) — Style Transfer 정규화
- [RMSNorm](/ko/docs/components/normalization/rms-norm) — LayerNorm의 경량화 버전
- [Dropout](/ko/docs/components/training/regularization/dropout) — 또 다른 정규화 기법
