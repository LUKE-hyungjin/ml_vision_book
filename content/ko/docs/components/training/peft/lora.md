---
title: "LoRA"
weight: 1
math: true
---

# LoRA (Low-Rank Adaptation)

{{% hint info %}}
**선수지식**: [행렬 곱셈](/ko/docs/math/linear-algebra/matrix) | [SVD](/ko/docs/math/linear-algebra/svd) | [Transformer](/ko/docs/architecture/transformer)
{{% /hint %}}

## 한 줄 요약

> **"거대한 가중치를 건드리지 않고, 작은 업데이트만 학습한다"**

---

## 왜 LoRA인가?

### 문제: 대형 모델 Fine-tuning의 비용

7B 파라미터 모델을 Full Fine-tuning 하려면?

```
파라미터: 7,000,000,000개
FP16 기준: 14GB (가중치만)
학습 시: ~56GB+ (Optimizer state, Gradient 포함)

→ A100 80GB도 빠듯!
```

### 해결: 업데이트만 따로 학습

> **핵심 통찰**: 대부분의 가중치 변화는 "저차원"에서 일어난다!

```
Full Fine-tuning:
W_new = W_old + ΔW   (ΔW: d×k 전체 업데이트)

LoRA:
W_new = W_old + B·A  (B: d×r, A: r×k, r << d,k)
```

---

## 핵심 아이디어

### 저랭크 분해 (Low-Rank Decomposition)

ΔW를 두 개의 작은 행렬로 분해합니다:

$$
\Delta W = B \cdot A
$$

{{< figure src="/images/math/training/peft/ko/lora-decomposition.png" caption="LoRA 저랭크 분해: W + B·A" >}}

**기호 설명:**
- $W$: 원본 가중치 (d × k), **고정**
- $B$: 다운 프로젝션 (d × r), **학습**
- $A$: 업 프로젝션 (r × k), **학습**
- $r$: 랭크 (보통 4~64), $r \ll \min(d, k)$

### 왜 "저랭크"가 작동하나?

> **Intrinsic Dimensionality 가설**: 사전학습된 모델의 업데이트는 대부분 저차원 부분공간에서 일어난다.

직관적으로:
- 사전학습 중 이미 "좋은 표현"을 배움
- Fine-tuning은 "미세 조정"만 하면 됨
- 미세 조정 = 전체 공간 중 일부만 변경

---

## 파라미터 절감

### 계산

| 방법 | 파라미터 수 |
|------|------------|
| **Full Fine-tuning** | $d \times k$ |
| **LoRA** | $d \times r + r \times k = r(d + k)$ |

### 예시: LLaMA-7B의 한 층

{{< figure src="/images/math/training/peft/ko/lora-savings.png" caption="LoRA 파라미터 절감 효과" >}}

$d = 4096$, $k = 4096$, $r = 8$

| 방법 | 파라미터 수 | 비율 |
|------|------------|------|
| Full | 16,777,216 | 100% |
| LoRA | 65,536 | **0.39%** |

**256배 절감!**

---

## 수학적 정의

### 순전파 (Forward Pass)

{{< figure src="/images/math/training/peft/ko/lora-forward.png" caption="LoRA Forward Pass: 원본 경로와 LoRA 경로" >}}

원본:
$$
h = Wx
$$

LoRA 적용:
$$
h = Wx + BAx = Wx + B(Ax)
$$

### 스케일링 팩터

학습 안정성을 위해 스케일링 적용:

$$
h = Wx + \frac{\alpha}{r} \cdot BAx
$$

**기호 설명:**
- $\alpha$: 스케일링 팩터 (하이퍼파라미터)
- $\frac{\alpha}{r}$: 랭크에 따른 정규화

> **팁**: $\alpha = r$로 설정하면 $\frac{\alpha}{r} = 1$이 되어 학습률 튜닝이 쉬워집니다.

### 초기화

- $A$: 가우시안 초기화 ($\mathcal{N}(0, \sigma^2)$)
- $B$: **0으로 초기화**

$$
\Delta W_{init} = B \cdot A = 0
$$

**이유**: 학습 시작 시 원본 모델과 동일하게 동작!

---

## 구현

### 기본 구현

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: int = 8):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False  # 원본 고정
        if original_layer.bias is not None:
            self.original.bias.requires_grad = False

        d, k = original_layer.weight.shape

        # LoRA 행렬
        self.lora_A = nn.Parameter(torch.randn(r, k) * 0.01)  # 가우시안 초기화
        self.lora_B = nn.Parameter(torch.zeros(d, r))         # 0 초기화
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 원본 출력 + LoRA 출력
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return original_out + lora_out
```

### 모델에 적용

```python
def apply_lora(model, target_modules, r=8, alpha=8):
    """특정 모듈에 LoRA 적용"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)

                # 교체
                setattr(parent, child_name, LoRALinear(module, r, alpha))

    return model
```

---

## 어디에 적용?

### Transformer 기준

{{< figure src="/images/math/training/peft/ko/lora-transformer.png" caption="Transformer에서 LoRA 적용 위치" >}}

```python
# Attention
target_modules = [
    "q_proj",  # Query (필수)
    "v_proj",  # Value (필수)
    "k_proj",  # Key (선택)
    "o_proj",  # Output (선택)
]

# FFN (MLP)
target_modules += [
    "gate_proj",  # 선택
    "up_proj",    # 선택
    "down_proj",  # 선택
]
```

### 권장 설정

| 태스크 | 적용 대상 | 이유 |
|--------|----------|------|
| **텍스트 생성** | q, v | 최소 비용으로 좋은 성능 |
| **분류** | q, k, v, o | 더 많은 표현력 필요 |
| **복잡한 태스크** | 모든 Linear | 최대 성능 |

---

## PEFT 라이브러리

### 기본 사용법

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA 설정
lora_config = LoraConfig(
    r=8,                                    # 랭크
    lora_alpha=16,                          # 스케일링 (alpha/r = 2)
    target_modules=["q_proj", "v_proj"],    # 적용 대상
    lora_dropout=0.05,                      # Dropout
    bias="none",                            # bias는 학습 안 함
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(base_model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || 0.06%
```

### 학습 후 병합

```python
# 방법 1: 영구 병합 (원본 모델 크기로 저장)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# 방법 2: Adapter만 저장 (용량 절약)
model.save_pretrained("./lora_adapter")  # ~10MB

# 나중에 로드
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base, "./lora_adapter")
```

---

## 추론 시 병합

학습 후 LoRA를 원본에 병합하면 **추론 속도 저하 없음**:

$$
W_{merged} = W + \frac{\alpha}{r} \cdot B \cdot A
$$

```python
# 수동 병합
with torch.no_grad():
    W_merged = W + (alpha / r) * B @ A
```

**장점**:
- 추론 시 추가 연산 없음
- 원본 모델과 동일한 아키텍처
- 여러 LoRA를 동적으로 교체 가능

---

## 하이퍼파라미터 가이드

| 파라미터 | 범위 | 권장 | 설명 |
|----------|------|------|------|
| **r (랭크)** | 4~256 | 8~64 | 높을수록 표현력↑, 파라미터↑ |
| **alpha** | r~4r | 2r | 학습률 스케일링, alpha/r이 중요 |
| **dropout** | 0~0.1 | 0.05 | 과적합 방지 |
| **target** | - | q, v | 최소한 이 두 개 |

### 랭크 선택 가이드

```
r=4:   매우 간단한 태스크 (스타일 변환)
r=8:   일반적인 Fine-tuning (권장 시작점)
r=16:  복잡한 태스크
r=64:  Full Fine-tuning에 가까운 성능 필요 시
r=256: 극단적 케이스
```

---

## LoRA vs 다른 방법

| 방법 | 학습 파라미터 | 추론 속도 | 메모리 | 성능 |
|------|--------------|----------|--------|------|
| **Full Fine-tuning** | 100% | 기준 | 높음 | 최고 |
| **LoRA** | 0.1~1% | 기준 (병합 후) | 낮음 | 근접 |
| **Adapter** | 1~5% | 느림 | 중간 | 좋음 |
| **Prefix Tuning** | 0.1% | 조금 느림 | 낮음 | 제한적 |

---

## 응용: 여러 LoRA 조합

### 동적 LoRA 스위칭

```python
# 여러 태스크용 LoRA 학습
lora_korean = "path/to/korean_adapter"
lora_coding = "path/to/coding_adapter"

# 런타임에 교체
model.load_adapter(lora_korean, "korean")
model.load_adapter(lora_coding, "coding")

model.set_adapter("korean")  # 한국어 모드
output = model.generate(...)

model.set_adapter("coding")  # 코딩 모드
output = model.generate(...)
```

### LoRA 합성 (Composition)

```python
# 두 LoRA를 가중 합산
W_combined = W + 0.7 * (B1 @ A1) + 0.3 * (B2 @ A2)
```

---

## 요약

| 질문 | 답변 |
|------|------|
| LoRA가 뭔가요? | 가중치 업데이트를 저랭크 행렬로 분해 |
| 왜 쓰나요? | 파라미터 99% 절감, 메모리 효율 |
| 어떻게 작동하나요? | $W' = W + BA$, W는 고정 |
| 성능은? | Full Fine-tuning의 90~99% |
| 추론 속도는? | 병합 후 원본과 동일 |

---

## 관련 콘텐츠

- [QLoRA](/ko/docs/components/training/peft/qlora) - LoRA + 4bit 양자화
- [Adapter](/ko/docs/components/training/peft/adapter) - 레이어 삽입 방식
- [Prefix Tuning](/ko/docs/components/training/peft/prefix-tuning) - 프롬프트 학습
- [SVD](/ko/docs/math/linear-algebra/svd) - 저랭크 분해 수학
- [Transformer](/ko/docs/architecture/transformer) - LoRA 적용 대상
