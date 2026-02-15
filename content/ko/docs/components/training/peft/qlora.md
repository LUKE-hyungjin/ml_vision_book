---
title: "QLoRA"
weight: 2
math: true
---

# QLoRA (Quantized LoRA)

{{% hint info %}}
**선수지식**: [LoRA](/ko/docs/components/training/peft/lora) | [Data Types](/ko/docs/components/quantization/data-types)
{{% /hint %}}

## 한 줄 요약
> **QLoRA는 베이스 모델 가중치를 4bit로 줄이고, 작은 LoRA 어댑터만 학습해서 메모리 비용을 크게 낮추는 미세튜닝 방법입니다.**

## 왜 필요한가?
큰 모델을 풀 파인튜닝하면 GPU 메모리 요구량이 너무 커서, 개인/소규모 팀이 실험하기 어렵습니다.

QLoRA는 문제를 이렇게 쪼갭니다.
- **베이스 모델**: 4bit로 압축해 메모리에 올림(주로 고정)
- **학습되는 부분**: LoRA 저랭크 어댑터만 업데이트

즉, "큰 본체는 최대한 가볍게 유지"하고, "작은 보정 파츠만 학습"하는 전략입니다.

## 핵심 아이디어 (3단 구성)
1. **NF4 4bit 양자화**: 정규분포형 가중치에 맞춘 4bit 표현
2. **Double Quantization**: scale 같은 양자화 상수도 추가로 압축
3. **LoRA 학습**: 원본 가중치는 고정하고 저랭크 행렬만 업데이트

## 수식과 기호
QLoRA의 핵심은 "양자화된 베이스 + LoRA 보정"으로 볼 수 있습니다.

$$
W' = \operatorname{dequant}(W_q) + \Delta W, \quad \Delta W = BA
$$

**각 기호의 의미:**
- $W_q$: 4bit로 저장된 베이스 가중치
- $\operatorname{dequant}(W_q)$: 연산 시 복원된 근사 가중치
- $\Delta W$: LoRA가 학습하는 보정량
- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$: 저랭크 행렬
- $r$: rank (작을수록 메모리 절약, 너무 작으면 표현력 부족)

Double Quantization은 개념적으로 아래처럼 이해하면 충분합니다.
$$
\text{store}(W) \approx \text{quant}(W; s_1), \quad s_1 \approx \text{dequant}(\text{quant}(s_1; s_2))
$$

핵심은 "가중치뿐 아니라 스케일까지 압축"한다는 점입니다.

## 직관
- LoRA만 쓸 때: 가벼운 어댑터를 붙이지만 베이스가 FP16/FP32면 메모리 부담이 큼
- QLoRA: 베이스를 4bit로 줄여 **상주 메모리 자체를 크게 축소**

비유하면,
- 큰 백과사전 본문은 압축본으로 보관하고,
- 중요한 수정사항만 포스트잇(LoRA)으로 붙여 학습하는 방식입니다.

## 최소 구현 (Hugging Face + bitsandbytes)
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# k-bit 학습 안정화를 위한 준비
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 학습되는 파라미터 비율 확인
```

## 초보자 필수 체크: 메모리 계산 감각
학습 전에 아래 2가지를 꼭 확인하세요.
1. **학습 파라미터 비율**: trainable params가 전체 대비 몇 %인지
2. **실제 VRAM 피크**: 배치 크기/시퀀스 길이에 따라 피크 메모리가 급증하는지

작게 시작해서(`batch=1~2`, 짧은 sequence) 안정화 후 올리는 게 안전합니다.

## 10분 미니 실습: "진짜로 LoRA만 학습 중인지" 검증하기
초보자가 가장 많이 막히는 지점은 **설정은 QLoRA처럼 했는데 실제로는 다른 파라미터가 같이 학습되는 경우**입니다.
아래 3가지를 한 번에 확인하면 초기 시행착오를 크게 줄일 수 있습니다.

1. `trainable params` 비율이 예상대로 작은가?
2. optimizer에 들어간 파라미터가 정말 `requires_grad=True`만인가?
3. 한 step 이후 베이스 가중치가 바뀌지 않았는가?

```python
# 1) 학습 가능 파라미터 집계
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
all_params = sum(p.numel() for _, p in model.named_parameters())
trainable_params = sum(p.numel() for _, p in trainable)

print(f"trainable ratio = {trainable_params / all_params:.4%}")

# 2) optimizer가 trainable 파라미터만 받는지 확인
optimizer = torch.optim.AdamW((p for _, p in trainable), lr=2e-4)

# 3) 베이스 가중치 불변성 빠른 점검(한 step 전후 비교)
base_name, base_param = next((n, p) for n, p in model.named_parameters() if not p.requires_grad)
before = base_param.detach().clone()

# ... forward / loss / backward / optimizer.step()

after = base_param.detach()
print("base changed:", not torch.allclose(before, after))  # 기대값: False
```

해석:
- `trainable ratio`가 지나치게 크면 LoRA 주입/동결 설정을 의심하세요.
- `base changed=True`면 freeze가 깨진 것입니다.
- 위 검증을 통과한 뒤에야 lr/rank/데이터 튜닝이 의미가 있습니다.

## 실무 디버깅 체크리스트
- [ ] `target_modules`가 모델 구조와 맞는가? (이름 불일치면 LoRA가 안 붙음)
- [ ] `prepare_model_for_kbit_training`를 호출했는가?
- [ ] BF16 가능한 GPU에서 `bnb_4bit_compute_dtype=torch.bfloat16`를 우선 검토했는가?
- [ ] OOM 시 batch/sequence/gradient checkpointing부터 줄였는가?
- [ ] 학습 후 `trainable params` 비율을 로그로 확인했는가?

## 자주 하는 실수 (FAQ)
**Q1. QLoRA면 무조건 full fine-tuning과 같은 성능이 나오나요?**  
A. 아닙니다. 작업 난이도/데이터 품질/하이퍼파라미터에 따라 차이가 날 수 있습니다.

**Q2. rank(r)를 크게 하면 항상 좋아지나요?**  
A. 일정 구간까지는 개선될 수 있지만, 메모리와 과적합 위험도 함께 늘어납니다.

**Q3. 4bit면 추론도 항상 빨라지나요?**  
A. 메모리는 줄지만, 커널/하드웨어 최적화 상태에 따라 속도 이득은 달라질 수 있습니다.

## 증상 → 원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 볼 항목 |
|---|---|---|
| 학습이 시작은 되지만 손실이 거의 안 내려감 | LoRA가 실제로 attach되지 않음 | `target_modules`, trainable params 출력 |
| 초반 OOM 반복 | sequence 길이/배치 과대 | `max_length`, batch, grad accumulation |
| 성능이 baseline보다 크게 낮음 | rank/alpha 과소 또는 데이터 품질 문제 | `r`, `lora_alpha`, 학습 데이터 샘플 점검 |
| 학습은 되는데 추론 품질 흔들림 | 과적합 또는 평가 프롬프트 불일치 | validation set, decoding 설정 |

## 다음 학습 링크
- [LoRA](/ko/docs/components/training/peft/lora)
- [Data Types](/ko/docs/components/quantization/data-types)
- [PTQ](/ko/docs/components/quantization/ptq)
- [QAT](/ko/docs/components/quantization/qat)
