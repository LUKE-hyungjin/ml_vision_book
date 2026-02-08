---
title: "QLoRA"
weight: 2
math: true
---

# QLoRA (Quantized LoRA)

## 개요

QLoRA는 베이스 모델을 4비트로 양자화하고 LoRA를 적용하여 메모리를 극적으로 줄입니다.

## 핵심 아이디어

```
기존 LoRA: FP16 모델 + LoRA 어댑터
QLoRA:     4bit 모델 + LoRA 어댑터 (FP16)
```

## 메모리 비교

65B 모델 기준:
| 방식 | GPU 메모리 |
|------|-----------|
| Full Fine-tuning (FP16) | >780GB |
| LoRA (FP16) | ~130GB |
| QLoRA (NF4) | ~33GB |

## 핵심 기술

### 1. NF4 (4-bit NormalFloat)

정규분포에 최적화된 4비트 양자화:

```python
# NF4 양자값 (비균등 간격)
nf4_values = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
               0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0]
```

### 2. Double Quantization

양자화 상수(scale)도 양자화:

$$
\text{저장} = \text{quant}(W, s_1) + \text{quant}(s_1, s_2)
$$

### 3. Paged Optimizers

GPU 메모리 부족 시 CPU로 자동 이동:

```python
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=1e-4,
    is_paged=True  # 메모리 부족 시 CPU로
)
```

## 구현

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Double Quantization
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Gradient checkpointing 준비
model = prepare_model_for_kbit_training(model)

# LoRA 적용
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

## 학습

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    optim="paged_adamw_8bit"  # Paged optimizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator
)

trainer.train()
```

## 주의사항

1. **추론 속도**: 4bit 양자화로 인해 약간 느려짐
2. **정확도**: Full fine-tuning 대비 미세한 성능 차이
3. **병합 불가**: 양자화된 모델에 LoRA 병합 어려움

## 관련 콘텐츠

- [LoRA](/ko/docs/components/training/peft/lora)
- [Quantization](/ko/docs/components/quantization)
- [Transformer](/ko/docs/architecture/transformer)
