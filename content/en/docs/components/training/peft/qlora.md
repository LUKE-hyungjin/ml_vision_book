---
title: "QLoRA"
weight: 2
math: true
---

# QLoRA (Quantized LoRA)

{{% hint info %}}
**Prerequisites**: [LoRA](/en/docs/components/training/peft/lora) | [Data Types](/en/docs/components/quantization/data-types)
{{% /hint %}}

## One-line Summary
> **QLoRA keeps base-model weights in 4-bit form and trains only small LoRA adapters, dramatically reducing fine-tuning memory cost.**

## Why is this needed?
Full fine-tuning of large models often exceeds realistic GPU memory budgets.

QLoRA splits the problem into two parts:
- **Base model**: loaded in 4-bit (mostly frozen)
- **Trainable part**: only low-rank LoRA adapters are updated

So you keep the large backbone lightweight and train only small correction modules.

## Core idea (3 building blocks)
1. **NF4 4-bit quantization**: optimized for roughly normal weight distributions
2. **Double Quantization**: compress quantization constants (like scales) too
3. **LoRA updates**: freeze base weights, train low-rank adapters only

## Formula and symbols
A useful view of QLoRA is "quantized base + LoRA correction":

$$
W' = \operatorname{dequant}(W_q) + \Delta W, \quad \Delta W = BA
$$

**Symbol meanings:**
- $W_q$: base weights stored in 4-bit format
- $\operatorname{dequant}(W_q)$: approximate real-valued weights used in compute
- $\Delta W$: LoRA trainable update
- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$: low-rank matrices
- $r$: LoRA rank (smaller saves memory; too small can hurt capacity)

Double quantization can be understood conceptually as:
$$
\text{store}(W) \approx \text{quant}(W; s_1), \quad s_1 \approx \text{dequant}(\text{quant}(s_1; s_2))
$$

The key is that not only weights, but also scale metadata, is compressed.

## Intuition
- LoRA alone: trainable adapters are cheap, but FP16/FP32 base can still be memory-heavy
- QLoRA: 4-bit base reduces **resident memory** of the backbone significantly

Analogy:
- keep the encyclopedia body in a compressed archive,
- and apply edits as sticky notes (LoRA) on top.

## Minimal implementation (Hugging Face + bitsandbytes)
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

# Stabilization steps for k-bit training
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
model.print_trainable_parameters()  # verify trainable ratio
```

## Beginner must-check: memory sanity
Before long runs, check these two first:
1. **Trainable parameter ratio**: what % of total params is trainable?
2. **Actual VRAM peak**: does memory spike with batch size / sequence length?

Start small (`batch=1~2`, short sequences), then scale up.

## 10-minute mini lab: verify that only LoRA is actually training
A very common beginner failure mode is: **the setup looks like QLoRA, but non-LoRA parameters are still being updated**.
This 3-step check catches that early.

1. Is the `trainable params` ratio actually small?
2. Does the optimizer receive only `requires_grad=True` params?
3. After one step, did frozen base weights stay unchanged?

```python
# 1) Count trainable parameters
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
all_params = sum(p.numel() for _, p in model.named_parameters())
trainable_params = sum(p.numel() for _, p in trainable)

print(f"trainable ratio = {trainable_params / all_params:.4%}")

# 2) Ensure optimizer only sees trainable params
optimizer = torch.optim.AdamW((p for _, p in trainable), lr=2e-4)

# 3) Quick invariant check on a frozen base weight (before/after one step)
base_name, base_param = next((n, p) for n, p in model.named_parameters() if not p.requires_grad)
before = base_param.detach().clone()

# ... forward / loss / backward / optimizer.step()

after = base_param.detach()
print("base changed:", not torch.allclose(before, after))  # expected: False
```

How to read results:
- If `trainable ratio` is unexpectedly large, inspect LoRA injection/freezing setup first.
- If `base changed=True`, freezing is broken.
- Tune lr/rank/data only after this sanity check passes.

## Practical debugging checklist
- [ ] Do `target_modules` match your model architecture names?
- [ ] Did you call `prepare_model_for_kbit_training`?
- [ ] On BF16-capable GPUs, did you test `bnb_4bit_compute_dtype=torch.bfloat16` first?
- [ ] If OOM happens, did you reduce batch/sequence/enable gradient checkpointing?
- [ ] Did you log the trainable-parameter ratio after LoRA injection?

## Common mistakes (FAQ)
**Q1. Does QLoRA always match full fine-tuning performance?**  
A. No. It can be close, but results depend on task complexity, data quality, and tuning.

**Q2. Is larger rank(r) always better?**  
A. Not always. It may improve capacity up to a point, then increase memory cost and overfitting risk.

**Q3. Does 4-bit always make inference faster?**  
A. Memory almost always improves, but speed gain depends on kernel and hardware support.

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First thing to inspect |
|---|---|---|
| training runs but loss barely moves | LoRA not actually attached | `target_modules`, printed trainable params |
| repeated OOM at early steps | sequence length / batch too large | `max_length`, batch size, grad accumulation |
| much worse quality than baseline | rank/alpha too small or weak data | `r`, `lora_alpha`, dataset sample quality |
| unstable generation quality after training | overfitting or eval prompt mismatch | validation split, decoding setup |

## Next learning links
- [LoRA](/en/docs/components/training/peft/lora)
- [Data Types](/en/docs/components/quantization/data-types)
- [PTQ](/en/docs/components/quantization/ptq)
- [QAT](/en/docs/components/quantization/qat)
