---
title: "Data Types"
weight: 3
math: true
---

# 데이터 타입

{{% hint info %}}
**선수지식**: 없음
{{% /hint %}}

## 한 줄 요약
> **데이터 타입은 같은 숫자를 몇 비트로 표현할지 정하는 규칙이며, 속도·메모리·정확도의 균형을 결정합니다.**

## 왜 필요한가?
Vision 모델은 보통 수백만~수억 개 파라미터를 다룹니다. 
같은 모델이라도 FP32로 돌릴지, FP16/BF16로 돌릴지, INT8로 양자화할지에 따라:

- GPU 메모리 사용량
- 추론 속도/전력
- 정확도 손실

이 크게 달라집니다.

비유하면, 같은 책을 **고해상도 스캔(P32)**으로 보관할지, **압축본(INT8)**으로 보관할지 결정하는 문제입니다.

## 핵심 수식 (양자화의 기본)
정수 양자화에서 자주 쓰는 기본식은 다음과 같습니다.

$$
q = \mathrm{round}\left(\frac{x}{s}\right) + z
$$

$$
\hat{x} = s(q - z)
$$

**각 기호의 의미:**
- $x$ : 원래 실수값(예: FP32 activation)
- $q$ : 양자화된 정수값(INT8 등)
- $s$ : scale (실수↔정수 크기 변환 계수)
- $z$ : zero-point (0을 정수 격자 어디에 둘지)
- $\hat{x}$ : 복원된 근사 실수값

핵심은 **비트를 줄일수록 저장/연산은 싸지지만, 복원 오차가 늘 수 있다**는 점입니다.

## 타입별 빠른 지도

| 타입 | 비트 | 범위/특징 | 주 사용처 |
|---|---:|---|---|
| FP32 | 32 | 넓은 범위, 높은 정밀도 | 기준 학습/디버깅 |
| FP16 | 16 | 범위 좁음, 빠름 | 추론, mixed precision |
| BF16 | 16 | FP32와 유사한 지수 범위 | 학습 안정성 + 속도 |
| INT8 | 8 | 정수 기반, 메모리 절감 큼 | PTQ/QAT 추론 |
| INT4 | 4 | 더 강한 압축 | 초경량 추론(품질 주의) |
| FP8 | 8 | 최신 가속기 최적화 | 최신 GPU 학습/추론 |

## 부동소수점(Floating Point)

### FP32 (Single Precision)
```
| 1 bit | 8 bits  | 23 bits    |
| sign  | exponent| mantissa   |
```
- 범위: 약 $\pm 3.4 \times 10^{38}$
- 정밀도: 약 7자리
- 장점: 안전한 기준 정밀도

### FP16 (Half Precision)
```
| 1 bit | 5 bits  | 10 bits    |
| sign  | exponent| mantissa   |
```
- 범위: 약 $\pm 65504$
- 정밀도: FP32 대비 낮음
- 장점: 메모리 절반, 연산 가속
- 주의: 작은 gradient가 0으로 언더플로우될 수 있음

### BF16 (Brain Float)
```
| 1 bit | 8 bits  | 7 bits     |
| sign  | exponent| mantissa   |
```
- 지수 비트가 FP32와 같아 **범위가 넓음**
- FP16보다 학습 안정성이 좋은 경우가 많음

## 정수(Integer)

### INT8
```
| 1 bit | 7 bits  |
| sign  | value   |
```
- 범위: -128~127 (signed)
- PTQ(Post-Training Quantization)의 기본 선택지

### INT4
```
| 4 bits |
| value  |
```
- 범위: -8~7 (signed)
- 메모리는 더 절약되지만 정확도 손실 가능성 증가

## 특수 포맷

### FP8 (E4M3, E5M2)
```
E4M3: | 1 | 4 | 3 |
E5M2: | 1 | 5 | 2 |
```
- E4M3: 정밀도 우선
- E5M2: 범위 우선

### NF4 (4-bit NormalFloat)
QLoRA에서 사용하는 비균등 4-bit 표현입니다.

```python
# 정규분포 가중치에 맞춘 대표 양자화 포인트 예시
nf4_quantiles = [
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0
]
```

## 최소 구현 예시

```python
import torch

x_fp32 = torch.randn(1024, 1024)
x_fp16 = x_fp32.half()
x_bf16 = x_fp32.bfloat16()

print("bytes per element")
print("fp32:", x_fp32.element_size())  # 4
print("fp16:", x_fp16.element_size())  # 2
print("bf16:", x_bf16.element_size())  # 2
```

## Mixed Precision Training (실전)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in train_loader:
    optimizer.zero_grad()

    with autocast():  # 일부 연산은 FP16/BF16로 자동 처리
        pred = model(x)
        loss = criterion(pred, y)

    scaler.scale(loss).backward()  # underflow 완화
    scaler.step(optimizer)
    scaler.update()
```

## 초보자가 가장 헷갈리는 2축: 대칭/비대칭 + per-tensor/per-channel
INT8를 처음 쓸 때 가장 많이 막히는 지점은 아래 두 가지를 섞어 생각하는 경우입니다.

1. **대칭(symmetric) vs 비대칭(asymmetric)**: zero-point($z$)를 0으로 둘지, 학습/데이터 분포에 맞춰 이동시킬지
2. **per-tensor vs per-channel**: scale($s$)를 텐서 전체 1개로 쓸지, 채널마다 따로 둘지

### (1) 대칭 vs 비대칭
- **대칭 양자화**: 보통 $z=0$ (구현 단순, 하드웨어 친화적)
- **비대칭 양자화**: $z\neq 0$ 허용 (분포가 한쪽으로 치우친 activation에서 유리할 수 있음)

정수 범위가 $[q_{\min}, q_{\max}]$일 때 대표식:

$$
s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \quad
z = \mathrm{round}\left(q_{\min} - \frac{x_{\min}}{s}\right)
$$

### (2) per-tensor vs per-channel
- **per-tensor**: 텐서 전체에 scale 1개 → 빠르고 단순
- **per-channel**: 채널마다 scale 별도 → 정확도 유지에 유리(특히 Conv weight)

실무에서는 보통 **activation은 per-tensor**, **weight는 per-channel** 조합을 자주 사용합니다.

## 5분 실험 2: per-tensor vs per-channel 오차 비교
아래 코드는 Conv weight를 가정하고, 두 방식의 재구성 오차(MAE)를 비교합니다.

```python
import torch


def fake_int8_quant_per_tensor(w: torch.Tensor):
    qmin, qmax = -128, 127
    w_min, w_max = w.min(), w.max()
    s = (w_max - w_min) / (qmax - qmin + 1e-12)
    z = torch.round(torch.tensor(qmin, device=w.device) - w_min / (s + 1e-12))
    q = torch.round(w / (s + 1e-12) + z).clamp(qmin, qmax)
    w_hat = (q - z) * s
    return w_hat


def fake_int8_quant_per_channel(w: torch.Tensor):
    # Conv2d weight 가정: [out_channels, in_channels, kH, kW]
    qmin, qmax = -128, 127
    w_flat = w.view(w.size(0), -1)  # 채널별로 펼침

    w_min = w_flat.min(dim=1, keepdim=True).values
    w_max = w_flat.max(dim=1, keepdim=True).values

    s = (w_max - w_min) / (qmax - qmin + 1e-12)
    z = torch.round(qmin - w_min / (s + 1e-12))

    q = torch.round(w_flat / (s + 1e-12) + z).clamp(qmin, qmax)
    w_hat = (q - z) * s
    return w_hat.view_as(w)


w = torch.randn(64, 64, 3, 3) * torch.linspace(0.2, 2.0, 64).view(64, 1, 1, 1)

w_hat_tensor = fake_int8_quant_per_tensor(w)
w_hat_channel = fake_int8_quant_per_channel(w)

mae_tensor = (w - w_hat_tensor).abs().mean().item()
mae_channel = (w - w_hat_channel).abs().mean().item()

print(f"MAE per-tensor : {mae_tensor:.6f}")
print(f"MAE per-channel: {mae_channel:.6f}")
```

보통 채널별 스케일 편차가 큰 모델일수록 per-channel이 유리합니다.

## 디버깅 체크리스트
- [ ] 학습 중 NaN/Inf가 생기면 FP16 언더플로우/오버플로우를 먼저 의심했는가?
- [ ] BF16 지원 GPU인데도 FP16만 고정 사용하고 있지 않은가?
- [ ] INT8 추론 시 calibration 데이터가 실제 배포 분포와 비슷한가?
- [ ] 정확도 하락이 크면 per-channel quantization 적용 여부를 확인했는가?
- [ ] activation/weight에 같은 양자화 전략을 무조건 적용하고 있지 않은가? (역할이 다름)

## 자주 하는 실수 (FAQ)
**Q1. FP16이 FP32보다 항상 좋은가요?**  
A. 속도/메모리 측면에서는 유리하지만, 수치 안정성은 작업에 따라 나빠질 수 있습니다.

**Q2. BF16과 FP16 중 무엇을 먼저 쓰나요?**  
A. 하드웨어가 지원하면 학습은 BF16을 우선 검토하는 경우가 많습니다(범위가 넓어 안정적).

**Q3. INT8로 바꾸면 정확도 손실이 무조건 큰가요?**  
A. 꼭 그렇진 않습니다. 모델/데이터/캘리브레이션 품질에 따라 손실이 작을 수도 큽니다.

## 초보자용 선택 가이드 (처음엔 뭘 고를까?)
아래 순서대로 선택하면 시행착오를 줄일 수 있습니다.

1. **학습 안정성 우선**이면 FP32 또는 BF16부터 시작
2. **메모리가 부족**하면 FP16/BF16 mixed precision 적용
3. **배포 지연 시간/전력 최적화**가 목표면 INT8(PTQ→필요 시 QAT) 검토

간단 의사결정 표:

| 상황 | 1순위 선택 | 이유 |
|---|---|---|
| 처음 모델 학습/디버깅 | FP32 또는 BF16 | 원인 분리와 재현성이 쉬움 |
| GPU 메모리 부족 | BF16/FP16 mixed precision | 메모리 절감 + 속도 향상 |
| 서버 추론 비용 절감 | INT8 PTQ | 구현 부담 대비 효과가 큼 |
| PTQ 정확도 하락이 큼 | INT8 QAT | 학습 중 양자화 오차를 적응 |

## 증상 → 원인 빠른 매핑
| 관측 증상 | 가장 흔한 원인 | 먼저 볼 항목 |
|---|---|---|
| 학습 중 loss가 갑자기 NaN/Inf | FP16 overflow/underflow | GradScaler 로그, max grad norm |
| 검증 정확도가 들쭉날쭉 | dtype 혼용(일부 FP32, 일부 FP16) | autocast 범위, 연산별 dtype |
| INT8 배포 후 특정 클래스만 급락 | calibration 분포 불일치 | calibration 샘플 클래스 분포 |
| 지연 시간은 줄었는데 메모리는 거의 동일 | activation은 여전히 FP16/FP32 | weight-only vs full quant 여부 |

## 5분 미니 실험: dtype별 오차 감각 잡기
아래 코드는 같은 텐서를 FP16/BF16으로 변환했다가 다시 FP32로 복원할 때,
평균 절대 오차(MAE)가 어느 정도인지 빠르게 확인하는 예시입니다.

```python
import torch


def reconstruction_mae(x: torch.Tensor, target_dtype: torch.dtype) -> float:
    x_q = x.to(target_dtype)
    x_hat = x_q.to(torch.float32)
    return (x - x_hat).abs().mean().item()


x = torch.randn(4096, dtype=torch.float32) * 3.0

mae_fp16 = reconstruction_mae(x, torch.float16)
mae_bf16 = reconstruction_mae(x, torch.bfloat16)

print(f"MAE(fp16): {mae_fp16:.6f}")
print(f"MAE(bf16): {mae_bf16:.6f}")
```

해석 포인트:
- 오차가 작을수록 원본 정보 보존이 잘 됩니다.
- 절대값이 큰 텐서/작은 텐서에서 오차 양상이 다를 수 있으니, 실제 feature 분포로도 반복해 보세요.

## 수식 → 코드 매핑 (양자화 4단계)
처음 양자화를 구현할 때는 아래 4단계를 고정 루틴으로 두면 실수가 크게 줄어듭니다.

1. **범위 수집(calibration)**
   - 수식 관점: $x_{\min}, x_{\max}$ 추정
   - 코드 관점: calibration 데이터로 텐서 최소/최대(또는 percentile) 집계

2. **스케일/제로포인트 계산**
   - 수식 관점: $s, z$ 계산
   - 코드 관점: dtype(INT8 등)에 맞춰 `qmin, qmax`를 놓고 scale/zero-point 산출

3. **양자화(실수→정수)**
   - 수식 관점: $q = \mathrm{round}(x/s) + z$
   - 코드 관점: round + clamp로 정수 범위 안에 고정

4. **복원/오차 확인(정수→실수)**
   - 수식 관점: $\hat{x} = s(q-z)$
   - 코드 관점: dequant 후 MAE/MSE 또는 task metric(top-1, mAP) 확인

초보자 포인트:
- 정확도 저하가 크면 1단계(범위 수집)부터 다시 봐야 합니다.
- 특히 배포 입력 분포가 calibration 분포와 다르면 2~4단계를 잘 구현해도 성능이 급락할 수 있습니다.

## 시각자료(나노바나나) 프롬프트
- **KO 다이어그램 1 (dtype 트레이드오프 맵)**  
  "다크 테마 배경(#1a1a2e), 데이터 타입 비교 인포그래픽. 축: x축 속도/메모리 효율, y축 정확도/안정성. FP32, FP16, BF16, INT8, INT4, FP8를 점으로 배치하고 각 점에 짧은 한국어 라벨(예: '기준 정밀도', '학습 안정', '배포 최적화'). 화살표로 '정밀도→효율' 트레이드오프 표시. 미니멀 벡터 스타일, 높은 가독성"

- **KO 다이어그램 2 (양자화 파이프라인 4단계)**  
  "다크 테마 배경(#1a1a2e), 4단계 양자화 파이프라인 도식. 1) Calibration(x_min/x_max) 2) Scale/Zero-point 계산 3) Quantize(round+clamp) 4) Dequantize 및 오차 측정(MAE/mAP). 각 단계 박스와 화살표, 한국어 레이블, 예시 수식 q=round(x/s)+z, x_hat=s(q-z) 포함. 깔끔한 교육용 벡터 스타일"

## 관련 콘텐츠
- [PTQ](/ko/docs/components/quantization/ptq)
- [QAT](/ko/docs/components/quantization/qat)
- [QLoRA](/ko/docs/components/training/peft/qlora)
