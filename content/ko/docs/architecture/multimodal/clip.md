---
title: "CLIP"
weight: 1
math: true
---

# CLIP (Contrastive Language-Image Pre-training)

## 개요

- **논문**: Learning Transferable Visual Models From Natural Language Supervision (2021)
- **저자**: Alec Radford et al. (OpenAI)
- **핵심 기여**: 이미지와 텍스트를 같은 공간에 임베딩하여 zero-shot 분류

## 핵심 아이디어

> "이미지와 텍스트 쌍을 대조 학습하여 공유 임베딩 공간 구축"

4억 개의 이미지-텍스트 쌍으로 학습하여, 한 번도 본 적 없는 클래스도 분류할 수 있습니다.

---

## 구조

### 전체 아키텍처

```
            ┌─────────────────┐
Image ────→ │  Image Encoder  │ ────→ Image Embedding (512-d)
            │  (ViT or ResNet)│                 ↓
            └─────────────────┘                 ↓ (cosine similarity)
                                                ↓
            ┌─────────────────┐                 ↓
Text  ────→ │   Text Encoder  │ ────→ Text Embedding (512-d)
            │  (Transformer)  │
            └─────────────────┘
```

### 인코더 옵션

| Image Encoder | Text Encoder |
|--------------|--------------|
| ViT-B/32, ViT-B/16, ViT-L/14 | Transformer (12-layer) |
| ResNet-50, ResNet-101 | |

---

## Contrastive Learning

### 학습 방식

배치 내 N개의 이미지-텍스트 쌍:

```
       Text1   Text2   Text3   ...   TextN
Image1  [✓]    [✗]     [✗]           [✗]
Image2  [✗]    [✓]     [✗]           [✗]
Image3  [✗]    [✗]     [✓]           [✗]
  ...
ImageN  [✗]    [✗]     [✗]           [✓]
```

- 대각선: 매칭 쌍 (positive) → 가깝게
- 비대각선: 비매칭 쌍 (negative) → 멀게

### 손실 함수

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)}$$

- $\text{sim}$: cosine similarity
- $\tau$: temperature (학습 가능)

### 대칭적 학습

Image→Text와 Text→Image 양방향으로 학습:

$$L_{total} = \frac{1}{2}(L_{I \to T} + L_{T \to I})$$

---

## Zero-shot Classification

학습 없이 새로운 클래스 분류:

```python
# 1. 클래스 이름을 텍스트 임베딩으로
classes = ["a photo of a dog", "a photo of a cat", ...]
text_embeddings = model.encode_text(classes)

# 2. 이미지 임베딩
image_embedding = model.encode_image(image)

# 3. 가장 유사한 클래스 선택
similarity = image_embedding @ text_embeddings.T
prediction = similarity.argmax()
```

### Prompt Engineering

분류 성능은 프롬프트에 따라 달라집니다:

```python
# 기본
"dog"

# 개선
"a photo of a dog"

# 더 개선 (앙상블)
templates = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a sculpture of a {}",
    ...
]
```

---

## 구현 예시

### 기본 사용

```python
import torch
import clip
from PIL import Image

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 전처리
image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device)

# 텍스트 토큰화
text = clip.tokenize(["a dog", "a cat", "a bird"]).to(device)

# 유사도 계산
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 정규화
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 유사도
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(similarity)  # [0.95, 0.03, 0.02]
```

### OpenCLIP (open source)

```python
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 사용법 동일
```

---

## 성능

### Zero-shot ImageNet

| Model | Top-1 Accuracy |
|-------|----------------|
| CLIP ViT-B/32 | 63.2% |
| CLIP ViT-B/16 | 68.3% |
| CLIP ViT-L/14 | 75.5% |
| CLIP ViT-L/14@336 | 76.2% |

지도 학습 없이도 상당한 성능을 달성합니다.

---

## CLIP의 활용

### 1. Image-Text Retrieval

```python
# 텍스트로 이미지 검색
query_embedding = model.encode_text("a red car")
similarities = query_embedding @ image_database.T
top_images = similarities.argsort(descending=True)[:10]
```

### 2. Stable Diffusion의 Text Encoder

```
Text Prompt → CLIP Text Encoder → Conditioning
```

### 3. 이미지 유사도 검색

```python
img1_feat = model.encode_image(img1)
img2_feat = model.encode_image(img2)
similarity = (img1_feat @ img2_feat.T).item()
```

### 4. CLIP Score (이미지-텍스트 정합성)

```python
def clip_score(image, text):
    img_feat = model.encode_image(image)
    txt_feat = model.encode_text(text)
    return (img_feat @ txt_feat.T).item()
```

---

## 한계점

- **세밀한 이해 부족**: 개수, 공간 관계, 텍스트 인식
- **편향**: 학습 데이터의 편향 반영
- **일부 도메인 약함**: 의료, 위성 등 특수 도메인

---

## 관련 콘텐츠

- [Contrastive Learning](/ko/docs/math/contrastive) - 수학적 기초
- [ViT](/ko/docs/architecture/transformer/vit) - Image encoder
- [VLM](/ko/docs/architecture/multimodal/vlm) - CLIP 발전형
- [Stable Diffusion](/ko/docs/architecture/generative/stable-diffusion) - CLIP 활용
