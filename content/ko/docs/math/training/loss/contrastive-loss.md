---
title: "Contrastive Loss"
weight: 3
math: true
---

# Contrastive Loss

## 개요

Contrastive Loss는 유사한 샘플은 가깝게, 다른 샘플은 멀게 임베딩하도록 학습합니다. CLIP, SimCLR 등 표현 학습의 핵심입니다.

## 기본 아이디어

```
같은 클래스/쌍 → 임베딩 거리 최소화
다른 클래스/쌍 → 임베딩 거리 최대화
```

## InfoNCE Loss (CLIP, SimCLR)

$$
L = -\log \frac{\exp(sim(z_i, z_j^+) / \tau)}{\sum_{k=1}^{N} \exp(sim(z_i, z_k) / \tau)}
$$

- **z_i**: 앵커 임베딩
- **z_j^+**: 긍정 샘플 (같은 것의 다른 뷰)
- **z_k**: 모든 샘플 (긍정 + 부정)
- **τ**: Temperature (보통 0.07)
- **sim**: 코사인 유사도

### 직관

- 분자: 긍정 쌍의 유사도를 높이려 함
- 분모: 다른 모든 것과 구별하려 함
- Softmax 형태의 분류 문제로 볼 수 있음

## 구현

```python
import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.07):
    """
    features: (2N, D) - N쌍의 augmented views
    첫 N개와 뒤 N개가 각각 쌍
    """
    batch_size = features.shape[0] // 2

    # 정규화
    features = F.normalize(features, dim=1)

    # 유사도 행렬 (2N, 2N)
    similarity = features @ features.T / temperature

    # 자기 자신 마스킹
    mask = torch.eye(2 * batch_size, device=features.device).bool()
    similarity.masked_fill_(mask, float('-inf'))

    # 긍정 쌍 인덱스
    # [0,1], [2,3], ... 또는 [0,N], [1,N+1], ...
    labels = torch.arange(batch_size, device=features.device)
    labels = torch.cat([labels + batch_size, labels])  # 쌍 인덱스

    loss = F.cross_entropy(similarity, labels)
    return loss
```

## CLIP Loss

이미지-텍스트 쌍에 대한 양방향 contrastive:

```python
def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds: (N, D)
    text_embeds: (N, D)
    i번째 이미지와 i번째 텍스트가 쌍
    """
    # 정규화
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)

    # 유사도 (N, N)
    logits = image_embeds @ text_embeds.T / temperature

    # 정답: 대각선 (i번째 이미지 ↔ i번째 텍스트)
    labels = torch.arange(len(image_embeds), device=logits.device)

    # 양방향 손실
    loss_i2t = F.cross_entropy(logits, labels)      # 이미지 → 텍스트
    loss_t2i = F.cross_entropy(logits.T, labels)    # 텍스트 → 이미지

    return (loss_i2t + loss_t2i) / 2
```

## Triplet Loss

앵커, 긍정, 부정 세 샘플 사용:

$$
L = \max(0, d(a, p) - d(a, n) + margin)
$$

```python
def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()
```

## Temperature의 역할

- **낮은 τ (0.01)**: 날카로운 분포, 어려운 부정 샘플 집중
- **높은 τ (1.0)**: 부드러운 분포, 모든 샘플 균등 고려
- **일반적**: 0.07 ~ 0.1

## 관련 콘텐츠

- [CLIP](/ko/docs/architecture/multimodal/clip)
- [Cross-Entropy](/ko/docs/math/training/loss/cross-entropy)
- [Self-supervised Learning](/ko/docs/task/self-supervised)
