---
title: "Cross-Attention"
weight: 3
math: true
---

# Cross-Attention

{{% hint info %}}
**Prerequisites**: [Self-Attention](/en/docs/components/attention/self-attention)
{{% /hint %}}

## One-line Summary
> **Connects two different sequences (e.g., image and text), allowing one to selectively retrieve information from the other**

## Why is this needed?

### The Problem

In Stable Diffusion, suppose you want to generate "a cute cat wearing a hat." The model must handle two things simultaneously:

1. **Image**: The picture being generated (gradually becoming clearer from noise)
2. **Text**: The user's prompt "a cute cat wearing a hat"

Each region of the image needs to know which words in the text are relevant:
- Head area → focus on the word "hat"
- Face area → focus on "cat," "cute"
- Background area → no particular word to focus on

This is **Cross-Attention** — one world (image) **asking questions** to another world (text).

### Difference from Self-Attention

Self-Attention is a **monologue**. It finds relationships within itself.
Cross-Attention is a **dialogue**. It asks questions to another party and receives answers.

```
Self-Attention:
  Image → [Q, K, V all from image] → Internal relationships
  "This patch is similar to that patch"

Cross-Attention:
  Image → [Q from image], Text → [K, V from text]
  "Which text tokens are relevant to this image region?"
```

---

## Analogy: The Interpreter

{{< figure src="/images/components/attention/ko/cross-attention-image-text.png" caption="Cross-Attention: Connecting Image Patches to Text Tokens" >}}

Cross-Attention works like an **interpreter**.

| Role | Interpreter Analogy | Stable Diffusion Example |
|------|---------------------|--------------------------|
| **Q** (Question) | Question from a speaker | Each image region: "What should I draw here?" |
| **K** (Keywords) | Document titles/tags | Text tokens: "cute", "cat", "hat" |
| **V** (Content) | Actual document content | Real information in text embeddings |
| **Attention** | Interpreter finds and translates relevant docs | Matching image regions to text tokens |

---

## Formula

### Self-Attention (Review)

$$
\text{SelfAttn}(X) = \text{softmax}\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}}\right)(XW_V)
$$

Q, K, V **all come from the same X**.

### Cross-Attention

$$
\text{CrossAttn}(X, C) = \text{softmax}\left(\frac{(XW_Q)(CW_K)^T}{\sqrt{d_k}}\right)(CW_V)
$$

**Symbol meanings:**
- $X$: Target sequence (image features) — **source of Query**
- $C$: Conditioning sequence (text embeddings) — **source of Key, Value**
- $W_Q, W_K, W_V$: Learnable weight matrices
- $d_k$: Key dimension (for scaling)

**Key difference: Q comes from X, while K and V come from C.**

### Masked Form (Practical)

To exclude padding tokens, add a mask before softmax:

$$
\text{CrossAttn}(X, C, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

- $M$ : mask matrix (0 for valid tokens, very negative values for padding, e.g., $-10^4$ or $-\infty$)
- In practice, $M$ is commonly broadcast as $\mathbb{R}^{B \times 1 \times 1 \times M}$ and added to score tensor $(B, h, N, M)$.
- After softmax, padded positions get near-zero attention weights.

### Bool Mask Convention (standardize early)

Different PyTorch codebases use opposite bool meanings, which wastes debugging time. In Cross-Attention, it is safer to **normalize mask semantics first**.

- Accept either external convention: `True=valid` or `True=blocked`
- Convert internal convention to `True=keep`
- Use `~keep` in `masked_fill` for score masking

```python
def normalize_keep_mask(mask: torch.Tensor, *, true_means_keep: bool) -> torch.Tensor:
    """
    mask: (B, M) bool
    return: keep mask (B, M), True=allowed
    """
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    return mask if true_means_keep else ~mask

# Example: tokenizer provides True=padding(blocked)
raw_mask = padding_mask_bool                     # True=blocked
keep = normalize_keep_mask(raw_mask, true_means_keep=False)
score_mask = keep[:, None, None, :]              # (B,1,1,M)
scores = scores.masked_fill(~score_mask, torch.finfo(scores.dtype).min)
```

Keeping this one rule fixed across the team avoids many `dim`, `broadcast`, and `~mask` mistakes.

### Step-by-Step

```
Input:
  X (image): 4096 patches × 320 dimensions
  C (text):  77 tokens × 768 dimensions

Step 1: Q = X × W_Q  → (4096, 320)   "Image asks questions"
        K = C × W_K  → (77, 320)     "Text provides tags"
        V = C × W_V  → (77, 320)     "Text provides information"

Step 2: scores = Q × K^T  → (4096, 77)
        "How relevant is each of the 77 text tokens to each image patch?"

Step 3: attn = softmax(scores / √d_k)  → (4096, 77)
        "Convert relevance to probabilities"

Step 4: output = attn × V  → (4096, 320)
        "Inject text information into each image position"
```

**Note**: The attention matrix is (4096 × 77). Much smaller than Self-Attention's (4096 × 4096)! Cross-Attention is generally lighter than Self-Attention.

### Intuition Recap: Why is Q from image, but K/V from text?

- **Q (image)** is the question: "What should I draw at this location?"
- **K (text)** is the index: "Which semantic tag does this token represent?"
- **V (text)** is the actual content to inject.

So the image builds queries, while text provides answer candidates. That is why Cross-Attention behaves as a **conditioning injection layer**, not just a simple feature merge.

## Common Implementation Pitfalls

1. **Apply a text padding mask**
   - If sentence lengths differ within a batch, attention can leak into padding tokens.
   - Add a mask (usually `-inf`) to Cross-Attention scores so padded positions are excluded by softmax.

2. **Keep the $\sqrt{d_k}$ scaling**
   - If you remove scaling, score variance grows and softmax becomes too sharp, making training unstable.
   - The issue is stronger when head dimension is large (e.g., 128).

3. **Check softmax stability in mixed precision**
   - In FP16/BF16, combining large negative masks with large scores can produce NaNs.
   - In practice, many implementations cast scores to FP32 for softmax, then cast back.

## Quick Shape Sanity Checklist

When debugging, printing these three lines often catches most bugs quickly:

- `q.shape == (B, h, N, d)`
- `k.shape == (B, h, M, d)`
- `mask.shape == (B, 1, 1, M)` (if mask is used)

If these are correct, the score tensor should become `(B, h, N, M)` automatically. Then softmax must be applied on the last axis `M`, so each image position properly chooses among text tokens.

## Softmax Axis Check (Common Bug)

One of the most frequent Cross-Attention bugs is using the wrong softmax axis.

```python
# Correct: normalize over text axis (M)
attn = F.softmax(scores, dim=-1)   # scores: (B, h, N, M)

# Wrong example: normalize over image axis (N)
bad = F.softmax(scores, dim=-2)
```

- With `dim=-1`: each image position chooses **which text tokens** to attend to.
- With `dim=-2`: interpretation flips toward "which image positions" per token, which can break intended conditioning behavior.

Quick checks:
- `attn.sum(dim=-1)` should be close to 1.
- Mean attention on masked text positions should stay near 0.

## Practical: Combining Causal + Padding Masks

In decoder cross-attention / hybrid blocks, you may need both masks at once.
The key is to **align mask shape to score shape, then apply one boolean mask**.

```python
# scores: (B, h, N, M)
# causal_mask:  (N, M) bool, True=allowed / False=blocked
# padding_mask: (B, M) bool, True=valid / False=padding

allow = causal_mask[None, None, :, :] & padding_mask[:, None, None, :]
# allow: (B, 1, N, M) -> broadcastable to scores

scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
```

- Printing `allow.shape` before `masked_fill` and checking `(B, 1, N, M)` catches many shape bugs.
- Using `torch.finfo(dtype).min` is usually safer than hardcoded `-1e9` in FP16/BF16 paths.

## Complexity Comparison

Let's compare Cross-Attention with Self-Attention:

| Operation | Self-Attention | Cross-Attention |
|-----------|----------------|-----------------|
| **Attention matrix** | $N \times N$ | $N \times M$ |
| **FLOPs (QK^T)** | $O(N^2 \cdot d)$ | $O(N \cdot M \cdot d)$ |
| **Memory** | $O(N^2)$ | $O(N \cdot M)$ |

**Stable Diffusion example** (64×64 latent, 77 CLIP tokens, d=320):

| | Self-Attention | Cross-Attention | Ratio |
|---|---|---|---|
| **Attention matrix** | 4096 × 4096 = 16.8M | 4096 × 77 = 315K | **53× smaller** |
| **Memory (FP16)** | 33.6 MB | 630 KB | **53× less** |
| **FLOPs** | ~10.7 GFLOPs | ~200 MFLOPs | **53× less** |

→ Cross-Attention is much lighter than Self-Attention. In many U-Net blocks, Self-Attention dominates attention cost.

### Dimension Mismatch Handling

{{< figure src="/images/components/attention/ko/cross-attention-dimension-gradient.jpeg" caption="Cross-Attention projection (768→320) and gradient flow — gradients to CLIP are blocked when CLIP is frozen" >}}

A practical challenge is when two sequences have different feature dimensions:

```
Image: (B, 4096, 320)   — 320 dims
Text:  (B, 77, 768)     — 768 dims (CLIP)

Q = X × W_Q ∈ ℝ^(320×320)  → (B, 4096, 320)
K = C × W_K ∈ ℝ^(768×320)  → (B, 77, 320)     ← 768→320 projection
V = C × W_V ∈ ℝ^(768×320)  → (B, 77, 320)     ← 768→320 projection

→ W_K and W_V project text space (768) into image-attention space (320)
→ Q and K must share the same last dimension d_k for dot product
```

This projection is one of the key learnable parts of Cross-Attention: learning a **shared representation space** across modalities.

### Gradient Flow

Gradients in Cross-Attention flow to both branches:

```
Forward:
  X → W_Q → Q ─┐
                ├→ Attention Score → Output → Loss
  C → W_K → K ─┘
  C → W_V → V ─────────────────────┘

Backward:
  ∂Loss/∂W_Q ← through Q (image-side query learning)
  ∂Loss/∂W_K ← through K (text→image matching learning)
  ∂Loss/∂W_V ← through V (content injection learning)

  ∂Loss/∂X ← gradient flows to image features
  ∂Loss/∂C ← gradient flows to text features (if text encoder is trainable)
```

**Important**: In Stable Diffusion, CLIP text encoder is usually **frozen**. So gradients wrt $C$ may exist in computation, but CLIP parameters are not updated. The trainable parts are mostly U-Net-side attention projections.

---

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """Cross-Attention: Connecting two sequences"""

    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)

        # Q from target (image), K/V from condition (text)
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)   # context → query dim
        self.to_v = nn.Linear(context_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, context, context_mask=None):
        """
        x:            (B, N, query_dim)   - image features (the asker)
        context:      (B, M, context_dim) - text embeddings (the answerer)
        context_mask: (B, M) bool, True=valid token / False=padding
        """
        B, N, C = x.shape

        # Q from image, K/V from text
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: which text token does each image region focus on?
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, M)

        # Exclude padding tokens (mask=False) from softmax
        if context_mask is not None:
            mask = context_mask[:, None, None, :]  # (B,1,1,M)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # Compute softmax in FP32 for mixed-precision stability
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)

        # Inject text info into image
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# Usage: Stable Diffusion's Cross-Attention
image_features = torch.randn(4, 4096, 320)  # 64x64 latent, 320 dim
text_embeddings = torch.randn(4, 77, 768)   # CLIP text (77 tokens, 768 dim)

cross_attn = CrossAttention(
    query_dim=320,     # image dimension
    context_dim=768,   # text dimension (CLIP)
    num_heads=8
)
out = cross_attn(image_features, text_embeddings)
print(out.shape)  # torch.Size([4, 4096, 320])
```

---

## Real-World Use Cases

### 1. Stable Diffusion — Text-to-Image Generation

Cross-Attention operates inside Stable Diffusion's U-Net:

```
Prompt: "a cute cat wearing a hat"

Image generation process:
  ┌─────────────────────────────┐
  │ U-Net Block                 │
  │   ├── Self-Attention        │  Internal image relationships (patch to patch)
  │   ├── Cross-Attention       │  Text ↔ Image connection
  │   └── FFN                   │  Non-linear transformation
  └─────────────────────────────┘

In Cross-Attention:
  Head region → focuses on "hat" → draws a hat
  Face region → focuses on "cat", "cute" → draws cute cat face
  Overall    → focuses on "wearing" → wearing pose
```

### 2. VLM (Vision-Language Models) — Visual Q&A

```
Question: "What is the red object in this photo?"

In Cross-Attention:
  Question tokens (Query)  ×  Image features (Key, Value)
  "red" → focuses on red regions in image
  "object" → identifies the meaning of that region
  → Answer: "It's a fire truck"
```

### 3. Transformer Decoder — Translation

```
Source (English): "I love cats"
Target (Korean): "나는 고양이를 좋아한다"

In Cross-Attention:
  Translation tokens (Query)  ×  Source encoding (Key, Value)
  "고양이를" → focuses on "cats"
  "좋아한다" → focuses on "love"
```

---

## Cross-Attention Map Visualization

Visualizing Cross-Attention weights shows "which text tokens correspond to which parts of the image."

```python
import matplotlib.pyplot as plt

def visualize_cross_attention(attn_map, text_tokens, image_size=(64, 64)):
    """
    Visualize Cross-Attention Map

    attn_map:    (num_heads, N_image, N_text) - attention weights
    text_tokens: list of text tokens (e.g., ["a", "cute", "cat", "hat"])
    image_size:  H, W of image (in latent space)
    """
    # Average across all heads
    attn = attn_map.mean(0)  # (N_image, N_text)
    H, W = image_size

    fig, axes = plt.subplots(1, len(text_tokens), figsize=(3*len(text_tokens), 3))
    for i, token in enumerate(text_tokens):
        ax = axes[i] if len(text_tokens) > 1 else axes
        # Attention from each image region for this text token
        ax.imshow(attn[:, i].reshape(H, W), cmap='hot')
        ax.set_title(token, fontsize=12)
        ax.axis('off')

    plt.suptitle('Cross-Attention Map: Text → Image Correspondence', fontsize=14)
    plt.tight_layout()
    plt.show()

# Usage
# attn_map = ...  # attention weights extracted from model
# visualize_cross_attention(attn_map, ["a", "cute", "cat", "wearing", "hat"])
```

This visualization lets you verify:
- Whether "cat" token has high values in the cat region of the image
- Whether "hat" token corresponds to the hat region

---

## Common Failure Patterns and Debugging

### 1. Attention Collapse — All Queries focus on one Key

```
Normal:
  Head region  → "hat"   (0.7)
  Face region  → "cat"   (0.6)
  Background   → near-uniform (0.15, 0.15, ...)

Collapsed:
  Head region  → "cat"   (0.9)
  Face region  → "cat"   (0.9)
  Background   → "cat"   (0.8)
  → Almost every region attends to a single token
```

**Likely causes**:
- Too large learning rate in early training
- Abnormally large norm for a specific text token embedding

**Quick diagnostic**: monitor attention entropy.

```python
import torch

def attention_entropy(attn_weights):
    """Entropy of attention distributions (higher = more spread)."""
    # attn_weights: (B, heads, N, M)
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)
    return entropy.mean()

# Rule of thumb (77-token text):
# Normal: entropy around 2-4
# Collapse: entropy around 0.1-0.5
```

### 2. Text-Image Mismatch — Prompt not reflected

If you ask for "a red car" but get a blue car, one frequent cause is weak attention on the token "red".
In diffusion pipelines, increasing CFG (Classifier-Free Guidance) often strengthens text conditioning.

### 3. Dimension Mismatch Bug

```python
# Common mistake: mixing context_dim and query_dim
cross_attn = CrossAttention(
    query_dim=320,      # image dim
    context_dim=320,    # ❌ wrong for CLIP text (usually 768)
    num_heads=8
)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (77x768 and 320x320)
```

---

## Self vs Cross Attention Comparison

| | Self-Attention | Cross-Attention |
|---|---|---|
| **Q source** | Self (X) | Target sequence (X) |
| **K, V source** | Self (X) | Conditioning sequence (C) |
| **Attention size** | N × N | N × M (usually smaller) |
| **Purpose** | Learn internal relationships | Incorporate external conditions |
| **Analogy** | Monologue | Dialogue (Q&A) |
| **Key models** | ViT, GPT | Stable Diffusion, VLM |

---

## Summary

| Question | Answer |
|----------|--------|
| How does it differ from Self? | Q from A, K/V from B → connects two sequences |
| Why is it needed? | Connects different modalities (image ↔ text) |
| Where is it used? | Stable Diffusion, VLM, Translation |
| Computation? | N×M (usually lighter than Self-Attention's N×N) |

## Related Content

- [Self-Attention](/en/docs/components/attention/self-attention) — Foundation of Cross-Attention
- [Positional Encoding](/en/docs/components/attention/positional-encoding) — Injecting position information
- [Layer Normalization](/en/docs/components/normalization/layer-norm) — Transformer block component
- [Stable Diffusion](/en/docs/architecture/generative/stable-diffusion) — Primary Cross-Attention use case
- [CLIP](/en/docs/architecture/multimodal/clip) — Foundation for image-text connection
