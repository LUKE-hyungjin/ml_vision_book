---
title: "Cross-Attention"
weight: 2
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
        self.scale = math.sqrt(self.head_dim)

        # Q from target (image), K/V from condition (text)
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)   # context → query dim
        self.to_v = nn.Linear(context_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        """
        x:       (B, N, query_dim)   - image features (the asker)
        context: (B, M, context_dim) - text embeddings (the answerer)
        """
        B, N, C = x.shape

        # Q from image, K/V from text
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: which text token does each image region focus on?
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, M)
        attn = F.softmax(attn, dim=-1)

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
