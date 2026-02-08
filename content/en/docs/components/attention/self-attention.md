---
title: "Self-Attention"
weight: 1
math: true
---

# Self-Attention

{{% hint info %}}
**Prerequisites**: [Matrix](/en/docs/math/linear-algebra/matrix) | [Calculus Basics](/en/docs/math/calculus/basics)
{{% /hint %}}

## One-line Summary
> **Computes relationships between all position pairs in a sequence, updating each position's representation into a "context-aware representation"**

## Why is this needed?

### The Problem

When understanding the sentence "The cat sat on the mat," we naturally think:
- "sat" → **who?** → "The cat" (subject-verb relationship)
- "sat" → **where?** → "on the mat" (location relationship)

This is exactly what **Self-Attention** does. Each word in a sentence looks at every other word and figures out "which words are relevant to me?"

### Why can't CNNs do this?

A CNN kernel is a **fixed-size window**. A 3x3 kernel only sees 3 adjacent elements.

```
Sentence: [The cat] [sat] [on] [the mat]

CNN (kernel size 3):
"the mat" can see: [sat, on, the mat]
→ "The cat" is not visible!
→ Need to stack deeper layers to see distant words

Self-Attention:
"the mat" can see: [The cat, sat, on, the mat] — everything!
→ Captures all relationships in a single step
```

**Key difference**: CNNs see "only nearby," Self-Attention sees "everything at once."

---

## QKV: The Library Analogy

The best analogy for Self-Attention is **searching for books in a library**.

{{< figure src="/images/components/attention/ko/cnn-vs-attention.png" caption="CNN vs Attention: Limited View vs Global View" >}}

| Role | Library Analogy | Meaning |
|------|----------------|---------|
| **Q** (Query) | Search query — "Find me books about cats" | "What am I looking for?" |
| **K** (Key) | Book titles/tags — "This book is about animals" | "What information do I contain?" |
| **V** (Value) | Book content — the actual text | "The actual information to deliver" |

**The search process:**
1. **Compare Query with Keys** → "How relevant is this book to my search?" (similarity score)
2. **Softmax** → Convert scores to probabilities (sum to 1)
3. **Probability × Value** → Pull more content from highly relevant books

---

## Formula: Step by Step

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

It looks complex, but breaks down into 3 simple steps:

{{< figure src="/images/components/attention/ko/self-attention-score-matrix.png" caption="Attention Score Matrix — Similarity between each word pair" >}}

### Step 1: Similarity Calculation — $QK^T$

$$
\text{scores} = QK^T
$$

- $Q$: (sequence length N) × (dimension d) matrix
- $K^T$: (dimension d) × (sequence length N) matrix
- Result: (N × N) matrix — **similarity between all position pairs**

```
Example: 4-word sentence

          The cat  sat   on   the mat
The cat  [ 0.8    0.1   0.0   0.3 ]
sat      [ 0.1    0.7   0.5   0.2 ]
on       [ 0.0    0.5   0.6   0.4 ]
the mat  [ 0.3    0.2   0.4   0.5 ]

→ Looking at the "sat" row: The cat(0.3), sat(0.2), on(0.4), the mat(0.5)
```

### Step 2: Scaling — $\div \sqrt{d_k}$

$$
\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}
$$

**Why divide?**

When $d_k$ is large, dot product values also become large. If values are too large, softmax concentrates 99% on one position — an "extreme" distribution. This causes gradients to approach 0, and learning stops.

```
When d_k = 64:
  Before scaling: [32, 1, 2]  → softmax → [1.00, 0.00, 0.00] (nearly one-hot)
  After scaling:  [4, 0.125, 0.25] → softmax → [0.95, 0.02, 0.03] (smooth distribution)
```

Dividing by $\sqrt{d_k}$ keeps the variance near 1.

### Step 3: Softmax + Value Weighted Sum

$$
\text{output} = \text{softmax}(\text{scaled\_scores}) \times V
$$

- softmax: Converts each row to probabilities (sum = 1)
- Probability × V: **More information from important words is reflected**

```
"sat" attention weights: [0.35, 0.15, 0.20, 0.30]

output = 0.35 × V(The cat) + 0.15 × V(sat) + 0.20 × V(on) + 0.30 × V(the mat)
→ A new representation with context: "The cat sat on the mat"
```

---

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """Single-head Self-Attention (for understanding)"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Weight matrices to create Q, K, V from input
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        # x: (Batch, Sequence Length, Embedding Dim)
        Q = self.W_q(x)  # "What am I looking for?"
        K = self.W_k(x)  # "What do I contain?"
        V = self.W_v(x)  # "Here's the actual info"

        # Step 1: Similarity calculation
        scores = Q @ K.transpose(-2, -1)  # (B, N, N)

        # Step 2: Scaling + softmax
        attn_weights = F.softmax(scores / self.scale, dim=-1)  # (B, N, N)

        # Step 3: Weighted sum
        output = attn_weights @ V  # (B, N, embed_dim)

        return output

# Usage example
x = torch.randn(2, 5, 64)      # batch 2, sequence 5, dim 64
attn = SelfAttention(64)
out = attn(x)                   # (2, 5, 64) — context-aware representations
print(out.shape)                # torch.Size([2, 5, 64])
```

---

## Multi-Head Attention

### Why Multiple Heads?

When one person analyzes a sentence, they see only one perspective. But **multiple experts** analyzing from different angles produce a richer understanding.

<!-- TODO: Multi-Head Attention image coming soon -->

For example, in "The cat sat on the mat":
- **Head 1**: Grammar → Subject of "sat" is "The cat"
- **Head 2**: Location → Place of "sat" is "on the mat"
- **Head 3**: Semantic similarity → "cat" and "mat" are related

### Formula

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

- Each head performs Attention independently
- Results are concatenated, then combined via linear transform ($W^O$)

### Implementation

```python
class MultiHeadSelfAttention(nn.Module):
    """Practical Multi-Head Self-Attention"""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 768 // 8 = 96

        # Compute Q, K, V in one shot (efficient)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape

        # Compute Q, K, V then split into heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention (independently per head)
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum, then merge heads
        out = (attn @ v)                               # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)     # (B, N, embed_dim)

        return self.proj(out)

# Usage: ViT-Base configuration
x = torch.randn(32, 196, 768)    # batch 32, 14x14 patches, dim 768
mha = MultiHeadSelfAttention(768, num_heads=12)
out = mha(x)                      # (32, 196, 768)
```

**Parameter count comparison:**

| Component | Calculation | Parameters |
|-----------|-----------|------------|
| QKV projection | 768 × (768×3) | 1,769,472 |
| Output projection | 768 × 768 | 589,824 |
| **Total** | | **~2.4M** |

---

## Computational Complexity

### The $O(N^2 \cdot d)$ Problem

The core operation in Self-Attention is $QK^T$. Comparing all N position pairs means $N^2$.

| Input | N | N² | Memory (FP32) |
|-------|---|----|----|
| Sentence (100 tokens) | 100 | 10,000 | ~40 KB |
| ViT (14×14 patches) | 196 | 38,416 | ~150 KB |
| High resolution (32×32) | 1,024 | 1,048,576 | ~4 MB |
| Very long sequence | 16,384 | 268M | ~1 GB |

As N grows, memory increases **quadratically**. This is why long sequences are challenging.

### Solutions

| Method | Core Idea | Complexity |
|--------|-----------|------------|
| **Flash Attention** | GPU memory optimization (tiling) | $O(N^2)$ but 2-4x faster in practice |
| **Window Attention** | Compute only within local windows (Swin Transformer) | $O(N \cdot W^2)$ |
| **Linear Attention** | Approximation via kernel trick | $O(N \cdot d)$ |

### Flash Attention Usage

```python
# PyTorch 2.0+ (most practical solution)
from torch.nn.functional import scaled_dot_product_attention

# Automatically uses Flash Attention on GPU
# Mathematically identical results, just memory-efficient
attn_output = scaled_dot_product_attention(q, k, v)
```

Flash Attention produces mathematically identical results while reducing memory from $O(N^2)$ to $O(N)$. Speed also improves 2-4x.

---

## Summary

| Question | Answer |
|----------|--------|
| What does it do? | Computes relationships between all positions in a sequence |
| Why is it needed? | Captures global relationships beyond CNN's fixed window |
| What are Q, K, V? | Search query, tags, actual content (library analogy) |
| Why divide by $\sqrt{d_k}$? | Prevents values from growing too large, stabilizes gradients |
| Why Multi-Head? | Analyzes simultaneously from multiple perspectives |
| Downside? | $O(N^2)$ complexity, solved by Flash Attention |

## Related Content

- [Cross-Attention](/en/docs/components/attention/cross-attention) — Connects two different sequences
- [Positional Encoding](/en/docs/components/attention/positional-encoding) — Injects order information
- [Layer Normalization](/en/docs/components/normalization/layer-norm) — Essential Transformer block component
- [ViT](/en/docs/architecture/transformer/vit) — Self-Attention applied to images
