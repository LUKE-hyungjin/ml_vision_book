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

{{< figure src="/images/components/attention/en/cnn-vs-attention.png" caption="CNN vs Attention: Limited View vs Global View" >}}

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

**Symbol meanings:**
- $Q$ (Query): query vector representing what the current token is looking for
- $K$ (Key): key vector representing what each token contains
- $V$ (Value): value vector carrying the actual information to aggregate
- $d_k$: key dimension used for scaling

**Tensor shapes (single-head):**
- input $X$: $(B, N, d)$
- $Q, K, V$: $(B, N, d_k)$
- score matrix $QK^T$: $(B, N, N)$
- output $Z$: $(B, N, d_v)$

Here, $B$ is batch size and $N$ is sequence length (number of tokens).

### Intuition from a Single Token's View

For a single token at position $i$, Self-Attention can be summarized in one line:

$$
z_i = \sum_{j=1}^{N} \alpha_{ij} v_j, \qquad
\alpha_{ij}=\frac{\exp(s_{ij})}{\sum_{t=1}^{N}\exp(s_{it})}, \qquad
s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}
$$

- $\alpha_{ij}$: how much token $i$ attends to token $j$
- $z_i$: the updated, context-aware representation of token $i$

So, **token $i$ updates itself by taking a weighted average of all Value vectors**.

It looks complex, but breaks down into 3 simple steps:

{{< figure src="/images/components/attention/en/self-attention-score-matrix.png" caption="Attention Score Matrix — Similarity between each word pair" >}}

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

→ Looking at the "the mat" row: The cat(0.3), sat(0.2), on(0.4), the mat(0.5)
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

A quick variance intuition:

$$
\mathrm{Var}(q \cdot k)=\sum_{j=1}^{d_k} \mathrm{Var}(q_j k_j)=d_k
$$

If each component is i.i.d. with mean 0 and variance 1, dot-product variance grows with $d_k$.
Scaling by $\sqrt{d_k}$ normalizes it back near 1, preventing softmax saturation.

### Step 3: Softmax + Value Weighted Sum

$$
\text{output} = \text{softmax}(\text{scaled\_scores}) \times V
$$

- softmax: Converts each row to probabilities (sum = 1)
- Probability × V: **More information from important words is reflected**

```
"the mat" attention weights: [0.35, 0.15, 0.20, 0.30]

output = 0.35 × V(The cat) + 0.15 × V(sat) + 0.20 × V(on) + 0.30 × V(the mat)
→ A context-aware representation focused on "the mat"
```

### Tiny Numerical Example (hand-calculation intuition)

A tiny case ($N=2$, $d_k=2$) makes the flow concrete.

$$
q_i=[1,2],
\quad
k_1=[1,0],\;k_2=[1,2],
\quad
v_1=[2,0],\;v_2=[0,4]
$$

1) Scores:
$$
s_{i1}=\frac{q_i\cdot k_1}{\sqrt{2}}=\frac{1}{\sqrt{2}},
\qquad
s_{i2}=\frac{q_i\cdot k_2}{\sqrt{2}}=\frac{5}{\sqrt{2}}
$$

2) Softmax weights (approx.):
$$
[\alpha_{i1},\alpha_{i2}] \approx [0.06,\;0.94]
$$

3) Output vector:
$$
z_i=\alpha_{i1}v_1+\alpha_{i2}v_2
\approx 0.06[2,0]+0.94[0,4]
\approx [0.12,\;3.76]
$$

So the output is dominated by $v_2$, because the corresponding key $k_2$ got the higher score.

### Attention Mask (Causal / Padding)

In practice, a token cannot always attend to every position. We add an **attention mask** to the score matrix to separate
"allowed positions" from "blocked positions."
The mask is applied **in score space before softmax**.

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

**Additional symbols:**
- $M$: mask matrix
- allowed position: $M_{ij}=0$
- blocked position: $M_{ij}=-\infty$ (implemented as a very large negative number, e.g., -1e9)

Adding the mask before softmax makes blocked positions get near-zero probability.

- **Causal mask**: token $i$ cannot see future tokens ($j>i$), used in autoregressive generation
- **Padding mask**: prevents attention to padding tokens in batched inputs

### Practical Mask Composition Tip (Causal + Padding)

In production, both masks are usually applied together. The most common bug is a **shape-broadcast mismatch**.

- attention score shape: `(B, heads, N, N)`
- recommended padding mask shape: `(B, 1, 1, N)`
- recommended causal mask shape: `(1, 1, N, N)`

Combine them once (OR or additive), then apply before softmax.

```python
# Assume True means "blocked"
combined_mask = padding_mask | causal_mask
scores = scores.masked_fill(combined_mask, torch.finfo(scores.dtype).min)
attn = torch.softmax(scores, dim=-1)
```

### Safely Handling Mask Conventions (True=keep vs True=block)

Different frameworks/codebases use different boolean-mask semantics:
- some use `True=keep (allowed)`
- others use `True=block (masked out)`

A tiny normalization helper prevents many train/serve mismatches.

```python
def normalize_bool_mask(mask_bool: torch.Tensor, true_means_keep: bool) -> torch.Tensor:
    """Returns mask in a unified convention: True=blocked."""
    return ~mask_bool if true_means_keep else mask_bool

# Example: external mask follows True=keep
block_mask = normalize_bool_mask(ext_mask_bool, true_means_keep=True)
scores = scores.masked_fill(block_mask, torch.finfo(scores.dtype).min)
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

        # (Optional) apply padding/causal mask: blocked positions -> very negative
        # scores = scores.masked_fill(mask == 0, float('-inf'))

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

## Practical Implementation Checklist

Self-Attention formulas are compact, but small implementation mistakes are common. Check these four items first.

1. **Apply mask before softmax**
   - Correct order: `scores + mask → softmax → dropout → V`
   - If you multiply mask after softmax, probability normalization is broken.

2. **Use `dim=-1` for softmax**
   - Normalize along the key axis (last dimension of attention scores).
   - Wrong axis means you normalize over queries, not "which tokens to attend to."

3. **Keep softmax numerically stable**
   - In manual implementations, subtract the row-wise maximum from `scores` before softmax to reduce overflow risk.
   - Example: `attn = torch.softmax(scores - scores.max(dim=-1, keepdim=True).values, dim=-1)`

4. **Separate train vs inference behavior**
   - Training: attention dropout enabled (`model.train()`)
   - Inference: dropout disabled (`model.eval()`)

5. **Plan for quadratic memory growth**
   - As sequence length or resolution grows, $N^2$ memory grows quickly.
   - For long contexts, prefer [Flash Attention](/en/docs/components/attention/flash-attention). For local structure, consider [Window Attention](/en/docs/components/attention/window-attention).

### Quick Debug Smoke Test

When training is unstable, printing these 3 checks together helps narrow the issue quickly.

```python
with torch.no_grad():
    # 1) Row sum should be 1 (checks softmax axis)
    row_sum = attn_weights.sum(dim=-1).mean().item()

    # 2) NaN / Inf checks
    has_nan = torch.isnan(attn_weights).any().item()
    has_inf = torch.isinf(attn_weights).any().item()

    # 3) Over-concentration check (mean row-wise max)
    peak = attn_weights.max(dim=-1).values.mean().item()

print(f"row_sum≈{row_sum:.4f}, nan={has_nan}, inf={has_inf}, peak={peak:.4f}")
```

- If `row_sum` significantly deviates from 1, check softmax axis and mask order first.
- If `peak` is too high (e.g., 0.98+), inspect scale, learning rate, and warmup.

## Common Failure Patterns

When Self-Attention becomes unstable in production training, checking these patterns first usually narrows the root cause quickly.

1. **Attention collapse to a single token**
   - Symptom: attention maps look nearly one-hot and validation quality plateaus.
   - Check: temperature/scale setup, learning rate, gradient norm, warmup length.

2. **Attention leakage into padding tokens**
   - Symptom: performance becomes unstable across variable-length batches, especially with short inputs.
   - Check: padding-mask broadcast shape and whether `-inf` is applied on the correct axis before softmax.

3. **Reversed causal-mask direction**
   - Symptom: autoregressive models peek at future tokens, causing train/serve mismatch.
   - Check: upper/lower triangular direction and framework-specific mask conventions (True=keep vs True=block).

### Mask Constant Choice in FP16/BF16

In mixed-precision training, using a hardcoded large negative value like `-1e9` can cause unexpected `-inf`/`nan` propagation after dtype conversion.
A safer pattern is to fill blocked positions with the **minimum finite value of the current dtype**.

```python
# Use a mask value that matches scores dtype/device
mask_value = torch.finfo(scores.dtype).min
scores = scores.masked_fill(mask == 0, mask_value)
attn = torch.softmax(scores, dim=-1)
```

- Key point: choose mask constants based on `scores.dtype`
- Debug tip: if NaNs appear under AMP, check mask constant + mask-before-softmax order first

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

- [Multi-Head Attention](/en/docs/components/attention/multi-head-attention) — Analyze multiple relationship patterns in parallel
- [Cross-Attention](/en/docs/components/attention/cross-attention) — Connects two different sequences
- [Positional Encoding](/en/docs/components/attention/positional-encoding) — Injects order information
- [Window Attention](/en/docs/components/attention/window-attention) — Efficient local-window attention
- [Flash Attention](/en/docs/components/attention/flash-attention) — GPU memory optimization for long sequences
- [Layer Normalization](/en/docs/components/normalization/layer-norm) — Essential Transformer block component
- [ViT](/en/docs/architecture/transformer/vit) — Self-Attention applied to images
