---
title: "Attention"
weight: 5
bookCollapseSection: true
math: true
---

# Attention

> **Core idea: "Look at everything, but focus more on what matters"**

## Why Attention?

Think about reading a book. Do you read every sentence with the same level of focus? No. You **concentrate on important parts** and quickly skim over less important ones. Highlighting key passages before an exam works the same way.

Neural networks are no different. When analyzing an image, they should **focus on objects** rather than the background. When understanding a sentence, not every word is equally important. Learning "where and how much to focus" is what Attention does.

### Limitations of CNNs

CNNs scan images with a small window (kernel). This is like a **magnifying glass with a fixed size**:

| Problem | Description |
|---------|-------------|
| **Fixed field of view** | A 3x3 kernel always sees only 3x3 |
| **Hard to capture global relationships** | Can't see left and right edges at once, requires deep stacking |
| **Fixed weights** | Same filter applied regardless of input |

### Advantages of Attention

Attention is like a **camera that can freely zoom in and out**:

| Advantage | Description |
|-----------|-------------|
| **Dynamic weights** | Connections change based on input — "this is important, let me look closer" |
| **Global interaction** | Can reference all positions at once |
| **Interpretability** | Can visualize where the model looks (Attention Map) |

## Core Concepts

| Concept | One-line Description | Key Models |
|---------|---------------------|------------|
| [Self-Attention](/en/docs/components/attention/self-attention) | Finds relationships within itself | Transformer, ViT |
| [Cross-Attention](/en/docs/components/attention/cross-attention) | Connects two different information sources | Stable Diffusion, VLM |
| [Positional Encoding](/en/docs/components/attention/positional-encoding) | Injects order information | All Transformer-based models |

## Reading Order

```
Self-Attention → Cross-Attention → Positional Encoding
```

Self-Attention is the foundation. Understanding it first makes the rest follow naturally.

## Related Content

**Prerequisite Math:**
- [Matrix](/en/docs/math/linear-algebra/matrix) — Foundation for Q, K, V operations
- [Probability Distributions (Softmax)](/en/docs/math/probability/distribution) — Computing attention weights

**Components Used Together:**
- [Layer Normalization](/en/docs/components/normalization/layer-norm) — Transformer block building

**Architectures Using This Concept:**
- [ViT](/en/docs/architecture/transformer/vit) — Applying Self-Attention to images
- [CLIP](/en/docs/architecture/multimodal/clip) — Connecting images and text
- [Stable Diffusion](/en/docs/architecture/generative/stable-diffusion) — Text conditioning via Cross-Attention
