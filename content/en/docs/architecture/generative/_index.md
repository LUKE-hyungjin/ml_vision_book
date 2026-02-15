---
title: "Generative"
weight: 6
bookCollapseSection: true
---

# Generative Models

> **Prerequisites**: [Probability Distribution](/en/docs/math/probability/distribution) | [VAE](/en/docs/architecture/generative/vae) | [GAN](/en/docs/architecture/generative/gan)

## Why is this needed?
The goal of a generative model is not to classify data, but to learn patterns in data and **create new samples**.
Like a painter who studies many artworks and then creates a new painting, the model learns a data distribution and samples from it.

## Formula and symbols
Generative modeling often learns a model distribution $p_\theta(x)$ that approximates the real data distribution $p_{\text{data}}(x)$.

$$
\theta^* = \arg\min_\theta D\big(p_{\text{data}}(x) \;\|\; p_\theta(x)\big)
$$

- $x$: observed data (e.g., images)
- $\theta$: model parameters
- $p_{\text{data}}(x)$: true data distribution
- $p_\theta(x)$: learned model distribution
- $D(\cdot\|\cdot)$: distribution distance/divergence (e.g., KL, JS, Wasserstein)

## Intuition: three major families
{{< figure src="/images/generative/en/generative-comparison.svg" caption="Comparison of VAE vs GAN vs Diffusion" >}}

| Family | Core idea | Intuition |
|--------|-----------|-----------|
| **VAE** | Compress to latent, then reconstruct | Rebuild from a compact summary |
| **GAN** | Generator-discriminator adversarial game | Counterfeiter vs detective |
| **Diffusion** | Remove noise step by step | Progressive photo restoration |

## Implementation path (recommended order)
1. [VAE](/en/docs/architecture/generative/vae): start with latent-variable modeling
2. [GAN](/en/docs/architecture/generative/gan): classic high-fidelity generation
3. [DDPM](/en/docs/architecture/generative/ddpm): modern diffusion foundation
4. [Stable Diffusion](/en/docs/architecture/generative/stable-diffusion): practical text-to-image standard
5. [Flux](/en/docs/architecture/generative/flux): recent flow-matching line

## Timeline (short)
- 2013: VAE
- 2014: GAN
- 2018: StyleGAN
- 2020: DDPM
- 2021: VQGAN, DALL-E
- 2022: Stable Diffusion
- 2023: ControlNet, DiT
- 2024: Flux

## Related content
- [Generative math components](/en/docs/components/generative)
- [Flow Matching](/en/docs/components/generative/flow-matching)
- [Probability Distribution](/en/docs/math/probability/distribution)
- [Generation task](/en/docs/task/generation)
