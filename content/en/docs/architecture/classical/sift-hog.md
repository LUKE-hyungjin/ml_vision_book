---
title: "SIFT & HOG"
weight: 1
math: true
---

# SIFT & HOG

{{% hint info %}}
**Prerequisites**: [Matrix](/en/docs/math/linear-algebra/matrix) | [Gradient](/en/docs/math/calculus/gradient) | [Conv2D](/en/docs/components/convolution/conv2d)
{{% /hint %}}

## One-line Summary
> **SIFT and HOG are classical feature extractors that convert images into robust numeric patterns before a classifier/detector makes final decisions.**

## Why are these needed?
Before CNNs became standard, we still needed ways to detect objects and match images.
SIFT/HOG solved this with hand-crafted visual cues that remain stable under common variations.

Analogy:
- **SIFT**: mark landmarks on a map that remain useful after zoom/rotation.
- **HOG**: summarize “which edge directions appear most” in each local region.

They are still useful when:
- labeled data is limited,
- an interpretable pipeline is required,
- a CPU-friendly baseline is needed.

---

## SIFT (Scale-Invariant Feature Transform)

### Core idea
Find repeatable keypoints under scale/rotation changes, then describe each keypoint with a 128-D descriptor.

### Pipeline
1. Build scale space with Gaussian blur
2. Compute DoG (Difference of Gaussians)
3. Detect extrema as keypoint candidates
4. Assign dominant orientation
5. Build 128-D descriptor

### Formula
Gaussian scale space:
$$
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
$$

Difference of Gaussian:
$$
D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)
$$

**Symbol meanings:**
- $I(x,y)$: input image intensity
- $G(x,y,\sigma)$: Gaussian kernel at scale $\sigma$
- $L(x,y,\sigma)$: blurred image at scale $\sigma$
- $k$: multiplicative scale step (e.g., $\sqrt{2}$)
- $D(x,y,\sigma)$: DoG response for keypoint detection

### Practical strengths / limits
- Strength: robust to scale and rotation
- Limit: relatively expensive for large real-time pipelines

---

## HOG (Histogram of Oriented Gradients)

### Core idea
Split the image into cells, then represent each cell by a histogram of gradient directions.

### Pipeline
1. Compute horizontal/vertical gradients
2. Split image into cells (typically 8x8)
3. Build orientation histograms (often 9 bins)
4. Normalize over local blocks (typically 2x2 cells)
5. Concatenate into one feature vector

### Formula
$$
G = \sqrt{G_x^2 + G_y^2}, \qquad
\theta = \arctan\left(\frac{G_y}{G_x}\right)
$$

**Symbol meanings:**
- $G_x, G_y$: x/y gradients
- $G$: gradient magnitude (edge strength)
- $\theta$: gradient direction angle

### Practical strengths / limits
- Strength: lightweight and strong for edge-heavy objects (e.g., pedestrians)
- Limit: sensitive to large scale changes without multi-scale search

---

## Minimal implementation (OpenCV)

```python
import cv2

# 1) SIFT keypoints + descriptors
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
print(f"#keypoints: {len(keypoints)}, descriptor shape: {descriptors.shape if descriptors is not None else None}")

# 2) HOG pedestrian detector baseline
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(img)
print(f"detections: {len(boxes)}")
```

---

## Practical debugging checklist
- [ ] Is preprocessing (grayscale/resize/normalization) consistent between train and inference?
- [ ] Is resolution too small (losing keypoints/edges) or too large (slow + noisy)?
- [ ] Are HOG `winStride` and `scale` overly aggressive?
- [ ] For SIFT matching, did you apply Lowe’s ratio test?
- [ ] Are confidence threshold + NMS applied before final detections?

## Common mistakes (FAQ)
**Q1. Why do I get very few SIFT keypoints?**  
A. The image may be low-texture, blurry, or too low-contrast. Adjust image quality/preprocessing and detector thresholds.

**Q2. Why does HOG produce many false positives?**  
A. Sliding-window detectors often confuse background patterns. First tune `detectMultiScale` params and NMS/confidence thresholds.

**Q3. Are SIFT/HOG still worth learning?**  
A. Yes. They are weaker than modern deep models on top benchmarks, but still valuable for intuition, debugging, and low-resource baselines.

## Symptom → likely cause quick map
| Observed symptom | Most common cause | First thing to inspect |
|---|---|---|
| Extremely few SIFT matches | low texture / heavy blur | image quality, contrast, keypoint thresholds |
| Too many HOG boxes | threshold/NMS not tuned | confidence cutoff, NMS settings |
| Large drop after scale change | single-scale setup | image pyramid, multi-scale scan |
| More false positives under lighting shift | inconsistent preprocessing | normalization / histogram equalization |

---

## SIFT/HOG vs CNN (quick comparison)

| Aspect | SIFT/HOG | CNN |
|---|---|---|
| Feature design | hand-crafted | learned from data |
| Data requirement | low to medium | medium to high |
| Accuracy ceiling | limited | generally higher |
| Interpretability | high | relatively lower |
| Deployment | CPU-friendly baseline | higher quality but heavier compute |

## Related Content
- [Conv2D](/en/docs/components/convolution/conv2d)
- [YOLO](/en/docs/architecture/detection/yolo)
- [Faster R-CNN](/en/docs/architecture/detection/faster-rcnn)
