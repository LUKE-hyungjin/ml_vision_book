---
title: "Faster R-CNN"
weight: 1
math: true
---

# Faster R-CNN

{{% hint info %}}
**Prerequisites**: [Conv2D](/en/docs/components/convolution/conv2d) · [Anchor Box](/en/docs/components/detection/anchor) · [NMS](/en/docs/components/detection/nms)
{{% /hint %}}

## One-line Summary
> **Faster R-CNN is a representative two-stage detector that combines learned region proposals (RPN) with precise classification and box refinement.**

## Why this model?
Earlier detectors relied on external proposal algorithms (such as Selective Search),
which were slow and difficult to optimize end-to-end.

Faster R-CNN internalized proposal generation with a neural module (**RPN**),
so both proposal generation and final detection can be trained in one framework.

## Overview

- **Paper**: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015)
- **Authors**: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- **Key Contribution**: End-to-end trainable detector with Region Proposal Network (RPN)

## R-CNN Lineage

```
R-CNN (2014)
    ↓ Introduced ROI Pooling
Fast R-CNN (2015)
    ↓ Introduced RPN
Faster R-CNN (2015)
```

| Model | Region Proposal | Speed |
|------|------------------|------|
| R-CNN | Selective Search | ~47s/image |
| Fast R-CNN | Selective Search | ~2s/image |
| Faster R-CNN | RPN (learned) | ~0.2s/image |

---

## Architecture

### Overall Pipeline

```
Input Image
    ↓
Backbone (ResNet/VGG) → Feature Map
    ↓
┌───────────────────────────────────┐
│      Region Proposal Network      │
│  Feature Map → Anchors → RoIs     │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│         RoI Pooling/Align         │
│ Variable-size RoI → fixed features│
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│     Detection Head (R-CNN Head)   │
│  Classification + Box Regression  │
└───────────────────────────────────┘
    ↓
Output: [class, x, y, w, h]
```

---

## Key Components

### 1. Backbone

Extracts feature maps from the image:
- VGG-16: used in the original paper
- ResNet-50/101: common in practice
- FPN: often added to improve multi-scale detection

### 2. Region Proposal Network (RPN)

Generates candidate object regions from feature maps:

```
Feature Map (H×W×C)
    ↓
3×3 Conv
    ↓
┌─────────────────────────────────┐
│  1×1 Conv (cls)    1×1 Conv (reg) │
│  2k scores        4k coordinates  │
└─────────────────────────────────┘
```

**Anchor Boxes:**
- k anchors per spatial location (default: 9)
- 3 scales × 3 aspect ratios = 9 anchors
- scales: 128², 256², 512²
- ratios: 1:1, 1:2, 2:1

### 3. RoI Pooling / RoI Align

Converts variable-size RoIs into fixed-size features (e.g., 7×7):

- **RoI Pooling**: has quantization-induced misalignment
- **RoI Align**: avoids quantization with bilinear interpolation (popularized by Mask R-CNN)

### 4. Detection Head

For each RoI:
- Classification: predicts class via softmax
- Box Regression: refines box coordinates

---

## Training

### Multi-task Loss

$$L = L_{cls} + \lambda L_{reg}$$

**Classification Loss (Cross-entropy):**
$$L_{cls} = -\log p_{c^*}$$

**Regression Loss (Smooth L1):**
$$L_{reg} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t_i - t_i^*)$$

### Box Parameterization

$$t_x = (x - x_a) / w_a, \quad t_y = (y - y_a) / h_a$$
$$t_w = \log(w / w_a), \quad t_h = \log(h / h_a)$$

### Training Strategy

1. Alternating optimization of RPN and R-CNN head (original paper)
2. End-to-end joint training (common in modern frameworks)

---

## Implementation Example

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pretrained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference
with torch.no_grad():
    predictions = model([image])

# predictions[0]['boxes']: (N, 4)
# predictions[0]['labels']: (N,)
# predictions[0]['scores']: (N,)
```

### Train with Custom Classes

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 10  # including background

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

---

## Performance (COCO)

| Backbone | mAP | mAP@50 |
|----------|-----|--------|
| VGG-16 | 21.9 | 42.7 |
| ResNet-50 | 35.7 | 57.0 |
| ResNet-101 | 37.4 | 58.8 |
| ResNet-50-FPN | 37.0 | 58.5 |

---

## Limitations

- Two-stage pipeline can be slower than one-stage detectors
- Small-object detection can still be difficult (partially improved by FPN)
- Anchor design often requires domain-specific tuning

## Beginner Debugging Checklist
- [ ] **Check proposal quality**: many proposals but low IoU with GT?
- [ ] **Review NMS threshold**: too low causes over-suppression; too high keeps duplicates
- [ ] **Verify coordinate scales in RoI stage**: avoid mixing image-space and feature-map-space coordinates
- [ ] **Track cls/reg loss balance**: ensure one loss is not dominating the whole optimization

## Common Mistakes (FAQ)
**Q1. If I increase the number of proposals, will performance always improve?**  
A. Not always. Recall may improve, but computation and false positives can increase. Tune proposal count gradually.

**Q2. Is Faster R-CNN always more accurate than YOLO?**  
A. Not always. It depends on data, backbone, and training setup. Two-stage methods often localize well, but modern one-stage models can also achieve strong accuracy.

**Q3. mAP@50 is okay but mAP@75 is low. Why?**  
A. This usually indicates localization precision issues rather than classification. Check anchor settings, box-regression learning rate, and RoI Align configuration.

---

## Related Content

- [YOLO](/en/docs/architecture/detection/yolo) - One-stage detector
- [Mask R-CNN](/en/docs/architecture/segmentation/mask-rcnn) - Faster R-CNN + segmentation
- [ResNet](/en/docs/architecture/cnn/resnet) - Common backbone
- [IoU](/en/docs/components/detection/iou) & [NMS](/en/docs/components/detection/nms) - Core detection concepts
