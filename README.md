# CIFAR-10 Image Classification -- Term 9 Sprint 3
**Masterschool MSIT Program | Computer Vision**

A full end-to-end image classification pipeline built with PyTorch and ResNet50, exploring transfer learning, layer-selective fine-tuning, progressive dataset scaling, and from-scratch custom architecture on the CIFAR-10 dataset (10 classes, 32x32 RGB images).

---

## Project Overview

This project was completed in five experimental phases, progressing from a baseline transfer learning setup through extended experiments and a fully custom architecture built without pretrained weights.

**Presentation recording:** [CIFAR-10 Sprint 3 -- Live Presentation](https://youtu.be/o95iweh0TKE)

---

## Environment

| Component | Details |
|-----------|---------|
| Language | Python 3.11.9 |
| Framework | PyTorch 2.5.1+cu121 |
| GPU | NVIDIA RTX 2060 6GB (CUDA) |
| IDE | VSCode + Jupyter Notebooks |
| OS | Windows 11 |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `cifar10_main.ipynb` | Core pipeline: EDA, preprocessing, ResNet50 transfer learning (S7-S9) |
| `cifar10_experiements.ipynb` | Extended experiments: progressive dataset scaling, layer-selective fine-tuning (S10-S11) |
| `cifar10_zerostart.ipynb` | From-scratch custom architecture: native 32x32 ResNet50 variant, no pretrained weights |

---

## Results Summary

| Model | Training Data | Test Accuracy |
|-------|--------------|---------------|
| S7 -- Head Only (transfer) | 10k | 78.02% |
| S8a -- Full Unfreeze (transfer) | 10k | 89.49% |
| S8b -- Layer4 Only (transfer) | 10k | 85.62% |
| S10 -- Head Only (transfer) | 40k | 81.78% |
| S11-R1 -- Layer3 Only (transfer) | 28k | 92.87% |
| S11-R3 -- Layer3+4 Stacked (transfer) | 28k | 92.85% |
| Phase 3b -- From Scratch (native 32x32) | 50k | 81.17% |

---

## Key Findings

- **Layer selection outperforms data scaling** -- unfreezing layer3 on 28k data outperformed head-only training on 40k data by over 11 percentage points.
- **ImageNet pretraining is worth ~11.7%** -- the gap between the best from-scratch result (81.17%) and the best transfer result (92.87%) under equivalent dataset conditions quantifies the value of pretrained priors.
- **Low-resolution upsampling (32->224px) creates artifacts** that mid-level spatial filters (layer3) partially compensate for -- native 32x32 processing eliminates this at the cost of pretrained weight compatibility.
- **Class collapse** (cat, deer, horse at 0% accuracy) was observed in early from-scratch phases with small datasets, resolving only at 50k training scale.
- **Gradient monitoring** (Phase 5) confirmed that the stem layer is the most actively adapting component in native 32x32 training, with gradient magnitude 8-10x higher than layer1 throughout training.

---

## Repository Structure

```
cifar10-cv-term9/
├── cifar10_main.ipynb             # Core pipeline (S1-S9)
├── cifar10_experiements.ipynb     # Extended experiments (S10-S11)
├── cifar10_zerostart.ipynb        # From-scratch architecture (Phase 5)
├── config.py                      # Shared configuration
├── phase-5-from-scratch-experiment.md  # Phase 5 design document
├── CIFAR10-Presentation.pdf       # Sprint presentation (PDF)
├── CIFAR10-Presentation.pptx      # Sprint presentation (PPTX)
└── [training artifacts: .png, checkpoints]
```

---

## Future Work

- **S13:** Progressive unfreezing (layer3 -> layer3+4 -> full) at full 50k scale using pretrained weights
- **Gradient-informed selective retraining:** Use Phase 5 gradient monitoring data to identify and selectively retrain high-gradient layers, bridging transfer learning and from-scratch findings