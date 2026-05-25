# Phase 5 - From Scratch Experiment

##### CIFAR-10 Phase 5

### Zero-Start Experiment Design Document

**Date:** 26.05.2026  
**Notebook:** cifar10\_zerostart.ipynb  
**Repo:** zachary4001/cifar10-cv-term9  
**Author:** Zachary | Prepared with Claude (Anthropic)

---

## 1.0 Context &amp; Motivation

All prior experiments (S7-S11) used ResNet50 with ImageNet pretrained weights and 32-&gt;224px upsampling. Phase 4's key finding: upsampling introduces interpolation artifacts that mid-level spatial filters (layer3) partially compensate for -- but the artifact is architectural, not solvable by fine-tuning alone.

Phase 5 removes both constraints entirely:

- No pretrained weights -- random Kaiming initialization
- No upsampling -- native 32x32 input processing
- Custom architecture designed specifically for small inputs

**Primary comparison target:** S11-R1 (92.87% test accuracy, layer3 unfrozen, 35k dataset, ImageNet pretrained)

---

## 2.0 The Stem Block Problem

Standard ResNet50 opens with `Conv2d(3, 64, kernel_size=7, stride=2)` followed by `MaxPool2d(stride=2)`. This aggressively halves spatial dimensions twice before any residual block fires -- appropriate for 224x224 inputs (50,176 pixels), destructive for 32x32 inputs (1,024 pixels).

On 32x32 inputs, the standard stem collapses spatial context to a 4x4 feature map before layer1 begins. Residual blocks receive insufficient spatial information to extract meaningful features.

**Decision: Replace stem with small-input-friendly design**

- `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`
- `BatchNorm2d(64)` + `ReLU`
- No MaxPool
- **Reasoning:** Preserves spatial context into layer1, allowing residual blocks to operate on meaningful feature maps.

---

## 3.0 Architecture Decisions

### 3.1 Block Type -- Bottleneck

**Decision:** Bottleneck blocks (1x1 -&gt; 3x3 -&gt; 1x1) preserved from ResNet50 convention.  
**Reasoning:** Maintains 2048-dimensional output from layer4, allowing the custom head to carry over unchanged from prior experiments. Preserves architectural comparability.

### 3.2 Block Configuration -- \[2, 2, 2, 2\]

**Decision:** 2 Bottleneck blocks per layer group (8 total), vs. true ResNet50's \[3, 4, 6, 3\] (16 total).  
**Reasoning:** Training from scratch on a small dataset without pretrained priors makes full ResNet50 depth high-risk for overfitting and slow convergence. A lighter configuration is more appropriate for the native 32x32 scale. A separate deeper experiment remains viable as future work.

### 3.3 Layer Naming

**Decision:** `layer1`, `layer2`, `layer3`, `layer4` -- matching prior experiment conventions.  
**Reasoning:** Enables direct code reuse and conceptual comparison across phases.

### 3.4 Channel Progression

Matching ResNet50 conventions:

- layer1: 64 -&gt; 256
- layer2: 128 -&gt; 512
- layer3: 256 -&gt; 1024
- layer4: 512 -&gt; 2048

### 3.5 Custom Head -- Identical to Prior Experiments

`Linear(2048,128) -> ReLU -> Linear(128,64) -> ReLU -> Dropout(0.2) -> Linear(64,10)`  
No Softmax -- CrossEntropyLoss handles it internally.

### 3.6 Kaiming (He) Initialization

**Decision:** Applied to all `Conv2d` and `Linear` layers at init time.  
**Reasoning:** Kaiming initialization calibrates random weight scale specifically for ReLU activations, accounting for ReLU's tendency to suppress ~50% of neurons. Without it, deep from-scratch networks frequently fail to learn in early epochs due to vanishing or exploding gradients.

### 3.7 Dropout in Bottleneck Blocks

**Decision:** `dropout=0.1` added after the final BatchNorm in each Bottleneck block.  
**Reasoning:** From-scratch training on small datasets is significantly more vulnerable to overfitting than fine-tuning -- there are no pretrained priors to act as implicit regularizers. Light dropout within blocks provides additional regularization without suppressing the network's learning capacity.

---

## 4.0 Training Strategy

### 4.1 Full Simultaneous Training -- All Layers Unfrozen

**Decision:** All layers train simultaneously from the first epoch in every phase.  
**Reasoning:** Freezing is a transfer learning concept -- it preserves pretrained knowledge while adapting new layers. Without pretrained weights, frozen layers contain only random Kaiming values, producing arbitrary projections with no semantic content. Training all layers simultaneously allows the full hierarchy to co-adapt through backpropagation, each layer receiving a gradient signal derived from the actual classification objective.

Sequential layer training (layer1/2 first, then layer3/4) was considered but rejected: layers trained in isolation have no task-connected loss signal, producing representations misaligned with the final objective. This is known as representational interference when later unfreezing occurs.

### 4.2 Phased Dataset Progression

**Decision:** Three sequential phases with increasing dataset size, each continuing from the prior checkpoint.

<table id="bkmrk-phase-dataset-optimi"><thead><tr><th>Phase</th><th>Dataset</th><th>Optimizer</th><th>Learning Rate</th><th>MIN\_DELTA</th><th>PATIENCE</th></tr></thead><tbody><tr><td>1</td><td>10k (8k train / 2k val)</td><td>Adam</td><td>1e-3</td><td>0.01</td><td>3</td></tr><tr><td>2</td><td>35k (28k train / 7k val)</td><td>AdamW</td><td>1e-4</td><td>0.005</td><td>3</td></tr><tr><td>3</td><td>50k (40k train / 10k val)</td><td>AdamW</td><td>1e-5</td><td>0.005</td><td>3</td></tr></tbody></table>

**Reasoning:** Progressive dataset scaling serves multiple purposes:

- Phase 1 establishes baseline behavior quickly with minimal compute, catching architectural problems early
- Phase 2 introduces more signal with a refined learning rate -- AdamW's weight decay adds regularization as dataset complexity increases
- Phase 3 uses the full dataset with a conservative learning rate to refine without destabilizing established representations
- Learning rate decay across phases mirrors the network's convergence state -- larger steps when far from optimum, smaller steps as representations stabilize
- MIN\_DELTA=0.01 in Phase 1 stops training earlier during the noisiest learning phase; relaxed to 0.005 in Phases 2/3 as training becomes more stable

### 4.3 Early Stopping

- `copy.deepcopy` for correct best weight capture (learned from S11-R1 correction)
- Checkpoint reload + val\_loss verification cell after each phase

---

## 5.0 Data Configuration

### 5.1 Normalization

**Decision:** CIFAR-10 native statistics.  
`mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`  
**Reasoning:** Prior experiments used ImageNet stats to match pretrained weight expectations. With no pretrained weights, ImageNet stats are incorrect for this data distribution. CIFAR-10 native stats correctly center and scale the actual pixel distribution.

### 5.2 Data Augmentation -- Training Only

**Decision:** `RandomHorizontalFlip` + `RandomCrop(32, padding=4)` applied to training transforms only.

- `RandomHorizontalFlip` -- randomly mirrors images left-to-right, forcing orientation-invariant feature learning
- `RandomCrop(32, padding=4)` -- pads to 36x36, randomly crops back to 32x32, forcing position-invariant feature learning. Padding=4 represents a 12.5% positional shift -- sufficient for variety without risking feature truncation on already-small images.
- Transforms compose: an image may be flipped, cropped, both, or neither in a given epoch

**Val/Test transforms:** `ToTensor()` + normalize only -- no augmentation.  
**Reasoning:** Val/test sets must remain static for reliable early stopping signals and valid benchmark comparisons against prior experiments.

### 5.3 DataLoader Settings

- `batch_size=64` -- larger than prior experiments (32); appropriate for from-scratch training and makes better use of available GPU memory
- `num_workers=8` -- increased from prior 6; CPU was underutilized at 6 workers
- `pin_memory=True`, `persistent_workers=True` -- retained from prior experiments

---

## 6.0 Evaluation

Each phase includes:

- `evaluate_model()` -- overall test loss and accuracy
- Per-class accuracy breakdown for all 10 classes
- Best epoch checkpoint verification (reload + val\_loss match)
- Markdown cell for hypothesis, observation, and interpretation

Final comparison table includes all three phases plus S11-R1 (92.87%) as the primary baseline.

---

## 7.0 Key Hypotheses

1. **S-Scratch Phase 1 will significantly underperform S7 (78.02%)** -- randomly initialized features carry no semantic structure; the head has nothing meaningful to classify from
2. **Phase 2/3 progressive training will close the gap with S11-R1** -- native 32x32 processing eliminates upsampling artifacts, potentially compensating for the absence of ImageNet priors
3. **Cat/deer/dog classes will show the largest relative improvement** -- these classes were most affected by upsampling artifact interference in Phase 4, per per-class analysis

---

## 8.0 Comparison Baselines

<table id="bkmrk-model-input-weights-"><thead><tr><th>Model</th><th>Input</th><th>Weights</th><th>Dataset</th><th>Test Acc</th></tr></thead><tbody><tr><td>S7 Head Only</td><td>224px upsampled</td><td>ImageNet</td><td>10k</td><td>78.02%</td></tr><tr><td>S11-R1 Layer3</td><td>224px upsampled</td><td>ImageNet</td><td>35k</td><td>92.87%</td></tr><tr><td>S-Scratch Phase 1</td><td>32px native</td><td>None</td><td>10k</td><td>TBD</td></tr><tr><td>S-Scratch Phase 2</td><td>32px native</td><td>None</td><td>35k</td><td>TBD</td></tr><tr><td>S-Scratch Phase 3</td><td>32px native</td><td>None</td><td>50k</td><td>TBD</td></tr></tbody></table>

---

## 9.0 Future Work

- **S-Scratch-4 (conditional):** Deeper block configuration \[3,4,6,3\] if Phase 3 shows meaningful headroom -- isolates depth as a variable
- **Greedy layer-wise pretraining:** Historical technique (Hinton et al., ~2006) -- train layers sequentially using auxiliary objectives. Rejected for this experiment due to task-signal problem, but remains a valid exploratory direction
- **Augmentation expansion:** Additional augmentations (ColorJitter, Cutout) if overfitting persists in Phase 3

---

*Document prepared: 26.05.2026 | Phase 5 design session*