# FilterNet: A Filtered Gait Motion Fusion Network for Classifying Adult Spinal Deformity

<div align="center">
  📌 Under Submission to ACM MM Asia 2025
</div>

---

## Overview

**FilterNet** is a lightweight, phase-aware frame filtering framework designed to enhance spatiotemporal representation learning in periodic motion analysis. This project focuses on classifying adult spinal deformity (ASD) from gait videos by identifying diagnostically relevant frames and feeding them into a temporal classification model.

![workflow](img/training_step.png)

---

## Core Contributions

- 🔍 A two-stage pipeline that first scores and filters diagnostically salient frames based on gait phase segmentation.
- 🧠 Introduction of **FilterNet**, a frame-wise scoring model that enhances interpretability and model efficiency.
- 📈 Comprehensive evaluation on a real-world clinical gait dataset, achieving state-of-the-art results in ASD classification.

---

## Method Pipeline

```mermaid
graph LR
A[Input Video] --> B[Person Detection]
B --> C[Silhouette Extraction]
C --> D[Gait Cycle Estimation]
D --> E[Gait Phase Segmentation]
E --> F[FilterNet: Frame Scoring]
F --> G[Key Frame Selection]
G --> H[PhaseMix Classifier]
H --> I[Diagnosis Prediction]
```

---

## Dataset

The dataset includes 1,957 clips (2–10s each) from 81 patients with ASD, LCS_HipOA, and DHS, captured at 30fps in 1920x1080 resolution. Gait labels were annotated by expert spine surgeons.

---

## Training Procedure

### Step 1: Train FilterNet
- Input: Phase-specific video frames
- Supervision: Diagnosis labels
- Output: Frame-level diagnostic relevance scores (0–1)

### Step 2: Train Classifier
- Input: Top-T scored frames per gait phase
- Model: 3D CNN or PhaseMix
- Output: ASD / Control prediction

---

## How to Run

### Installation
```bash
git clone https://github.com/ChenKaiXuSan/FilterNet_ASD_PyTorch.git
cd FilterNet_ASD_PyTorch
pip install -e .
pip install -r requirements.txt
```

### Step-by-step
```bash
# Train FilterNet
python -m project.filter_train.main

# Inference: Generate frame scores
python -m project.filter_score.main

# Train classifier with filtered frames
python -m project.phasemix_main
```

> 💡 All modules are Hydra-compatible for configuration management.

---

## Docker Support

```bash
docker build -t filternet .
docker run -it filternet
cd /workspace/FilterNet_ASD_PyTorch
```

---

## Baseline Comparison

| Model | Accuracy | Precision | F1 Score |
|-------|----------|-----------|----------|
| CNN (no filter) | 52.56 | 81.11 | 54.01 |
| 3D CNN | 66.30 | 73.26 | 68.55 |
| PhaseMix + 3D CNN | 71.43 | 72.80 | 71.15 |
| **FilterNet → PhaseMix + 3D CNN** | **74.52** | **75.20** | **74.26** |

---

## Citation

**IEEE Access Version (PhaseMix)**
```bibtex
@ARTICLE{10714330,
  author={Chen, Kaixu and Xu, Jiayi and Asada, Tomoyuki and Miura, Kousei and Sakashita, Kotaro and Sunami, Takahiro and Kadone, Hideki and Yamazaki, Masashi and Ienaga, Naoto and Kuroda, Yoshihiro},
  journal={IEEE Access},
  title={PhaseMix: A Periodic Motion Fusion Method for Adult Spinal Deformity Classification},
  year={2024},
  volume={12},
  pages={152358--152376},
  doi={10.1109/ACCESS.2024.3479165}
}
```

**Frontiers in Neuroscience Version (Two-Stage CNN)**
```bibtex
@article{chen2023two,
  title={Two-stage video-based convolutional neural networks for adult spinal deformity classification},
  author={Chen, Kaixu and Asada, Tomoyuki and Ienaga, Naoto and Miura, Kousei and Sakashita, Kotaro and Sunami, Takahiro and Kadone, Hideki and Yamazaki, Masashi and Kuroda, Yoshihiro},
  journal={Frontiers in Neuroscience},
  volume={17},
  pages={1278584},
  year={2023},
  publisher={Frontiers Media SA}
}
```

---

## Acknowledgments
This project is supported by the University of Tsukuba and conducted under ethical approval (H30-087).