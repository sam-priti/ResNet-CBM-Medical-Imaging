# 🏥 ResNet-CBM: Concept-Based Reasoning for Accountable Medical Diagnosis

> An interpretable AI system for chest X-ray classification using a ResNet-50 Concept Bottleneck Model (CBM) — achieving 87.43% accuracy with full clinical auditability.

---

## Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Why CBM over Black-Box Models?](#why-cbm)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

This project implements a **Concept Bottleneck Model (CBM)** built on a **ResNet-50 backbone** for interpretable, accountable diagnosis of chest X-ray images. It classifies images into 4 classes:

- 🟢 Normal
- 🔴 COVID-19
- 🟡 Lung Opacity
- 🟠 Viral Pneumonia

Unlike black-box AI systems, the CBM forces all predictions to flow through a human-interpretable **concept bottleneck layer**, providing codified, linguistic justification for every diagnosis — critical for clinical adoption and regulatory compliance.

```
Input X-Ray (224×224)
        ↓
ResNet-50 Encoder → Global Average Pooling → [2048-dim features]
        ↓
Concept Bottleneck Layer (64 nodes)
        ↓
Classifier Head → Final Prediction (4 Classes)
```

---

## Key Results

| Metric | Result | Interpretation |
|--------|--------|----------------|
| **Accuracy** | 87.43% | Strong multi-class performance |
| **F1-Score (Weighted)** | 0.8734 | Balanced across imbalanced classes |
| **AUC (One-vs-Rest)** | 0.9754 | Excellent discrimination ability |
| **Clinical Audit Rate (CAR)** | 17.40% ✅ | Low = high accountability |

> **CAR (Clinical Audit Rate):** Measures the % of wrong predictions made with deceptive high confidence (≥90%). A **low CAR means 82.6% of errors were made with low confidence** — making them non-deceptive and fully auditable.

---

## Architecture

### ResNet-CBM (3-Stage Pipeline)

**Stage 1 — Feature Extractor:**
- ResNet-50 backbone with ImageNet pre-trained weights
- Global Average Pooling condenses features to a 2048-dim vector
- Transfer learning for efficient training on medical data

**Stage 2 — Concept Bottleneck (64 nodes):**
- Linear compression from 2048 → 64 dimensions
- Forces the model to encode only the most salient diagnostic information
- Each node represents an interpretable concept (e.g., consolidation, opacity)

**Stage 3 — Classifier Head:**
- Linear mapping from 64 concept scores → 4-class prediction
- Fully auditable: diagnosis = linear function of human-readable concepts

### Why Not Post-Hoc XAI (Grad-CAM, LIME, SHAP)?
Post-hoc methods suffer from the **Abstraction Gap** — they explain *where* the model looked, not *why* it made the decision. The CBM resolves this by making the reasoning process intrinsically transparent.

---

## Dataset

**COVID-19 Radiography Database** (Kaggle)
- Modality: Chest X-Ray (CXR)
- Classes: Normal (~20.4k), COVID (~732), Lung Opacity (~1.2k), Viral Pneumonia (~2.7k)
- Split: 80% Train / 10% Val / 10% Test (stratified sampling)

> ⚠️ Dataset not included in repo. Download from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

---

## Project Structure

```
ResNet-CBM-Medical-Imaging/
├── cbm_model.ipynb        # Main Colab notebook (full pipeline)
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── report.pdf             # Full project report
└── checkpoints/           # Saved model weights (auto-created)
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/sam-priti/ResNet-CBM-Medical-Imaging.git
cd ResNet-CBM-Medical-Imaging
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
1. Go to: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. Download and extract
3. Update `DATA_DIR` in the notebook config

### 4. Run on Google Colab (Recommended)
Upload `cbm_model.ipynb` to [Google Colab](https://colab.research.google.com/) and run all cells.

---

## Usage

### Configuration (inside notebook)
```python
class Config:
    DATA_DIR = '/content/drive/MyDrive/COVID-19_Radiography_Dataset'
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-5
    NUM_CONCEPTS = 64       # Concept bottleneck size
    NUM_CLASSES = 4
```

### Training
```python
trained_model, history = train_cbm(cbm_model, train_loader, val_loader)
```

### Evaluation
```python
results = evaluate_model(cbm_model, test_loader, history)
# Returns: accuracy, f1, auc, confusion matrix, CAR
```

---

## Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall correct predictions
- **F1-Score (Weighted)**: Handles class imbalance
- **AUC (One-vs-Rest)**: Multi-class discrimination

### Custom Metric: Clinical Audit Rate (CAR)
```
CAR = (High-confidence wrong predictions / Total wrong predictions) × 100
```
- High confidence = model confidence ≥ 90%
- **Lower CAR = more trustworthy model**
- CBM achieved: **17.40% CAR** ✅

---

## Why CBM?

| Model | Accuracy | Trust Metric | Audit Type |
|-------|----------|--------------|------------|
| **CBM (This repo)** | 87.43% (4-class) | CAR: 17.40% ✅ | Codified concept logic |
| PGCDA | ~92% (binary) | ECS: 94% | Visual prototypes |
| Uncertainty Hybrid | 97.4% (3-class) | AUC: 0.996 | Post-hoc (Grad-CAM) |
| D-FAE (Rule-based) | 55% | RSS: 0.39% | Feature importance |

The CBM is selected as the **best final solution** because:
- ✅ Intrinsically interpretable (no post-hoc approximation)
- ✅ Regulatory compliant (FDA/CE Mark alignment)
- ✅ Handles harder 4-class task
- ✅ Lowest Clinical Audit Rate = most structurally accountable

---

## Future Work

- **Concept Annotation**: Map 64 concept nodes to clinical terms (e.g., "Ground-Glass Opacity", "Consolidation") with radiologist collaboration
- **Federated Learning**: Train across decentralized hospital datasets for GDPR compliance
- **Uncertainty Fusion**: Integrate Bayesian uncertainty into the concept space
- **Concept Intervention**: Allow physicians to manually adjust concept scores and observe diagnosis shifts

---

## References

- Koh, P. W., et al. (2020). *Concept Bottleneck Models.* ICML.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Rudin, C. (2019). *Stop explaining black box ML models for high stakes decisions.* Nature Machine Intelligence.
- COVID-19 Radiography Database — Kaggle.

---

## Team

This project was developed as part of **Deep Learning (22CSE619)** coursework.

| Member | Role |
|--------|------|
| **Sampriti Mohanty** | CBM Implementation & Systematic Interpretability |
| Shreyansh Gaur | PGCDA & Explanation Coherence Score |
| Shubhanshu Singh Patel | High-Accuracy Uncertainty Hybrid Model |
| Tanishq Katoch | D-FAE & Comparative Framework |
| Sripradeep Nekkanti | Physician Interface & UI Design |

---
