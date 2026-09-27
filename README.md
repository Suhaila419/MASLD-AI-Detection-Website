# MASLD Ultrasound AI Pipeline

An end-to-end deep learning pipeline for analyzing liver ultrasound images from patients with **MASLD** (Metabolic dysfunction-Associated Steatotic Liver Disease). The system segments the liver from raw ultrasound frames and then classifies disease severity along three clinical axes: **Grade**, **Fibrosis stage**, and **Steatosis stage**.

## Overview

MASLD is typically assessed using invasive biopsy or specialized equipment (e.g. FibroScan). This project explores whether liver ultrasound images alone — combined with an automatically predicted liver mask, and optionally patient clinical/lab data — can be used to predict the same clinical stages with a deep learning pipeline.

**Dataset**
- 2,031 ultrasound images
- 143 patients
- Ground-truth labels derived from clinical grading, FibroScan reports (fibrosis stage, extracted from report images via OCR into an Excel sheet), and patient clinical/lab records

## Pipeline

The pipeline is built in stages, each one tested independently so that the contribution of every additional input (mask, clinical data, fine-tuning) can be measured:

```
Raw ultrasound image
        │
        ▼
┌───────────────────────┐
│ 1. Segmentation        │   Attention U-Net → predicted liver mask
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 2. Grade classification│   DenseNet121 (image only) → Grade 1 / 2 / 3
└───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 3. Fibrosis / Steatosis staging│  DenseNet121 (image + mask) → stage
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4. Multi-modal fusion (optional)        │  DenseNet121 features + patient
│    Image + Mask + Clinical/Lab data     │  clinical/lab features → stage
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 5. Tabular-only baseline        │  XGBoost on clinical/lab data alone
└───────────────────────────────┘
```

### 1. Liver Segmentation — Attention U-Net
Locates the liver region in the raw ultrasound frame. Attention gates on the skip connections help the model focus on the liver and ignore background/on-screen text typical of ultrasound machines.
- Loss: binary cross-entropy · Metrics: accuracy, Mean IoU
- **Validation accuracy: 95.6%**, Validation loss: 0.107

### 2. Grade Classification — DenseNet121 (image only)
Predicts MASLD **Grade 1 / 2 / 3** directly from the raw ultrasound image.
- **Accuracy: 99.0%**

### 3. Fibrosis & Steatosis Staging — DenseNet121 (image + predicted mask)
The ultrasound image is combined with its predicted liver mask and used to predict:
- **Fibrosis stage** (F0-F1 / F2-F3 / F3-F4) — **Accuracy: 99.5%**
- **Steatosis stage** (S0 / S1 / S2 / S3) — **Accuracy: 99.5%**

### 4. Multi-modal Fusion — DenseNet121 (image + mask) + Clinical/Lab Data
Image features are concatenated with a tabular branch built from patient clinical/lab data (age, BMI, ALT, AST, lipid profile, glycemic markers, etc.) before the final classification layer. Partial fine-tuning of the last 50 DenseNet121 layers was also tested.
- Steatosis (fusion): **88.9%** accuracy (no change after fine-tuning)
- Fibrosis (fusion): **79.3%** accuracy (no change after fine-tuning)
- Fusing tabular data reduced accuracy compared with the image+mask-only models (see [Results](#results) below).

### 5. Tabular-Only Baseline — XGBoost
Clinical/lab data alone (no image), used to measure how much signal the tabular data carries on its own.
- Steatosis: **87%** accuracy
- Fibrosis: **73%** accuracy

## Preprocessing

**Image pipeline**
- On-the-fly data augmentation: random horizontal flip applied to each image and its mask using the same seed, keeping them aligned. Generated at training time (no extra files saved), reducing overfitting at no extra storage cost.
- `tf.data.AUTOTUNE` used throughout the data pipeline (mapping, batching, prefetching) to remove CPU/GPU idle time and speed up training.

**Tabular pipeline**
- Label Encoding for categorical clinical columns (Gender, DM, HTN, virology, etc.)
- Standard Scaling for numeric lab/clinical columns (BMI, ALT, AST, triglyceride, HDL, cholesterol, FBG, HBA1C, etc.)
- Class-balancing augmentation: minority-class rows resampled and perturbed with small Gaussian noise on numeric columns to reduce class imbalance
- Feature selection: clinically relevant columns retained based on importance to the target stage

## Results

| Task | Input | Model | Test size | Accuracy |
|---|---|---|---|---|
| Segmentation | Image | Attention U-Net | — | 95.6% (val.) |
| Grade classification | Image only | DenseNet121 | 204 | **99.0%** |
| Fibrosis staging | Image + Mask | DenseNet121 | 204 | **99.5%** |
| Steatosis staging | Image + Mask | DenseNet121 | 204 | **99.5%** |
| Steatosis staging (fusion) | Image + Mask + Tabular | DenseNet121 + fusion | 198 | 88.9% |
| Steatosis staging (fusion, fine-tuned) | Image + Mask + Tabular | DenseNet121 (partial FT) + fusion | 198 | 88.9% |
| Fibrosis staging (fusion) | Image + Mask + Tabular | DenseNet121 + fusion | 198 | 79.3% |
| Fibrosis staging (fusion, fine-tuned) | Image + Mask + Tabular | DenseNet121 (partial FT) + fusion | 198 | 79.3% |
| Steatosis staging (tabular only) | Clinical/lab data only | XGBoost | 39 | 87% |
| Fibrosis staging (tabular only) | Clinical/lab data only | XGBoost | 26 | 73% |

### Key findings
- **Image + mask input with DenseNet121 is the best-performing configuration** for every classification task (Grade 99.0%, Fibrosis 99.5%, Steatosis 99.5%), including strong performance on rare/minority classes.
- **Feature-level fusion of clinical/tabular data hurt performance** rather than helping it (Steatosis 99.5% → 88.9%, Fibrosis 99.5% → 79.3%), with the biggest drop in recall on minority classes.
- **Partial fine-tuning of the last 50 DenseNet121 layers made no measurable difference** on the fusion models — the bottleneck lies elsewhere (likely the fusion strategy itself or the patient-level vs. image-level mismatch of the tabular data).
- **Clinical/lab data does carry real signal on its own** (XGBoost baseline: 87% steatosis, 73% fibrosis), suggesting a decision-level ensemble (e.g. combining predicted probabilities from the image model and the tabular model) may be a better way to use it than feature-level fusion.

## Repository Structure

```
├── notebooks/
│   ├── masald-segmentation-attention-unet.ipynb          # Liver segmentation (Attention U-Net)
│   ├── masald-classification-densenet121-grades.ipynb    # Grade classification (image only)
│   ├── masald-classification-stage-fibrosis-onlyimage.ipynb   # Fibrosis staging (image + mask)
│   ├── masald-classification-stage-steatosis.ipynb        # Steatosis staging (image + mask)
│   ├── masald-classification-stage-fibrosis.ipynb          # Fibrosis staging (image + mask + tabular fusion)
│   ├── masald-classification-stage-steatosis-tabular.ipynb # Steatosis staging (image + mask + tabular fusion)
│   ├── stage-fibrosis-classifier.ipynb                     # Fibrosis staging (tabular only, XGBoost)
│   └── stage-steatosis.ipynb                               # Steatosis staging (tabular only, XGBoost)
└── README.md
```

## Tech Stack
- **Deep Learning:** TensorFlow / Keras (Attention U-Net, DenseNet121)
- **Classical ML:** scikit-learn, XGBoost
- **Data processing:** pandas, NumPy, OpenCV / PIL

## Metrics Glossary
- **Precision** — of everything predicted as a class, how much was actually correct.
- **Recall** — of everything that truly belongs to a class, how much was correctly detected.
- **F1-score** — harmonic mean of precision and recall; most informative on imbalanced classes.
- **Support** — number of true samples of that class in the test set.
- **Macro avg** — unweighted average across classes (treats rare classes equally).
- **Weighted avg** — average weighted by class size (leans toward the majority class).

## Disclaimer
This project is a research prototype for academic/experimental purposes and is **not** a certified diagnostic tool. Results should not be used for clinical decision-making without validation by qualified medical professionals.
