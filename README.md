# 🍎 Comparative Analysis of CNN and Transfer Learning Models with Explainable AI

> **Fresh vs. Stale Food Classification** using Custom CNN, MobileNetV2, and ResNet50 — with Grad-CAM and SHAP explanations.

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Explainability](#explainability)
- [Outputs](#outputs)
- [Setup & Usage](#setup--usage)
- [Requirements](#requirements)

---

## Overview

This project trains and compares three image classification models on a 22-class fresh/stale food dataset and then explains their predictions using two XAI techniques:

- **Grad-CAM** — visualises which regions of an image drive a model's prediction
- **SHAP** — quantifies each pixel's contribution (positive or negative) to the output

The full pipeline runs end-to-end in a single Kaggle notebook (`cnn-final.ipynb`) and exports every figure as a PDF and every table as a CSV.

---

## Dataset

**[Fresh and Stale Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)** by Swoyam Siddharth Nayak.

| Split      | Images |
|------------|-------:|
| Train      | 20,994 |
| Validation |  2,625 |
| Test       |  6,738 |
| **Total**  | **30,357** |

### Classes (22 total)

The dataset contains 11 fresh and 11 rotten categories:

| Fresh | Rotten |
|-------|--------|
| freshapples | rottenapples |
| freshbanana | rottenbanana |
| freshbittergroud | rottenbittergroud |
| freshcapsicum | rottencapsicum |
| freshcucumber | rottencucumber |
| freshokra | rottenokra |
| freshoranges | rottenoranges |
| freshpotato | rottenpotato |
| freshtomato | rottentomato |
| … | … |

Class imbalance is handled via **weighted loss** during training (weights ranging from 0.33 to 954 depending on class frequency).

---

## Project Structure

```
.
├── cnn-final.ipynb              # Main notebook (run on Kaggle)
├── README.md
└── outputs/
    ├── figures/                 # All plots saved as PDF
    │   ├── class_distribution.pdf
    │   ├── figure_1_dataset_samples.pdf
    │   ├── figure_2_model_design_workflow.pdf
    │   ├── figure_3_accuracy_loss_curves.pdf
    │   ├── figure_4_confusion_matrix_comparison.pdf
    │   ├── figure_5_gradcam_correct_prediction_examples.pdf
    │   ├── figure_6_gradcam_misclassification_examples.pdf
    │   ├── figure_7_shap_explanation_custom_cnn.pdf
    │   ├── figure_7_shap_explanation_mobilenetv2.pdf
    │   └── figure_7_shap_explanation_resnet50.pdf
    ├── tables/                  # All data saved as CSV
    │   ├── dataset_description.csv
    │   ├── class_distribution.csv
    │   ├── model_architecture_table.csv
    │   ├── training_history_{model}.csv
    │   ├── evaluation_results.csv
    │   ├── classification_report_{model}.csv
    │   ├── confusion_matrix_{model}.csv
    │   ├── predictions_{model}.csv
    │   ├── gradcam_examples.csv
    │   ├── shap_examples.csv
    │   ├── final_comparison_table.csv
    │   ├── research_question_summary.csv
    │   └── output_manifest.csv
    └── models/                  # Saved Keras models
        ├── custom_cnn.keras     (~24 MB)
        ├── mobilenetv2.keras    (~22 MB)
        └── resnet50.keras       (~212 MB)
```

---

## Models

All models take **224 × 224 RGB** images as input and are trained with:

- **Optimiser:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Initial epochs:** 20 + **fine-tuning epochs:** 8
- **Callbacks:** ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

### Data Augmentation (applied to all models)

```python
RandomFlip("horizontal")
RandomRotation(0.10)
RandomZoom(0.12)
RandomContrast(0.15)
```

### Model Architectures

| Model | Architecture | Parameters | Pretrained |
|-------|-------------|------------|------------|
| **Custom CNN** | 4 × [Conv→BN→Conv→BN→MaxPool] blocks (32→64→128→256→384 filters) + GAP + Dropout(0.4) + Dense | ~2.1M | ❌ |
| **MobileNetV2** | MobileNetV2 backbone (ImageNet) frozen → fine-tune last 40 layers + GAP + Dropout(0.3) + Dense | ~2.3M | ✅ ImageNet |
| **ResNet50** | ResNet50 backbone (ImageNet) frozen → fine-tune last 40 layers + GAP + Dropout(0.3) + Dense | ~23.6M | ✅ ImageNet |

Fine-tuning unfreezes the last 40 layers of the pretrained backbone and retrains at a lower learning rate (1e-5).

---

## Results

Final evaluation on the held-out **test set (6,738 images)**:

| Model | Accuracy | Weighted F1 | Training Time | Parameters |
|-------|----------|-------------|---------------|------------|
| **ResNet50** | **81.08%** | **0.8107** | 1,653 s | 23,632,790 |
| MobileNetV2 | 80.81% | 0.8082 | **679 s** | 2,286,166 |
| Custom CNN | 77.80% | 0.7791 | 2,180 s | 2,069,686 |

**Key findings:**

- ResNet50 achieves the highest accuracy and F1-score, benefiting from its deeper residual architecture and ImageNet pretraining.
- MobileNetV2 is the best efficiency trade-off — nearly matching ResNet50 accuracy in less than a third of ResNet's training time, and with 10× fewer parameters.
- The Custom CNN is a competitive from-scratch baseline, reaching ~78% accuracy without any pretrained weights.
- All three models struggled with low-frequency classes (e.g. `freshpatato`, `freshtamto`) due to extreme class imbalance, which class weighting only partially mitigates.

---

## Explainability

### Grad-CAM (Figures 5 & 6)

Gradient-weighted Class Activation Mapping highlights the spatial regions that most influence each prediction. Applied to both correctly classified and misclassified examples to reveal where models succeed and fail.

- **Figure 5** — Grad-CAM on correctly predicted images: all three models focus on surface texture and colour change regions (e.g. discolouration patches on rotten items).
- **Figure 6** — Grad-CAM on misclassified images: incorrect predictions often show attention drifting to background regions or similar-looking neighbouring classes.

### SHAP (Figure 7)

SHAP (SHapley Additive exPlanations) with a `GradientExplainer` assigns pixel-level importance scores:

- **Red pixels** — positively contribute to the predicted class
- **Blue pixels** — negatively suppress the predicted class

One sample per model is visualised, alongside the original image and SHAP overlay.

---

## Outputs

The notebook produces **34 output files** in total:

| Type | Count | Location |
|------|------:|----------|
| PDF figures | 10 | `outputs/figures/` |
| CSV tables | 21 | `outputs/tables/` |
| Keras models | 3 | `outputs/models/` |

All outputs are also bundled into `cnn_project_export.zip` at the end of the notebook for easy download from Kaggle.

---

## Setup & Usage

### Running on Kaggle (recommended)

1. Open [Kaggle](https://www.kaggle.com) and create a new notebook.
2. Attach the dataset: **[Fresh and Stale Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)**.
3. Enable GPU accelerator (Tesla P100 or T4 recommended).
4. Upload `cnn-final.ipynb` and run all cells.
5. Outputs are saved to `/kaggle/working/outputs/` and zipped for download.

### Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset and place it in ./Data/
#    (or set the DATA_ROOT environment variable)
export DATA_ROOT=/path/to/fresh-and-stale-classification

# 4. Launch the notebook
jupyter notebook cnn-final.ipynb
```

The notebook auto-detects the dataset location via `DATA_ROOT`, the Kaggle input path, or a local `./Data/` folder — whichever is found first.

---

## Requirements

```
tensorflow>=2.15
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
shap
```

Tested on **Python 3.12**, **TensorFlow 2.19**, GPU: Tesla P100-PCIE-16GB.

Install all dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn Pillow shap
```

---

## Research Questions Addressed

| # | Question | How to Answer |
|---|----------|---------------|
| RQ1 | What classification accuracy does a custom CNN achieve from scratch? | `evaluation_results.csv` → Custom CNN row |
| RQ2 | How do MobileNetV2 and ResNet50 compare to the custom CNN? | Compare all three rows in `final_comparison_table.csv` |
| RQ3 | Which model offers the best accuracy-to-compute trade-off? | Sort by F1 vs. `training_time_seconds` |
| RQ4 | Which image regions drive correct predictions? | Figure 5 (Grad-CAM correct examples) |
| RQ5 | Do the three models focus on the same regions? | Compare columns in Figure 5 |
| RQ6 | What positive/negative evidence does SHAP reveal? | Figure 7 (red = positive, blue = negative) |
| RQ7 | Can XAI help diagnose misclassifications? | Figure 6 + `gradcam_examples.csv` |

---

## License

This project is released under the [MIT License](LICENSE). The dataset is subject to its own [Kaggle license terms](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification).
