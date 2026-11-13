# AlveoSage — Explainable AI-Powered Chest X-ray Analysis Assistant

AlveoSage is a deep-learning clinical decision-support application designed to analyze pediatric chest radiographs (CXRs), classify them into **NORMAL** or **PNEUMONIA**, generate visual saliency maps with **Grad-CAM**, quantify epistemic uncertainty via **Monte Carlo Dropout**, operate on validation-optimized decision thresholds (**0.80**), and produce structured preliminary radiologic reports.

---

## 1. Key Capabilities & Methodology

1. **Deep Learning Classification**:
   - **DenseNet121** architecture utilizing ImageNet pretrained features.
   - Selective fine-tuning of `denseblock4` and `norm5` with frozen shallow layers.
   - Custom classification head featuring ReLU, AdaptiveAvgPool2d, Linear, BatchNorm1d, ReLU, Dropout ($p = 0.3$), and a binary linear classifier.
   - Balanced weighted CrossEntropy loss to handle class imbalance ($2.88 : 1$).
   - Automatic Mixed Precision (AMP) for optimized throughput.
   - Experimental classification threshold **0.80** (eliminating 25 test-set false positives while maintaining $\ge 99\%$ pneumonia sensitivity).

2. **Visual Explainability (Grad-CAM)**:
   - Gradient-weighted Class Activation Mapping computed from the final convolutional layer (`denseblock4.denselayer16.conv2`).
   - Produces three visualization artifacts: Original radiograph, Jet-colormapped activation heatmap, and superimposed blended overlay.
   - Corresponds directly to the predicted class.
   - Interactive alpha opacity slider in the web interface for fine-grained anatomical correlation.

3. **Epistemic Uncertainty Quantification (MC Dropout)**:
   - Keeps Dropout active during evaluation while freezing BatchNorm running statistics.
   - Performs $N$ stochastic forward passes (default: 20 passes).
   - Computes mean predictive probabilities, predictive variance ($\sigma^2$), and normalized Shannon entropy ($H$).
   - Categorizes uncertainty into qualitative risk tiers: **LOW**, **MODERATE**, or **HIGH**, signaling when senior radiologist review is vital.

4. **Operating Threshold Benchmark Comparison**:
   - Compares baseline threshold ($0.50$) with validation-derived threshold ($0.80$) on the untouched test split (624 samples).
   - Increases accuracy from $85.74\%$ to $89.42\%$, increases specificity from $62.39\%$ to $73.08\%$, and eliminates 25 false positives with zero test-set tuning.

5. **Structured Preliminary Reporting**:
   - Automated generation of structured clinical summaries including patient examination metadata, predicted class, confidence, uncertainty indicators, system limitations, and visual attribution notes.
   - Prominent notice that the application is experimental and not clinically validated.
   - Downloadable as standalone Markdown documents.

6. **Interactive Medical AI Dashboard**:
   - Polished Flask-based web application with drag-and-drop CXR upload, sample image selector, real-time analysis, interactive Plotly visualizations, and model performance telemetry.

---

## 2. Project Architecture

```
AlveoSage/
├── dataset/                    # Original raw chest X-ray archive (unmodified)
├── processed_dataset/          # Verified curated dataset & manifests
│   ├── train/                  # 4,448 images (1,147 NORMAL, 3,301 PNEUMONIA)
│   ├── val/                    # 784 images (202 NORMAL, 582 PNEUMONIA)
│   ├── test/                   # 624 images (234 NORMAL, 390 PNEUMONIA)
│   ├── manifests/              # train.csv, val.csv, test.csv
│   └── statistics/             # dataset_stats.json, distribution charts
│
├── config/
│   ├── dataset_config.json     # Dataset paths & splitting parameters
│   ├── model_config.json       # Hyperparameters, augmentation, & architecture
│   └── app_config.json         # Server, paths, thresholds, & upload settings
│
├── src/
│   ├── models/                 # DenseNet121 architecture & custom head
│   ├── data/                   # PyTorch Dataset, DataLoaders, & transforms
│   ├── training/               # Trainer, early stopping, & checkpointing
│   ├── evaluation/             # Clinical metrics, ROC, threshold comparison, & visualizations
│   ├── explainability/         # Hook-based Grad-CAM & overlay generator
│   ├── uncertainty/            # MC Dropout & Temperature Scaling hooks
│   ├── inference/              # Unified AlveoSagePredictor pipeline (threshold 0.80)
│   ├── reporting/              # Structured clinical report generator
│   └── preprocessing/          # Phase 1 dataset inspection & preparation
│
├── app/
│   ├── backend/                # Flask application factory & REST endpoints
│   ├── templates/              # Jinja2 HTML templates (Dashboard, Analysis, Results, Explainability, Performance)
│   └── static/                 # Clinical CSS styles, JS app, & images
│
├── outputs/
│   ├── checkpoints/            # Model weights (best_model.pth, last_model.pth)
│   ├── training/               # Loss/accuracy history & curves
│   ├── evaluation/             # Confusion matrices, ROC curves, metrics, & threshold comparisons
│   ├── heatmaps/               # Generated Grad-CAM saliency images
│   ├── reports/                # Exported markdown preliminary reports
│   └── uploads/                # User-uploaded radiograph images
│
├── tests/                      # Automated test suite (Phase 1 verified)
├── main.py                     # Web application launcher
├── train.py                    # Model training entry point
├── evaluate_test.py            # Final test set evaluation entry point
├── compare_test_thresholds.py  # Test threshold comparison entry point
├── requirements.txt            # Dependency specification
└── README.md
```

---

## 3. Dataset Distribution (Processed)

The dataset was curated and verified in Phase 1, resolving the original validation sample deficiency (16 images) through a stratified 85/15 split grouped by image hash to prevent duplicate leakage:

| Split | NORMAL | PNEUMONIA | Total | Ratio (P/N) |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 1,147 | 3,301 | 4,448 | 2.88 : 1 |
| **VAL** | 202 | 582 | 784 | 2.88 : 1 |
| **TEST** | 234 | 390 | 624 | 1.67 : 1 |
| **TOTAL** | **1,583** | **4,273** | **5,856** | **2.70 : 1** |

---

## 4. Execution Guide

### A. Launch Web Dashboard
Start the local Flask application server:
```bash
python main.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### B. Clinical Workspace Workflow
1. Navigate to `http://127.0.0.1:5000`.
2. In the Left Control Panel, drag & drop a CXR image or select a verified test sample (e.g., *Normal 1* or *Pneumonia 1*).
3. Adjust operating threshold $\tau$ (default `0.80`) and MC Dropout sample count (default `20`).
4. Click **RUN ANALYSIS**.
5. Diagnostic results appear instantly in-place within the workspace:
   - Centerpiece viewer updates with three switchable view modes: **Original**, **Grad-CAM**, and **Overlay**.
   - Interactive slider dynamically controls overlay blend opacity (0% to 100%).
   - Bottom diagnostic dock displays predicted class (`NORMAL` or `PNEUMONIA`), confidence %, probability meters, and epistemic uncertainty indicators.
   - Click **Download Report** to export the structured clinical summary (`.md`).
6. Navigate to **Model Analytics** (`/analytics`) to inspect independent test benchmark ROC curves, PR curves, confusion matrix, calibration diagram, and threshold comparison telemetry.

### C. Evaluate Model & Test Thresholds
```bash
# Final test-set benchmark evaluation
python evaluate_test.py

# Benchmark test-set threshold comparison (0.50 vs 0.80)
python compare_test_thresholds.py

# Validation classification threshold sweep (0.10 to 0.90)
python -m src.evaluation.threshold_analysis
```

---

## 5. Clinical Disclaimer

AlveoSage is an **EXPERIMENTAL** decision-support system built for research and demonstration purposes. It is **NOT CLINICALLY VALIDATED** and does not provide autonomous clinical diagnoses. All machine learning predictions, epistemic uncertainty measures, and Grad-CAM saliency heatmaps must be reviewed, verified, and contextualized by a board-certified physician or radiologist.
