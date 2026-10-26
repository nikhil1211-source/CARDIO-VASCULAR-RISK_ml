# CardioRisk ML — Heart Disease Classification & Research Analytics

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53.1-FF4B4B.svg)](https://streamlit.io/)
[![Tableau](https://img.shields.io/badge/Tableau-BI%20Ready-E97627.svg)](https://www.tableau.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning research system and interactive healthcare analytics dashboard for cardiovascular disease classification. Trained and validated on **68,584 verified patient records** using zero-leakage Scikit-learn pipelines, Stratified 5-Fold Cross-Validation, calibrated probability scoring, and clinical model explainability.

---

## 📌 Executive Summary

Cardiovascular diseases (CVDs) are the leading cause of global mortality. **CardioRisk ML** demonstrates a production-grade machine learning workflow designed for educational research, MCA technical training, and placement portfolio presentation:

- **Strict Zero Data Leakage:** All preprocessing transforms, scalers, and domain derivations fit strictly on training folds within cross-validation.
- **Robust Clinical Outlier Filtering:** Cleans physiological measurement artifacts (e.g., negative blood pressures, typographical recording errors) while preserving **97.98% of authentic cohort data** (68,584 clean cases).
- **Multi-Algorithm Benchmark:** Directly implements and contrasts **Logistic Regression**, **Decision Tree Classifier**, **Support Vector Machine (Calibrated Linear SVM)**, and **Random Forest Classifier (Benchmark)**.
- **Calibrated Probabilistic Inference:** Outputs exact posterior risk probabilities alongside discrete binary predictions.
- **Dual Visual Intelligence:** Features an 8-page healthcare-themed **Streamlit web application** and a structured **Tableau Business Intelligence** integration suite.

> **Research & Portfolio Notice:** This application is developed strictly for educational research and technical demonstration. It is **not** a certified medical diagnostic device and must not be used for individual patient diagnosis or clinical triage.

---

## 🏗️ Project Architecture

```
CardioRisk-ML/
│
├── app.py                     # Premium 8-Page Interactive Streamlit Web Dashboard
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive technical documentation
├── PRESENTATION_AND_VIVA.md   # Presentation script and MCA viva defense Q&A
│
├── data/
│   ├── raw/                   # Original cardio_train.csv (70,000 records)
│   └── processed/             # cardio_cleaned.csv, test_set.csv, cleaning_audit.json
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV ingestion, schema validation, delimiter detection
│   ├── preprocessing.py      # Clinical boundary cleaning & Scikit-learn Pipeline builder
│   ├── train.py               # 5-fold CV, model training, artifact serialization
│   ├── evaluate.py            # Diagnostic metrics (ROC-AUC, PR-AUC, CM, Specificity)
│   ├── predict.py             # Inference engine with probability calibration & risk tiers
│   └── visualization.py      # Custom Plotly healthcare chart themes
│
├── models/
│   ├── best_model_pipeline.joblib   # Persisted champion model pipeline (Random Forest)
│   ├── all_model_pipelines.joblib    # Persisted dictionary of all 4 trained pipelines
│   ├── all_models_metrics.json       # Evaluated metrics across all algorithms
│   ├── feature_importances.json      # Relative clinical feature weights
│   └── model_comparison.csv         # Tabular benchmark comparison
│
├── tableau/
│   ├── tableau_patient_analytics.csv   # Enriched cohort records with AHA & WHO categories
│   ├── tableau_model_comparison.csv    # Cross-algorithm performance metrics
│   ├── tableau_test_predictions.csv    # Test set predictions, actuals, probabilities
│   ├── tableau_feature_importance.csv  # Ranked clinical risk drivers
│   ├── tableau_data_dictionary.csv     # Metadata and field glossary
│   └── TABLEAU_DASHBOARD_GUIDE.md      # Step-by-step Tableau assembly tutorial
│
├── notebooks/
│   └── 01_eda_and_experiments.ipynb    # Jupyter research & profiling notebook
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py       # Unit tests for loading, bounds, features, and inference
```

---

## 📊 Dataset Structure & Clinical Features

The system is trained on the Cardiovascular Disease dataset (`cardio_train.csv`):
- **Cohort Size:** 70,000 raw records $\rightarrow$ **68,584 cleaned verified records** (97.98% retention).
- **Target Distribution:** Extremely balanced ($\approx 50.5\%$ Negative / $49.5\%$ Positive).

| Feature | Type | Unit / Encoding | Clinical Relevance |
| :--- | :--- | :--- | :--- |
| `age_years` | Numeric | Years (converted from days) | Primary non-modifiable risk factor |
| `gender` | Categorical | 1: Female, 2: Male | Biological sex stratification |
| `height` | Numeric | cm (120 to 220 cm) | Anthropometric marker for BMI |
| `weight` | Numeric | kg (30 to 200 kg) | Anthropometric marker for BMI |
| `ap_hi` | Numeric | mmHg (70 to 240 mmHg) | Systolic Blood Pressure (Peak arterial pressure) |
| `ap_lo` | Numeric | mmHg (40 to 140 mmHg) | Diastolic Blood Pressure (Resting arterial pressure) |
| `pulse_pressure` | Numeric | mmHg (`ap_hi - ap_lo`) | Derived marker of arterial compliance & stiffness |
| `map` | Numeric | mmHg (`ap_lo + 1/3 * PP`) | Mean Arterial Pressure (Perfusion pressure) |
| `bmi` | Numeric | $\text{kg/m}^2$ (`weight / (height/100)²`) | Adiposity and metabolic syndrome indicator |
| `cholesterol` | Ordinal | 1: Normal, 2: Above Normal, 3: Well Above | Serum lipid atherogenicity |
| `gluc` | Ordinal | 1: Normal, 2: Above Normal, 3: Well Above | Fasting blood glucose / Diabetes marker |
| `smoke` | Binary | 0: No, 1: Yes | Active tobacco consumption |
| `alco` | Binary | 0: No, 1: Yes | Regular alcohol intake |
| `active` | Binary | 0: Sedentary, 1: Physically Active | Cardiorespiratory fitness protective factor |
| `cardio` | Binary (Target) | 0: Healthy, 1: Cardiovascular Disease | Supervised ground truth target |

---

## 🏆 Model Performance & Comparative Benchmark

All models were evaluated on an identical **held-out test set of 13,717 patients (20%)** using models trained via **5-Fold Stratified Cross-Validation**:

| Model Algorithm | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Benchmark)** 🏆 | **0.7240** | **0.7453** | 0.6715 | 0.7754 | **0.7065** | **0.7942** | **0.7776** | **0.7354 ± 0.0046** |
| **Logistic Regression** | 0.7222 | 0.7441 | 0.6684 | 0.7749 | 0.7042 | 0.7860 | 0.7714 | 0.7286 ± 0.0031 |
| **Support Vector Machine (Linear)** | 0.7224 | 0.7463 | 0.6649 | **0.7787** | 0.7032 | 0.7857 | 0.7709 | 0.7279 ± 0.0033 |
| **Decision Tree Classifier** | 0.7211 | 0.7296 | **0.6930** | 0.7485 | 0.7109 | 0.7852 | 0.7588 | 0.7318 ± 0.0038 |

### Key Diagnostic Takeaways:
1. **Consistency Across Paradigms:** All four distinct mathematical paradigms (linear log-odds, single tree, maximal margin, ensemble bagging) converged to between **72.1% and 73.5% accuracy** and **0.785 to 0.794 ROC-AUC**. This demonstrates that the dataset's discriminative ceiling under routine clinical markers is ~79.4% ROC-AUC.
2. **Clinical Error Analysis:**
   - **False Negatives:** In healthcare screening, False Negatives carry the greatest clinical penalty (delayed diagnostic angiography or statin therapy). The Decision Tree yielded the highest Recall (69.3%), while the Random Forest maximized overall discriminative precision and ROC-AUC.
   - **False Positives:** Addressed through calibrated probability scores, ensuring patients are not placed into alarmist risk categories without high posterior confidence.

---

## 🖥️ Streamlit Web Application (8 Pages)

Launch the interactive dashboard with:
```bash
streamlit run app.py
```

### Application Pages:
1. **Executive Overview:** High-level KPI summary, class balance donut chart, and benchmark comparison table.
2. **Exploratory Data Analysis:** Cohort distributions, interactive filters by gender/age/smoking, AHA blood pressure stages, and correlation heatmaps.
3. **Model Laboratory:** Interactive comparison charts, hyperparameter inspection, and cross-validation variance analysis.
4. **Model Evaluation:** Interactive confusion matrices, ROC curves, PR curves, and full classification reports.
5. **Feature Insights:** Horizontal bar chart of model weights, feature attribution, and causation-vs-association analysis.
6. **Research Prediction Interface:** Clinical input form with dynamic derivations (`BMI`, `PP`, `MAP`), 3 quick archetypal presets ("Low Risk", "Hypertensive", "Borderline"), and calibrated risk tier output.
7. **Dataset Explorer:** Filterable data table, cleaning audit metrics, and one-click download for cleaned CSV data.
8. **Project Information:** Architecture overview, Data Dictionary, leakage prevention methodology, and MCA viva reference materials.

---

## 📈 Tableau BI Integration

Dedicated Tableau CSVs are pre-generated in the `tableau/` folder. Follow `tableau/TABLEAU_DASHBOARD_GUIDE.md` to build:
- Executive KPI Cards
- Algorithm Comparison Bar Charts
- 2x2 Interactive Confusion Analysis Matrix
- Clinical Feature Importance Rankings
- 100% Stacked Bar Charts of AHA Blood Pressure Stages vs. Disease Incidence

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Git

### Setup
```bash
# 1. Clone or navigate to the project directory
cd CardioRisk-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests
python tests/test_pipeline.py

# 4. (Optional) Re-run the full training pipeline
python -m src.train

# 5. Launch the Streamlit Web Application
streamlit run app.py
```

---

## 🛡️ Medical & Legal Disclaimer

**CardioRisk ML is an educational and academic machine learning research project.** It is designed solely for computational experimentation, MCA coursework demonstration, and portfolio presentation. It does **not** provide clinical diagnosis, medical guidance, or individual risk prognosis. Individuals with health concerns should consult certified medical professionals.
