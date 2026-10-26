# CardioRisk ML — Presentation Outline, MCA Viva Questions & Deployment Guide

This document prepares you for academic viva examinations, placement technical interviews, and capstone project demonstrations.

---

## 🎤 1. Executive Elevator Pitch (60 Seconds)

> *"CardioRisk ML is an end-to-end Machine Learning research system developed to classify cardiovascular disease risk across 68,584 verified patient records. Rather than relying on simple toy scripts, the project implements a rigorous production pipeline with zero data leakage, Scikit-learn ColumnTransformers, and Stratified 5-Fold Cross-Validation.*
>
> *We systematically compared four algorithms: Logistic Regression, Decision Tree, Support Vector Machine with Platt calibration, and Random Forest. Random Forest emerged as our champion with a test ROC-AUC of 0.7942 and 72.40% accuracy. The system features an 8-page healthcare-themed Streamlit dashboard and pre-configured Tableau BI integration, delivering live clinical feature derivations like BMI, Pulse Pressure, and Mean Arterial Pressure."*

---

## 📑 2. Slide-by-Slide Project Presentation Outline

### Slide 1: Title & Introduction
- **Header:** CardioRisk ML: Heart Disease Classification & Research Analytics
- **Sub-bullet:** Supervised Binary Classification on 68,584 Verified Records
- **Points:** Motivation (CVD is the world's leading cause of mortality), project objectives, technology stack (Python, Scikit-learn, Streamlit, Tableau).

### Slide 2: Dataset Architecture & Clinical Cleaning
- **Raw Data:** 70,000 cases, 12 clinical & lifestyle attributes, binary target `cardio`.
- **Cleaning Methodology:** Filtered recording noise ($ap\_hi > 240$, $ap\_lo < 40$, $ap\_lo > ap\_hi$). Retained 68,584 cases (97.98%).
- **Class Balance:** 50.5% Negative / 49.5% Positive (natural class balance).

### Slide 3: Feature Engineering & Physiological Indicators
- **`age_years`:** Normalized from recording days (`age / 365.25`).
- **`bmi`:** Quetelet index (`weight / height_m²`) indicating adiposity.
- **`pulse_pressure`:** Arterial compliance marker (`ap_hi - ap_lo`).
- **`map`:** Mean Arterial Pressure perfusion index (`ap_lo + 1/3 * PP`).

### Slide 4: Zero Data Leakage Pipeline
- **Diagram:** Raw Features $\rightarrow$ `ClinicalFeatureEngineer` $\rightarrow$ `ColumnTransformer` (StandardScaler on numerics, Passthrough on categoricals) $\rightarrow$ Estimator.
- **Core Principle:** Preprocessing fits exclusively on `X_train` within each cross-validation fold.

### Slide 5: Multi-Algorithm Experimental Results
- **Comparison Table:** Logistic Regression (ROC-AUC: 0.7860), Decision Tree (ROC-AUC: 0.7852), SVM (ROC-AUC: 0.7857), Random Forest (ROC-AUC: 0.7942).
- **Key Observation:** Strong convergence around 72.1% - 73.5% across linear, tree, margin, and ensemble paradigms.

### Slide 6: Model Evaluation & Clinical Error Analysis
- **Confusion Matrix:** True Positives (4,606), True Negatives (5,324), False Positives (1,544), False Negatives (2,243).
- **Error Trade-off:** False Negatives represent missed clinical intervention; False Positives cause unnecessary follow-up testing.

### Slide 7: Interactive Streamlit Platform & Tableau BI
- **Live Demo Tour:** Executive Overview, EDA, Model Lab, Evaluation, Feature Insights, Research Prediction Interface with real-time derived markers, Dataset Explorer.
- **Tableau BI:** Executive dashboard connecting exported analytical CSVs.

### Slide 8: Limitations, Ethics & Future Work
- **Limitations:** Cross-sectional cohort (no longitudinal survival tracking); self-reported lifestyle markers.
- **Ethical Notice:** Educational tool, not a clinical diagnostic device.
- **Future Enhancements:** XGBoost/LightGBM integration, SHAP explainability values, FHIR medical data interoperability.

---

## ❓ 3. Top 20 MCA Viva Voce Questions & Answers

### Q1: Why did you choose this dataset, and what is its target variable?
**Answer:** The dataset comprises 70,000 observational patient records for cardiovascular disease classification. The target variable is `cardio`, a binary variable where `0` denotes absence and `1` denotes clinical presence of cardiovascular disease. The dataset is balanced (~50.5% to 49.5%), eliminating the need for synthetic oversampling like SMOTE.

### Q2: Why did you filter blood pressure records where `ap_lo > ap_hi`?
**Answer:** Systolic blood pressure (`ap_hi`) represents maximum pressure during ventricular contraction, whereas diastolic pressure (`ap_lo`) represents baseline pressure during cardiac relaxation. In human physiology, systolic pressure is always greater than diastolic pressure. Rows where $ap\_lo > ap\_hi$ or where values exceeded 300 mmHg are typographical or sensor artifacts. Removing them preserved 97.98% of clean, authentic data.

### Q3: What is "Data Leakage," and how did you prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as test set statistics) influences model training, producing overly optimistic evaluation metrics. We prevented this by encapsulating all scaling and transformations inside Scikit-learn `Pipeline` and `ColumnTransformer` objects. The scalers fit strictly on `X_train` and merely transform `X_test`.

### Q4: Why did you engineer Pulse Pressure and Mean Arterial Pressure?
**Answer:** In cardiology, raw systolic and diastolic pressures interact strongly. Pulse Pressure ($\text{SBP} - \text{DBP}$) is a direct clinical index of arterial stiffening and vascular aging. Mean Arterial Pressure ($\text{DBP} + \frac{1}{3}\text{PP}$) measures average tissue perfusion. Providing these domain-derived features allows both linear and tree models to capture vascular stress without needing complex polynomial terms.

### Q5: How did you select the Champion Model?
**Answer:** The champion model was selected based on the highest 5-Fold Stratified Cross-Validation ROC-AUC score on training data. Random Forest achieved a CV ROC-AUC of $0.8017 \pm 0.0029$ and a test ROC-AUC of $0.7942$, outperforming Logistic Regression, Decision Tree, and Linear SVM.

### Q6: Why is ROC-AUC often preferred over Accuracy in healthcare ML?
**Answer:** While our dataset is balanced, accuracy evaluates only a single default threshold ($p = 0.5$). In clinical screening, false negatives are more critical than false positives. ROC-AUC evaluates the model's discriminative ability across all possible classification thresholds, measuring the probability that the classifier ranks a randomly chosen positive patient higher than a randomly chosen negative patient.

### Q7: How does Platt Scaling work in your Support Vector Machine?
**Answer:** Standard Support Vector Machines output uncalibrated signed distances to the separating hyperplane via `decision_function`, rather than posterior probabilities. We wrapped `LinearSVC` inside `CalibratedClassifierCV(cv=3)`, which fits a logistic sigmoid function over the SVM decision values using cross-validation (Platt scaling), yielding valid probabilities $P(Y=1|X)$.

### Q8: What is the inductive bias of a Decision Tree vs Logistic Regression?
**Answer:** Logistic Regression assumes a linear relationship between features and the log-odds of disease, creating a hyperplanar decision boundary. Decision Trees make no linear assumptions; their inductive bias is orthogonal, axis-aligned recursive partitioning, allowing them to capture non-linear step functions and interactions without feature scaling.

### Q9: Why did you constrain the Decision Tree depth to 6?
**Answer:** An unconstrained decision tree will split until all leaves are pure, capturing sample-specific noise and overfitting heavily (achieving ~99% train accuracy but dropping to <68% test accuracy). Setting `max_depth=6`, `min_samples_split=25`, and `min_samples_leaf=15` regularizes tree complexity and ensures high generalization.

### Q10: How does Random Forest reduce model variance?
**Answer:** Random Forest uses bootstrap aggregation (bagging) over $B = 100$ decorrelated decision trees, combined with random feature subspace sampling at each split ($m = \sqrt{p}$). Averaging the predictions across decorrelated trees reduces the ensemble's variance by approximately a factor of $B$, preventing the high variance of individual deep trees.

### Q11: What is the difference between Sensitivity and Specificity in this project?
**Answer:**
- **Sensitivity (Recall for Class 1):** Proportion of actual CVD patients correctly detected: $\frac{\text{TP}}{\text{TP} + \text{FN}} = 67.15\%$.
- **Specificity (True Negative Rate):** Proportion of healthy individuals correctly identified: $\frac{\text{TN}}{\text{TN} + \text{FP}} = 77.54\%$.

### Q12: Which features demonstrated the highest predictive weight?
**Answer:** Across all four models, the dominant predictors were Systolic Blood Pressure (`ap_hi`), Mean Arterial Pressure (`map`), Patient Age (`age_years`), and Serum Cholesterol (`cholesterol`). Anthropometric factors like BMI and weight contributed moderately, while self-reported habits (smoking, alcohol) had smaller relative weights in this specific dataset.

### Q13: Does feature importance establish causality?
**Answer:** No. Feature importance in observational ML indicates statistical association and split frequency within the dataset. It does not establish biological causality or directional etiology. We highlight this prominently in the application and documentation.

### Q14: How does your prediction interface handle missing or raw patient inputs?
**Answer:** The interface accepts patient inputs in standard clinical units (e.g., age in years, height in cm, weight in kg). `src.predict` validates the inputs, applies the custom `ClinicalFeatureEngineer` to compute derived variables, and passes the formatted row to the saved Scikit-learn pipeline for standardized transformation and inference.

### Q15: Why didn't you use deep learning (e.g., MLP or Neural Networks)?
**Answer:** Tabular clinical datasets with low-to-medium feature dimensionality (12 to 16 features) are well known to be optimally handled by regularized linear models and gradient-boosted or bagged decision trees. Deep learning adds substantial training complexity, latency, and reduced interpretability without meaningful performance gains on tabular medical datasets of this scale.

### Q16: How did you ensure reproducibility in your experiments?
**Answer:** All stochastic operations—including train/test splitting, K-Fold cross-validation folds, and tree bootstrapping—were initialized with a fixed random seed (`RANDOM_SEED = 42`). Model pipelines were serialized with exact dependency specifications.

### Q17: What is the PR-AUC metric and when is it crucial?
**Answer:** PR-AUC (Precision-Recall Area Under Curve / Average Precision) plots Precision against Recall across varying thresholds. It is especially critical in imbalanced classification where the negative class dominates, as ROC-AUC can present an overly optimistic assessment due to a large True Negative denominator.

### Q18: What are the main clinical limitations of this project?
**Answer:**
1. Cross-sectional nature: Records represent a single medical examination snapshot rather than longitudinal time-to-event outcomes.
2. Unmeasured confounders: Key cardiac biomarkers (e.g., troponin, hs-CRP, LDL/HDL ratio, ECG readings, family history) were not collected in the source dataset.
3. Self-reported behavioral attributes: Smoking, alcohol, and physical activity are subject to patient recall bias.

### Q19: How is the project structured for modularity and production readiness?
**Answer:** Code is separated into dedicated single-responsibility modules: `data_loader.py` for I/O, `preprocessing.py` for transforms, `train.py` for pipeline building, `evaluate.py` for metrics computation, `predict.py` for inference, and `visualization.py` for reusable charts. Automated tests in `tests/test_pipeline.py` verify functionality.

### Q20: How does the Tableau dashboard complement the Streamlit application?
**Answer:** The Streamlit app provides an interactive user-facing clinical simulation and prediction interface. The Tableau integration provides an enterprise business intelligence perspective, allowing hospital administrators to analyze population health distributions, demographic risk patterns, and cohort trends through pre-aggregated CSV feeds.

---

## 📸 4. Project Demonstration & Screenshot Guide

To capture compelling demonstration screenshots for your portfolio or LinkedIn:

1. **Executive Overview:**
   - Run `streamlit run app.py`.
   - Maximize browser (`F11`).
   - Capture the dark navy hero banner, the 4 top KPI cards, the algorithm benchmark table, and the donut chart.
2. **Interactive Clinical Prediction Interface:**
   - Navigate to Page 6.
   - Click the **"⚠️ Hypertensive Patient"** preset button.
   - Click **"Compute Cardiovascular Risk Assessment"**.
   - Capture the red/coral result card showing `85.0% Probability Score`, risk tier badge, derived BMI/PP/MAP values, and flag alerts.
3. **Confusion Matrix & Diagnostic Curves:**
   - Navigate to Page 4.
   - Select **Random Forest (Benchmark)**.
   - Capture the interactive confusion matrix with percentages alongside the ROC Curve and classification report.
4. **Feature Importance Ranking:**
   - Navigate to Page 5.
   - Capture the horizontal bar chart showing the dominant weights of Blood Pressure, Age, and Cholesterol.

---

## 🌐 5. GitHub Repository Setup & Deployment

### Step 5.1: Initialize Local Git Repository
```bash
cd C:\Users\NIKHIL\.gemini\antigravity\scratch\CardioRisk-ML
git init
git add .
git commit -m "feat: complete CardioRisk ML research and analytics platform"
```

### Step 5.2: Push to GitHub
```bash
# Create a new repository on github.com named CardioRisk-ML
git remote add origin https://github.com/<YOUR_USERNAME>/CardioRisk-ML.git
git branch -M main
git push -u origin main
```

### Step 5.3: Deploy to Streamlit Community Cloud (Free Hosting)
1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **"New App"**.
3. Select your repository: `<YOUR_USERNAME>/CardioRisk-ML`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**. Your live healthcare dashboard will be online in under 2 minutes.
