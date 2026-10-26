"""
CardioRisk ML — Heart Disease Classification & Research Analytics
End-to-End Machine Learning Research Dashboard & Prediction System
Author: Senior ML Engineer & UI/UX Designer
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Internal package modules
from src.data_loader import load_raw_data, inspect_raw_data
from src.preprocessing import clean_raw_data, BOUNDS, ALL_MODEL_FEATURES
from src.predict import predict_patient_risk, load_inference_pipeline
from src.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance_bar,
    plot_model_comparison_radar_or_bar,
    plot_target_distribution_donut,
    plot_continuous_distribution,
    plot_correlation_matrix,
    THEME
)

# ---------------------------------------------------------
# Page Configuration & Healthcare Aesthetic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk ML — Research Analytics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "cardio_train.csv")
DATA_CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "cardio_cleaned.csv")
METRICS_PATH = os.path.join(BASE_DIR, "models", "all_models_metrics.json")
IMPORTANCES_PATH = os.path.join(BASE_DIR, "models", "feature_importances.json")
COMPARISON_PATH = os.path.join(BASE_DIR, "models", "model_comparison.csv")
AUDIT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaning_audit.json")


# ---------------------------------------------------------
# Custom CSS for Premium Healthcare Design
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Top Header Bar */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 800px;
        line-height: 1.5;
        margin-bottom: 16px;
    }
    .hero-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(13, 148, 136, 0.18);
        border: 1px solid rgba(13, 148, 136, 0.4);
        color: #2DD4BF;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .hero-badge-alt {
        background: rgba(14, 165, 233, 0.18);
        border: 1px solid rgba(14, 165, 233, 0.4);
        color: #38BDF8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .metric-caption {
        font-size: 0.8rem;
        color: #0D9488;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Clinical Disclaimer Box */
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 18px 0;
        color: #92400E;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    
    /* Result Cards */
    .result-card-positive {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 14px;
        padding: 24px;
        color: #991B1B;
    }
    .result-card-negative {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 14px;
        padding: 24px;
        color: #166534;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
        color: #E2E8F0 !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] .stButton {
        width: 100%;
        text-align: left;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #E2E8F0 !important;
        text-align: left;
        padding: 0.1rem 0.3rem !important;
        font-size: 1.05rem;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(148, 163, 184, 0.08) !important;
        color: #F8FAFC !important;
    }
    
    /* Tabs & Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data & Artifact Loaders
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_CLEAN_PATH):
        df = pd.read_csv(DATA_CLEAN_PATH)
    else:
        df = load_raw_data(DATA_RAW_PATH)
    # Add age_years if not present
    if "age_years" not in df.columns and "age" in df.columns:
        df["age_years"] = (df["age"] / 365.25).round(1)
    if "bmi" not in df.columns and "height" in df.columns and "weight" in df.columns:
        df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(1)
    if "pulse_pressure" not in df.columns and "ap_hi" in df.columns and "ap_lo" in df.columns:
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    if "map" not in df.columns and "ap_hi" in df.columns and "ap_lo" in df.columns:
        df["map"] = (df["ap_lo"] + (df["ap_hi"] - df["ap_lo"]) / 3.0).round(1)
    return df


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return []


@st.cache_data
def load_comparison_df():
    if os.path.exists(COMPARISON_PATH):
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_data
def load_feature_importances():
    if os.path.exists(IMPORTANCES_PATH):
        with open(IMPORTANCES_PATH, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_cleaning_audit():
    if os.path.exists(AUDIT_PATH):
        with open(AUDIT_PATH, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 12px 0 18px 0;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>🫀 CardioRisk ML</span>
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;">
                Research Analytics Platform
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size: 1.1rem; font-weight: 700; color: #E2E8F0; margin: 8px 0 10px 0;">Navigation</div>', unsafe_allow_html=True)

    nav_options = [
        "1. Executive Overview",
        "2. Exploratory Data Analysis",
        "3. Model Laboratory",
        "4. Model Evaluation",
        "5. Feature Insights",
        "6. Research Prediction Interface",
        "7. Dataset Explorer",
        "8. Project Information",
    ]

    if "active_page" not in st.session_state:
        st.session_state.active_page = nav_options[0]

    for option in nav_options:
        label = option.split(". ", 1)[1] if ". " in option else option
        is_active = option == st.session_state.active_page
        button_label = f"{'● ' if is_active else '○ '}{label}"
        if st.button(button_label, key=f"nav_{option}", use_container_width=True):
            st.session_state.active_page = option
            st.rerun()

    page = st.session_state.active_page
    
    st.markdown("---")
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid #334155; border-radius: 10px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Active Model</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #38BDF8; margin-top: 2px;">Random Forest (Benchmark)</div>
            <div style="font-size: 0.75rem; color: #2DD4BF; margin-top: 4px;">Test ROC-AUC: <b>0.7942</b></div>
            <div style="font-size: 0.72rem; color: #CBD5E1; margin-top: 6px;">5-Fold Stratified CV Applied</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.75rem; color: #64748B; line-height: 1.4;">
            <b>Educational & Research Demonstration</b><br>
            MCA Machine Learning Project<br>
            Not for Clinical Diagnostics.
        </div>
    """, unsafe_allow_html=True)


# Load data and artifacts
df_clean = load_dataset()
metrics_list = load_metrics()
comp_df = load_comparison_df()
importances_dict = load_feature_importances()
cleaning_audit = load_cleaning_audit()


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "1. Executive Overview":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">
                <span>🫀 CardioRisk ML</span>
            </div>
            <div class="hero-subtitle">
                An end-to-end Machine Learning research and analytics system for cardiovascular disease risk classification,
                trained on 68,584 verified patient records with rigorous cross-validation and clinical explainability.
            </div>
            <div class="hero-badges">
                <span class="hero-badge">Verified Dataset (70K Records)</span>
                <span class="hero-badge-alt">Zero Data Leakage Pipeline</span>
                <span class="hero-badge">Stratified 5-Fold CV</span>
                <span class="hero-badge-alt">Tableau Ready</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cleaned Cohort Records</div>
                <div class="metric-value">{len(df_clean):,}</div>
                <div class="metric-caption">97.98% retained from 70,000 raw</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Class Balance Ratio</div>
                <div class="metric-value">50.5% / 49.5%</div>
                <div class="metric-caption">Balanced: 34.6k Healthy | 33.9k CVD</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Champion Algorithm</div>
                <div class="metric-value" style="font-size: 1.4rem; padding-top: 6px;">Random Forest</div>
                <div class="metric-caption">Highest Cross-Validation ROC-AUC</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Champion ROC-AUC</div>
                <div class="metric-value">0.7942</div>
                <div class="metric-caption">Held-Out Test Set (13,717 cases)</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Content Split
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 🏆 Algorithm Benchmark Performance")
        st.markdown(
            "All four algorithms were trained using identical stratified 80/20 train/test splits and evaluated on strictly held-out patient data. "
            "Feature scaling and clinical feature engineering were strictly fit inside cross-validation folds to avoid data leakage."
        )
        if not comp_df.empty:
            st.dataframe(
                comp_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall (Sensitivity)": "{:.4f}",
                    "Specificity": "{:.4f}",
                    "F1-Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "PR-AUC": "{:.4f}"
                }).highlight_max(subset=["Accuracy", "ROC-AUC", "F1-Score"], color="#CCFBF1"),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("""
            <div class="disclaimer-box">
                <b>Academic & Portfolio Disclaimer:</b> CardioRisk ML is developed for educational research, 
                MCA technical curriculum demonstration, and ML engineering portfolio review. It is not approved for 
                clinical diagnosis, patient triaging, or individual medical risk estimation.
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📊 Cohort Class Distribution")
        donut_fig = plot_target_distribution_donut(df_clean["cardio"].value_counts().to_dict())
        st.plotly_chart(donut_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚡ Quick Research Navigation")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.info("🔍 **Exploratory Data Analysis**\n\nDeep-dive into clinical distributions, blood pressure anomalies, correlations, and patient demographics.")
    with q2:
        st.success("🔬 **Model Laboratory & Evaluation**\n\nInspect ROC curves, confusion matrices, false positive/negative analysis, and cross-validation variance.")
    with q3:
        st.warning("🩺 **Research Prediction Interface**\n\nSimulate patient profiles with live clinical feature engineering (BMI, Pulse Pressure, MAP) and risk scoring.")


# ==============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif page == "2. Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis & Physiological Profiling")
    st.markdown(
        "Explore distributions, demographic risk stratifications, and clinical feature relationships across the 68,584 verified patient records."
    )
    
    # Interactive Filters Row
    with st.expander("🛠️ Interactive Cohort Subsetting Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            gender_filter = st.selectbox("Filter Gender", ["All", "Women (Code 1)", "Men (Code 2)"])
        with f2:
            smoke_filter = st.selectbox("Smoking Status", ["All", "Non-Smokers Only", "Smokers Only"])
        with f3:
            age_range = st.slider("Age Range (Years)", min_value=30, max_value=65, value=(30, 65))
            
    # Apply filtering
    df_filtered = df_clean.copy()
    if gender_filter == "Women (Code 1)":
        df_filtered = df_filtered[df_filtered["gender"] == 1]
    elif gender_filter == "Men (Code 2)":
        df_filtered = df_filtered[df_filtered["gender"] == 2]
        
    if smoke_filter == "Non-Smokers Only":
        df_filtered = df_filtered[df_filtered["smoke"] == 0]
    elif smoke_filter == "Smokers Only":
        df_filtered = df_filtered[df_filtered["smoke"] == 1]
        
    df_filtered = df_filtered[(df_filtered["age_years"] >= age_range[0]) & (df_filtered["age_years"] <= age_range[1])]
    
    st.caption(f"Currently displaying **{len(df_filtered):,}** records based on active filters.")
    
    # Tabs for EDA
    t1, t2, t3, t4 = st.tabs(["Hemodynamics & Blood Pressure", "Demographics & Body Habitus", "Biochemical & Lifestyle Markers", "Correlation Matrix"])
    
    with t1:
        st.markdown("#### Hemodynamic Predictors: Systolic vs Diastolic Blood Pressure")
        st.markdown(
            "Blood pressure is the single highest-weighted predictor across all algorithms. "
            "Notice how individuals with cardiovascular disease cluster prominently in hypertensive regions (ap_hi > 130 mmHg and ap_lo > 85 mmHg)."
        )
        col_bp1, col_bp2 = st.columns(2)
        with col_bp1:
            fig_sbp = plot_continuous_distribution(df_filtered, "ap_hi", "Systolic Blood Pressure (ap_hi)")
            st.plotly_chart(fig_sbp, use_container_width=True)
            st.caption("Interpretation: Patients with CVD exhibit a median systolic blood pressure of ~130-140 mmHg compared to 120 mmHg in healthy controls.")
        with col_bp2:
            fig_pp = plot_continuous_distribution(df_filtered, "pulse_pressure", "Derived Pulse Pressure (ap_hi - ap_lo)")
            st.plotly_chart(fig_pp, use_container_width=True)
            st.caption("Interpretation: Pulse pressure is an established clinical proxy for arterial stiffening, showing marked elevation in the disease cohort.")

    with t2:
        st.markdown("#### Age & Body Mass Index (BMI)")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_age = plot_continuous_distribution(df_filtered, "age_years", "Patient Age (Years)")
            st.plotly_chart(fig_age, use_container_width=True)
            st.caption("Interpretation: Cardiovascular disease prevalence scales steeply past 50 years of age in this cohort.")
        with col_d2:
            fig_bmi = plot_continuous_distribution(df_filtered, "bmi", "Body Mass Index (BMI kg/m²)")
            st.plotly_chart(fig_bmi, use_container_width=True)
            st.caption("Interpretation: The disease cohort shows higher central tendency towards overweight and obesity (BMI > 27.5 kg/m²).")

    with t3:
        st.markdown("#### Categorical Risk Factor Prevalence")
        st.markdown("Analysis of serum cholesterol, blood glucose, and behavioral lifestyle attributes.")
        c_cat1, c_cat2 = st.columns(2)
        with c_cat1:
            # Cholesterol rate by class
            chol_agg = df_filtered.groupby(["cholesterol", "cardio"]).size().reset_index(name="count")
            chol_agg["cardio_label"] = chol_agg["cardio"].map({0: "Healthy (0)", 1: "CVD (1)"})
            chol_agg["chol_label"] = chol_agg["cholesterol"].map({1: "1: Normal", 2: "2: Above Normal", 3: "3: Well Above"})
            fig_chol = px.bar(
                chol_agg, x="chol_label", y="count", color="cardio_label",
                barmode="group",
                color_discrete_map={"Healthy (0)": "#0EA5E9", "CVD (1)": "#F43F5E"},
                title="<b>Disease Prevalence by Serum Cholesterol Tier</b>"
            )
            st.plotly_chart(fig_chol, use_container_width=True)
            st.caption("Interpretation: While category 1 contains the bulk of patients, categories 2 and 3 exhibit a disproportionate skew toward CVD.")
        with c_cat2:
            # Active vs Inactive
            act_agg = df_filtered.groupby(["active", "cardio"]).size().reset_index(name="count")
            act_agg["cardio_label"] = act_agg["cardio"].map({0: "Healthy (0)", 1: "CVD (1)"})
            act_agg["active_label"] = act_agg["active"].map({0: "Sedentary (0)", 1: "Active (1)"})
            fig_act = px.bar(
                act_agg, x="active_label", y="count", color="cardio_label",
                barmode="group",
                color_discrete_map={"Healthy (0)": "#0EA5E9", "CVD (1)": "#F43F5E"},
                title="<b>Disease Prevalence by Physical Activity</b>"
            )
            st.plotly_chart(fig_act, use_container_width=True)
            st.caption("Interpretation: Physical inactivity correlates with higher proportion of cardiovascular disease.")

    with t4:
        st.markdown("#### Clinical Variable Correlation Matrix")
        st.markdown("Pearson correlation across continuous physiological markers and the target variable.")
        corr_fig = plot_correlation_matrix(df_filtered)
        st.plotly_chart(corr_fig, use_container_width=True)
        st.caption("Notice the strong positive correlation between `ap_hi`, `ap_lo`, `map`, and `cardio`, followed by `age_years` and `cholesterol`.")


# ==============================================================================
# PAGE 3: MODEL LABORATORY
# ==============================================================================
elif page == "3. Model Laboratory":
    st.title("🔬 Model Laboratory & Algorithmic Comparison")
    st.markdown(
        "Compare four classification algorithms evaluated under identical stratified cross-validation conditions. "
        "Hyperparameters were constrained to balance model complexity and avoid overfitting."
    )
    
    # Comparison chart
    if not comp_df.empty:
        st.markdown("### 📊 Cross-Model Performance Overview")
        comp_bar_fig = plot_model_comparison_radar_or_bar(comp_df)
        st.plotly_chart(comp_bar_fig, use_container_width=True)
        
        st.markdown("### 📋 Detailed Benchmark Metric Table")
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall (Sensitivity)": "{:.4f}",
                "Specificity": "{:.4f}",
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### ⚙️ Algorithm Architecture & Inductive Bias")
    
    m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
        "Logistic Regression",
        "Decision Tree Classifier",
        "Support Vector Machine",
        "Random Forest (Benchmark)"
    ])
    
    with m_tab1:
        st.markdown("""
            **Algorithm:** Regularized Logistic Regression (L2 penalty)  
            - **Solver:** L-BFGS, `max_iter=1000`, `C=1.0`, balanced class weighting.  
            - **Strengths:** Highly interpretable log-odds coefficients, fast convergence, direct calibrated probability estimates.  
            - **Inductive Bias:** Assumes a linear decision boundary in log-odds space between physiological predictors and CVD presence.  
            - **Validation Performance:** Mean CV Accuracy: `72.86%` | Test ROC-AUC: `0.7860`.
        """)
        
    with m_tab2:
        st.markdown("""
            **Algorithm:** Single Decision Tree Classifier  
            - **Hyperparameters:** `max_depth=6`, `min_samples_split=25`, `min_samples_leaf=15`, balanced class weighting.  
            - **Strengths:** Explicit hierarchical thresholding (e.g. if `ap_hi > 129.5`), captures non-linear interactions without feature scaling.  
            - **Inductive Bias:** Axis-aligned orthogonal decision boundaries. Depth capped at 6 to prevent severe leaf-node overfitting.  
            - **Validation Performance:** Mean CV Accuracy: `73.18%` | Test ROC-AUC: `0.7852`.
        """)
        
    with m_tab3:
        st.markdown("""
            **Algorithm:** Support Vector Machine (Linear SVM with CalibratedClassifierCV)  
            - **Hyperparameters:** `LinearSVC(C=0.5, dual=False)` with 3-fold Platt scaling calibration.  
            - **Strengths:** Maximum margin hyperplane separation, robust against high-dimensional collinearity.  
            - **Inductive Bias:** Maximizes the geometric margin between support vectors of healthy vs CVD cohorts.  
            - **Validation Performance:** Mean CV Accuracy: `72.79%` | Test ROC-AUC: `0.7857`.
        """)
        
    with m_tab4:
        st.markdown("""
            **Algorithm:** Random Forest Classifier (Selected Champion)  
            - **Hyperparameters:** Ensemble of 100 decision trees (`n_estimators=100`, `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`).  
            - **Strengths:** Reduces single-tree variance through bootstrap bagging and random feature subsampling. Handles correlated hemodynamic features smoothly.  
            - **Inductive Bias:** Ensemble bagging over decorrelated decision trees.  
            - **Validation Performance:** Mean CV Accuracy: `73.54%` | Test ROC-AUC: `0.7942` (Highest overall).
        """)


# ==============================================================================
# PAGE 4: MODEL EVALUATION
# ==============================================================================
elif page == "4. Model Evaluation":
    st.title("📈 Model Evaluation & Diagnostic Curves")
    st.markdown(
        "Examine granular diagnostic performance on the held-out test cohort (13,717 cases). "
        "Select any of the four trained algorithms to inspect its confusion matrix, ROC curve, and classification report."
    )
    
    # Model Selector
    model_options = [m["model_name"] for m in metrics_list] if metrics_list else ["Random Forest (Benchmark)"]
    selected_model_name = st.selectbox("Select Model to Evaluate", model_options, index=0)
    
    selected_metric = next((m for m in metrics_list if m["model_name"] == selected_model_name), None)
    
    if selected_metric:
        # Metrics summary cards
        c_e1, c_e2, c_e3, c_e4, c_e5, c_e6 = st.columns(6)
        c_e1.metric("Accuracy", f"{selected_metric['accuracy']:.4f}")
        c_e2.metric("ROC-AUC", f"{selected_metric['roc_auc']:.4f}")
        c_e3.metric("F1-Score", f"{selected_metric['f1']:.4f}")
        c_e4.metric("Precision", f"{selected_metric['precision']:.4f}")
        c_e5.metric("Recall (Sensitivity)", f"{selected_metric['recall']:.4f}")
        c_e6.metric("Specificity", f"{selected_metric['specificity']:.4f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confusion matrix + Curves
        col_cm, col_curves = st.columns([1, 1])
        with col_cm:
            st.markdown("### 🔲 Confusion Matrix")
            cm_fig = plot_confusion_matrix(selected_metric["confusion_matrix"], selected_model_name)
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # Clinical error explanation
            cm_vals = selected_metric["confusion_matrix"]
            st.markdown(f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px; font-size: 0.86rem;">
                    <b>Clinical Error Interpretation:</b>
                    <ul style="margin-top: 6px; padding-left: 18px; margin-bottom: 0;">
                        <li><b>False Negatives ({cm_vals['fn']:,} patients):</b> The model predicted healthy, but the patient had cardiovascular disease. In clinical medicine, this is the most critical error as it delays life-saving interventions.</li>
                        <li><b>False Positives ({cm_vals['fp']:,} patients):</b> The model predicted disease, but the patient was healthy. This results in anxiety and unnecessary secondary diagnostic testing.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col_curves:
            curve_tab1, curve_tab2 = st.tabs(["ROC Curve", "Precision-Recall Curve"])
            with curve_tab1:
                roc_fig = plot_roc_curve(selected_metric["roc_curve"], selected_model_name, selected_metric["roc_auc"])
                st.plotly_chart(roc_fig, use_container_width=True)
            with curve_tab2:
                pr_fig = plot_precision_recall_curve(selected_metric["pr_curve"], selected_model_name, selected_metric["pr_auc"])
                st.plotly_chart(pr_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Full Classification Report")
        cr = selected_metric["classification_report"]
        cr_rows = []
        for class_key in ["0", "1"]:
            if class_key in cr:
                cr_rows.append({
                    "Cohort": "Healthy (0)" if class_key == "0" else "Cardiovascular Disease (1)",
                    "Precision": round(cr[class_key]["precision"], 4),
                    "Recall": round(cr[class_key]["recall"], 4),
                    "F1-Score": round(cr[class_key]["f1-score"], 4),
                    "Support": int(cr[class_key]["support"])
                })
        st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 5: FEATURE INSIGHTS
# ==============================================================================
elif page == "5. Feature Insights":
    st.title("💡 Feature Insights & Model Explainability")
    st.markdown(
        "Understand the relative weight and discriminative contribution of clinical markers across models. "
        "Derived physiological metrics (`map`, `pulse_pressure`, `bmi`, `age_years`) play a dominant role in risk stratification."
    )
    
    # Model Selector for Feature Importance
    model_opts = list(importances_dict.keys()) if importances_dict else ["Random Forest (Benchmark)"]
    selected_fi_model = st.selectbox("Select Model for Feature Attribution", model_opts, index=0)
    
    if selected_fi_model in importances_dict:
        col_fi1, col_fi2 = st.columns([3, 2])
        with col_fi1:
            fi_fig = plot_feature_importance_bar(importances_dict[selected_fi_model], selected_fi_model, top_n=12)
            st.plotly_chart(fi_fig, use_container_width=True)
            
        with col_fi2:
            st.markdown("### 🩺 Clinical Analysis of Key Drivers")
            st.markdown("""
                1. **Blood Pressure (`ap_hi`, `map`, `pulse_pressure`):**  
                   Consistently accounts for over 45% of total predictive weight. Systolic hypertension exerts direct shear stress on the vascular endothelium.
                2. **Patient Age (`age_years`):**  
                   Non-modifiable biological risk factor reflecting cumulative lifetime vascular exposure and arterial compliance loss.
                3. **Serum Cholesterol (`cholesterol`):**  
                   Patients in tiers 2 & 3 show strong positive association with coronary plaque burden.
                4. **Body Mass Index (`bmi`) & Weight:**  
                   Adiposity contributes moderately, frequently interacting with blood pressure and glucose tolerance.
            """)
            
            st.markdown("""
                <div class="disclaimer-box">
                    <b>Causation vs Association:</b> Feature weights represent statistical correlation patterns within 
                    this observational cohort. They do <b>NOT</b> prove direct medical causation for an individual patient.
                </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: RESEARCH PREDICTION INTERFACE
# ==============================================================================
elif page == "6. Research Prediction Interface":
    st.title("🩺 Research Prediction Interface")
    st.markdown(
        "Enter clinical examination values below to evaluate cardiovascular disease likelihood using the trained ML pipeline. "
        "Derived clinical metrics (`BMI`, `Pulse Pressure`, `Mean Arterial Pressure`) are computed dynamically."
    )
    
    # Presets for demonstration
    st.markdown("#### ⚡ Quick Clinical Presets")
    col_pre1, col_pre2, col_pre3, _ = st.columns([1, 1, 1, 2])
    
    if "p_age" not in st.session_state:
        st.session_state.p_age = 52
        st.session_state.p_gender = "Male"
        st.session_state.p_height = 172
        st.session_state.p_weight = 78.0
        st.session_state.p_ap_hi = 125
        st.session_state.p_ap_lo = 80
        st.session_state.p_chol = "Normal (1)"
        st.session_state.p_gluc = "Normal (1)"
        st.session_state.p_smoke = False
        st.session_state.p_alco = False
        st.session_state.p_active = True

    if col_pre1.button("🏃 Low Risk Profile"):
        st.session_state.p_age = 35
        st.session_state.p_gender = "Female"
        st.session_state.p_height = 165
        st.session_state.p_weight = 60.0
        st.session_state.p_ap_hi = 110
        st.session_state.p_ap_lo = 70
        st.session_state.p_chol = "Normal (1)"
        st.session_state.p_gluc = "Normal (1)"
        st.session_state.p_smoke = False
        st.session_state.p_alco = False
        st.session_state.p_active = True
        st.rerun()

    if col_pre2.button("⚠️ Hypertensive Patient"):
        st.session_state.p_age = 58
        st.session_state.p_gender = "Male"
        st.session_state.p_height = 175
        st.session_state.p_weight = 88.0
        st.session_state.p_ap_hi = 155
        st.session_state.p_ap_lo = 95
        st.session_state.p_chol = "Well Above Normal (3)"
        st.session_state.p_gluc = "Above Normal (2)"
        st.session_state.p_smoke = True
        st.session_state.p_alco = False
        st.session_state.p_active = False
        st.rerun()

    if col_pre3.button("⚖️ Borderline Case"):
        st.session_state.p_age = 50
        st.session_state.p_gender = "Female"
        st.session_state.p_height = 160
        st.session_state.p_weight = 72.0
        st.session_state.p_ap_hi = 130
        st.session_state.p_ap_lo = 85
        st.session_state.p_chol = "Above Normal (2)"
        st.session_state.p_gluc = "Normal (1)"
        st.session_state.p_smoke = False
        st.session_state.p_alco = False
        st.session_state.p_active = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form"):
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown("##### 👤 Demographics & Anthropometrics")
            age_input = st.number_input("Age (Years)", min_value=18, max_value=100, value=st.session_state.p_age, step=1)
            gender_input = st.selectbox("Biological Sex", ["Female", "Male"], index=0 if st.session_state.p_gender == "Female" else 1)
            height_input = st.number_input("Height (cm)", min_value=120, max_value=220, value=st.session_state.p_height, step=1)
            weight_input = st.number_input("Body Weight (kg)", min_value=30.0, max_value=200.0, value=float(st.session_state.p_weight), step=0.5)
            
            st.markdown("##### 🍷 Lifestyle Habits")
            smoke_input = st.checkbox("Active Tobacco Smoker", value=st.session_state.p_smoke)
            alco_input = st.checkbox("Consumes Alcohol Regularly", value=st.session_state.p_alco)
            active_input = st.checkbox("Physically Active (Exercise ≥150 min/wk)", value=st.session_state.p_active)
            
        with col_f2:
            st.markdown("##### 🩺 Hemodynamics & Blood Pressure")
            ap_hi_input = st.number_input("Systolic Blood Pressure — ap_hi (mmHg)", min_value=70, max_value=240, value=st.session_state.p_ap_hi, step=1)
            ap_lo_input = st.number_input("Diastolic Blood Pressure — ap_lo (mmHg)", min_value=40, max_value=140, value=st.session_state.p_ap_lo, step=1)
            
            st.markdown("##### 🧪 Serum Laboratory Markers")
            chol_options = ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"]
            chol_idx = 0 if "Normal" in st.session_state.p_chol and "Above" not in st.session_state.p_chol else (1 if "Above Normal (2)" in st.session_state.p_chol else 2)
            chol_input = st.selectbox("Serum Cholesterol Level", chol_options, index=chol_idx)
            
            gluc_options = ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"]
            gluc_idx = 0 if "Normal" in st.session_state.p_gluc and "Above" not in st.session_state.p_gluc else (1 if "Above Normal (2)" in st.session_state.p_gluc else 2)
            gluc_input = st.selectbox("Fasting Blood Glucose Level", gluc_options, index=gluc_idx)
            
            # Model Selection for Prediction
            pred_model = st.selectbox(
                "Prediction Model Engine",
                ["Random Forest (Benchmark)", "Logistic Regression", "Support Vector Machine", "Decision Tree"]
            )
            
        submitted = st.form_submit_button("🚀 Compute Cardiovascular Risk Assessment", use_container_width=True)

    if submitted:
        # Preprocessing & Validation
        if ap_lo_input >= ap_hi_input:
            st.error("⚠️ Invalid Blood Pressure: Systolic BP (ap_hi) must be strictly greater than Diastolic BP (ap_lo).")
        else:
            # Map inputs
            gender_val = 2 if gender_input == "Male" else 1
            chol_val = 1 if "Normal (1)" in chol_input else (2 if "Above Normal (2)" in chol_input else 3)
            gluc_val = 1 if "Normal (1)" in gluc_input else (2 if "Above Normal (2)" in gluc_input else 3)
            
            patient_payload = {
                "age_years": float(age_input),
                "gender": gender_val,
                "height": int(height_input),
                "weight": float(weight_input),
                "ap_hi": int(ap_hi_input),
                "ap_lo": int(ap_lo_input),
                "cholesterol": chol_val,
                "gluc": gluc_val,
                "smoke": 1 if smoke_input else 0,
                "alco": 1 if alco_input else 0,
                "active": 1 if active_input else 0,
            }
            
            with st.spinner("Analyzing physiological markers via ML pipeline..."):
                result = predict_patient_risk(patient_payload, model_name=pred_model)
                
            st.markdown("### 📋 Prediction Diagnostic Report")
            
            # Risk Display Card
            if result["prediction_binary"] == 1:
                st.markdown(f"""
                    <div class="result-card-positive">
                        <div style="font-size: 1.3rem; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                            <span>⚠️ {result['prediction_label']}</span>
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; margin: 10px 0;">
                            {result['probability_percent']}% <span style="font-size: 1.1rem; font-weight: 600;">Probability Score</span>
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600;">
                            Classification Tier: <span style="color: {result['risk_color']};">{result['risk_tier']}</span>
                        </div>
                        <div style="font-size: 0.9rem; margin-top: 6px;">
                            {result['summary_message']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card-negative">
                        <div style="font-size: 1.3rem; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                            <span>✅ {result['prediction_label']}</span>
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; margin: 10px 0;">
                            {result['probability_percent']}% <span style="font-size: 1.1rem; font-weight: 600;">Probability Score</span>
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600;">
                            Classification Tier: <span style="color: {result['risk_color']};">{result['risk_tier']}</span>
                        </div>
                        <div style="font-size: 0.9rem; margin-top: 6px;">
                            {result['summary_message']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Derived Features & Clinical Flags
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("##### 📐 Derived Clinical Metrics")
                st.markdown(f"""
                    - **Body Mass Index (BMI):** `{result['computed_features']['bmi']} kg/m²`
                    - **Pulse Pressure (PP):** `{result['computed_features']['pulse_pressure']} mmHg` (SBP - DBP)
                    - **Mean Arterial Pressure (MAP):** `{result['computed_features']['mean_arterial_pressure']} mmHg` (Perfusion marker)
                    - **Evaluated Model:** `{pred_model}`
                """)
            with col_res2:
                st.markdown("##### 🚩 Clinical Flag Alerts")
                if result["clinical_flags"]:
                    for flag in result["clinical_flags"]:
                        st.markdown(f"- ⚠️ {flag}")
                else:
                    st.markdown("- ✅ All examined markers within baseline normative ranges.")
                    
            st.markdown(f"""
                <div class="disclaimer-box">
                    <b>Ethical Research Notice:</b> {result['disclaimer']}
                </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: DATASET EXPLORER
# ==============================================================================
elif page == "7. Dataset Explorer":
    st.title("📂 Dataset Explorer & Quality Audit")
    st.markdown(
        "Inspect the verified cohort data, review cleaning transformations, and download cleaned analytic records."
    )
    
    # Audit Summary Cards
    if cleaning_audit:
        ca1, ca2, ca3, ca4 = st.columns(4)
        ca1.metric("Raw Dataset Size", f"{cleaning_audit.get('initial_rows', 70000):,} rows")
        ca2.metric("Duplicates Removed", f"{cleaning_audit.get('duplicate_rows_removed', 24):,}")
        ca3.metric("Outliers Filtered", f"{cleaning_audit.get('outlier_rows_removed', 1392):,}")
        ca4.metric("Retention Rate", f"{cleaning_audit.get('retention_percentage', 97.98)}%")
        
    st.markdown("---")
    
    # Search and Filter
    st.markdown("### 🔎 Interactive Data Table")
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        target_view = st.radio("Cohort View", ["All Cohort", "Healthy Only (cardio = 0)", "CVD Only (cardio = 1)"], horizontal=True)
    with col_s2:
        n_rows = st.slider("Display Rows", min_value=10, max_value=200, value=50, step=10)
        
    df_view = df_clean.copy()
    if target_view == "Healthy Only (cardio = 0)":
        df_view = df_view[df_view["cardio"] == 0]
    elif target_view == "CVD Only (cardio = 1)":
        df_view = df_view[df_view["cardio"] == 1]
        
    st.dataframe(df_view.head(n_rows), use_container_width=True)
    
    # Export button
    csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv_bytes,
        file_name="cardio_cleaned.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==============================================================================
# PAGE 8: PROJECT INFORMATION
# ==============================================================================
elif page == "8. Project Information":
    st.title("ℹ️ Project Architecture & Documentation")
    st.markdown(
        "Technical documentation, data provenance, methodology, and MCA viva project reference materials."
    )
    
    t_info1, t_info2, t_info3, t_info4 = st.tabs([
        "Project Scope & Objectives",
        "Data Dictionary & Provenance",
        "ML Engineering Methodology",
        "Tableau BI Integration & Viva Questions"
    ])
    
    with t_info1:
        st.markdown("""
            ### 🎯 Project Purpose
            **CardioRisk ML** was engineered as an advanced Machine Learning research application and technical portfolio piece for:
            - **MCA Technical Demonstration:** Applying rigorous statistical learning, Scikit-learn pipelines, and interactive deployment.
            - **Preventing Target & Preprocessing Leakage:** Ensuring all scalers and transforms fit exclusively on training data.
            - **Healthcare UI/UX Standards:** Developing a clean, accessible, modern clinical dashboard rather than default Streamlit widgets.
            - **Multi-Algorithm Comparison:** Evaluating linear, tree-based, margin-based, and ensemble algorithms under identical Stratified 5-Fold CV.
        """)

    with t_info2:
        st.markdown("""
            ### 📚 Data Dictionary & Provenance
            - **Dataset Source:** Cardiovascular Disease dataset (70,000 patient records).
            - **Target Variable:** `cardio` (Binary: `0` = Absence, `1` = Presence of Cardiovascular Disease).
        """)
        dict_data = [
            {"Feature": "age / age_years", "Type": "Integer / Float", "Unit": "Days / Years", "Description": "Patient age at examination"},
            {"Feature": "gender", "Type": "Categorical", "Unit": "Code", "Description": "1: Female, 2: Male"},
            {"Feature": "height", "Type": "Integer", "Unit": "cm", "Description": "Patient standing height"},
            {"Feature": "weight", "Type": "Float", "Unit": "kg", "Description": "Patient body mass"},
            {"Feature": "ap_hi", "Type": "Integer", "Unit": "mmHg", "Description": "Systolic blood pressure"},
            {"Feature": "ap_lo", "Type": "Integer", "Unit": "mmHg", "Description": "Diastolic blood pressure"},
            {"Feature": "cholesterol", "Type": "Ordinal", "Unit": "1, 2, 3", "Description": "1: Normal, 2: Above Normal, 3: Well Above Normal"},
            {"Feature": "gluc", "Type": "Ordinal", "Unit": "1, 2, 3", "Description": "1: Normal, 2: Above Normal, 3: Well Above Normal"},
            {"Feature": "smoke", "Type": "Binary", "Unit": "0 / 1", "Description": "Tobacco smoking status"},
            {"Feature": "alco", "Type": "Binary", "Unit": "0 / 1", "Description": "Alcohol consumption status"},
            {"Feature": "active", "Type": "Binary", "Unit": "0 / 1", "Description": "Regular physical activity"},
            {"Feature": "bmi (Derived)", "Type": "Float", "Unit": "kg/m²", "Description": "Weight / (Height in meters)²"},
            {"Feature": "pulse_pressure (Derived)", "Type": "Integer", "Unit": "mmHg", "Description": "ap_hi - ap_lo (arterial stiffness index)"},
            {"Feature": "map (Derived)", "Type": "Float", "Unit": "mmHg", "Description": "ap_lo + (ap_hi - ap_lo) / 3.0 (Mean Arterial Pressure)"},
            {"Feature": "cardio (Target)", "Type": "Binary", "Unit": "0 / 1", "Description": "0: Negative (Healthy), 1: Positive (Cardiovascular Disease)"},
        ]
        st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)

    with t_info3:
        st.markdown(r"""
            ### 🔬 ML Engineering & Leakage Prevention
            1. **Zero Data Leakage:**
               - All scaling parameters ($\mu, \sigma$) and ordinal mappings are computed strictly on the training fold.
               - Custom `ClinicalFeatureEngineer` transformer handles feature derivations within the `Pipeline`.
            2. **Physiological Noise Filtering:**
               - Outliers such as $ap\_hi > 240$ mmHg or $ap\_lo < 40$ mmHg or $ap\_lo > ap\_hi$ are filtered prior to splitting to eliminate recording noise.
            3. **Calibration & Probability Scoring:**
               - Linear SVM is wrapped in `CalibratedClassifierCV` via Platt scaling to produce well-calibrated posterior probabilities.
        """)

    with t_info4:
        st.markdown("""
            ### 📊 Tableau BI Dashboard Integration
            Dedicated Tableau CSVs are pre-generated in the `tableau/` directory:
            - `tableau_patient_analytics.csv` (Enriched demographics and clinical classifications)
            - `tableau_model_comparison.csv` (Cross-algorithm metrics comparison)
            - `tableau_test_predictions.csv` (Ground truth vs model predictions with probabilities)
            - `tableau_feature_importance.csv` (Ranked clinical risk factors)
            - `tableau_data_dictionary.csv` (Metadata and variable definitions)
            
            Refer to `tableau/TABLEAU_DASHBOARD_GUIDE.md` for full step-by-step connection and dashboard assembly instructions.
        """)
