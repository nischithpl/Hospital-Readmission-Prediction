"""
CarePredict Improved Dashboard — Hospital Readmission Analytics & Prediction
=============================================================================
Streamlit application supporting the Improved Machine Learning Pipeline with:
- 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Complete evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Precision-Recall and ROC curve visualizers
- Top feature importance visualizations
- Patient risk assessment simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import confusion_matrix
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CarePredict 2.0 | Improved Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "improved"
ORIG_MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "datasets" / "dataset1"
RESULTS_DIR = ROOT / "results"

st.markdown("""
<style>
.stApp { background: #F8FAFC; color: #1E293B; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
h1, h2, h3 { color: #0F172A !important; font-weight: 700; }
section[data-testid="stSidebar"] { background: #0F172A; }
section[data-testid="stSidebar"] * { color: white !important; }
.hero {
    background: linear-gradient(135deg, #1E3A8A 0%, #0D9488 100%);
    padding: 28px;
    border-radius: 14px;
    margin-bottom: 22px;
    color: white;
}
.hero h1 { color: white !important; margin-bottom: 6px; }
.hero p { color: #E2E8F0 !important; font-size: 1.05rem; }
.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_improved_artifacts():
    artifacts = {}
    artifacts["models"] = {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression_model.pkl"),
        "Decision Tree": joblib.load(MODEL_DIR / "decision_tree_model.pkl"),
        "Random Forest": joblib.load(MODEL_DIR / "random_forest_model.pkl"),
        "XGBoost": joblib.load(MODEL_DIR / "xgboost_model.pkl")
    }
    artifacts["preprocessor"] = joblib.load(MODEL_DIR / "preprocessor.pkl")
    artifacts["thresholds"] = joblib.load(MODEL_DIR / "model_thresholds.pkl")
    artifacts["input_features"] = joblib.load(MODEL_DIR / "input_features.pkl")
    artifacts["metadata"] = joblib.load(MODEL_DIR / "feature_metadata.pkl")
    artifacts["evaluation"] = joblib.load(MODEL_DIR / "model_evaluation.pkl")
    artifacts["confusion_matrices"] = joblib.load(MODEL_DIR / "confusion_matrices.pkl")
    return artifacts

try:
    ART = load_improved_artifacts()
except Exception as e:
    st.error(f"Error loading improved model artifacts: {e}")
    st.info("Tip: Run `python notebooks/train_improved_pipeline.py` first to generate models and results.")
    st.stop()

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_DIR / "diabetic_data.csv")
    df.replace("?", np.nan, inplace=True)
    return df

df_raw = load_dataset()

# Sidebar navigation
st.sidebar.markdown("## 🏥 CarePredict 2.0")
st.sidebar.caption("Improved ML Pipeline Edition")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Executive Summary",
        "Pipeline Improvements",
        "Model Comparison & Metrics",
        "ROC & PR Curves",
        "Confusion Matrices",
        "Feature Importance",
        "Class Imbalance Analysis",
        "Patient Readmission Screener"
    ]
)

# 1. Executive Summary
if menu == "Executive Summary":
    st.markdown("""
    <div class="hero">
        <h1>🏥 CarePredict 2.0 — Executive Summary</h1>
        <p>A rigorous, leak-free machine learning framework for predicting 30-day hospital readmissions among diabetic inpatients.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Encounters Analyzed", f"{len(df_raw):,}")
    with col2:
        st.metric("Unique Patients", f"{df_raw['patient_nbr'].nunique():,}")
    with col3:
        st.metric("30-Day Readmission Rate", f"{(df_raw['readmitted'] == '<30').mean()*100:.2f}%")
    with col4:
        st.metric("Engineered Features", f"{len(ART['input_features'])} (205 dummy)")

    st.markdown("### Key Results Overview")
    comp_file = RESULTS_DIR / "full_baseline_vs_improved_comparison.csv"
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file)
        try:
            st.dataframe(comp_df, width="stretch")
        except TypeError:
            st.dataframe(comp_df, use_container_width=True)

    st.markdown("### Core Methodological Highlights")
    st.markdown("""
    - **Zero Data Leakage:** Group-based split (`GroupShuffleSplit` on `patient_nbr`) ensures patients never cross between training and test sets.
    - **Dense Clinical Representation:** Grouped 2,200+ sparse ICD-9 codes into 9 established clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, and Other).
    - **Cross-Validation:** 5-fold `StratifiedGroupKFold` cross-validation on 57,214 training patients.
    - **Balanced Cost-Sensitive Weighting:** Demonstrated superior calibration and PR-AUC over SMOTE oversampling.
    """)

# 2. Pipeline Improvements
elif menu == "Pipeline Improvements":
    st.title("⚙️ Preprocessing & Pipeline Improvements")
    st.markdown("""
    The improved pipeline resolves 5 fundamental vulnerabilities identified in the initial prototype:
    """)

    improvements = [
        {"Issue": "1. Data Leakage in Splitting", "Original Flaw": "Preprocessor was fit on a subtrain split, and threshold tuning was initially run on test data.", "Improvement": "GroupShuffleSplit (80/20) by patient_nbr. StratifiedGroupKFold on train set only. Preprocessor strictly fit on train set."},
        {"Issue": "2. High Dimensionality & Sparse Overfitting", "Original Flaw": "Over 2,200 sparse dummy variables for raw ICD-9 codes.", "Improvement": "ICD-9 codes grouped into 9 clinical categories. Reduced dimensions to 205 dense, clinically meaningful features."},
        {"Issue": "3. Utilization & Polypharmacy Signals", "Original Flaw": "Ignored clinical ratios and interaction terms.", "Improvement": "Created inpatient/emergency ratios, daily procedure/lab intensities, medication change/titration counts, and clinical complexity score."},
        {"Issue": "4. Model Exploration", "Original Flaw": "Only evaluated 3 models without Decision Tree baseline.", "Improvement": "Evaluated 4 algorithms: Logistic Regression, Decision Tree, Random Forest, and XGBoost."},
        {"Issue": "5. Evaluation Rigor", "Original Flaw": "Reported only Accuracy, Precision, Recall, F1, and ROC-AUC. Omitted PR-AUC.", "Improvement": "Added PR-AUC (Average Precision), out-of-fold threshold curves, and direct ablation comparisons."}
    ]
    st.table(pd.DataFrame(improvements))

# 3. Model Comparison & Metrics
elif menu == "Model Comparison & Metrics":
    st.title("📊 Model Comparison on Held-Out Test Set (20,153 Encounters)")
    eval_df = ART["evaluation"]
    styled_df = eval_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"], color="#D1FAE5")
    try:
        st.dataframe(styled_df, width="stretch")
    except TypeError:
        st.dataframe(styled_df, use_container_width=True)

    bar_path = RESULTS_DIR / "model_comparison_barchart.png"
    if bar_path.exists():
        try:
            st.image(str(bar_path), caption="Comparison of Key Metrics Across All 4 Models", width="stretch")
        except TypeError:
            st.image(str(bar_path), caption="Comparison of Key Metrics Across All 4 Models", use_container_width=True)

# 4. ROC & PR Curves
elif menu == "ROC & PR Curves":
    st.title("📈 ROC and Precision-Recall Curves")
    col1, col2 = st.columns(2)
    roc_path = RESULTS_DIR / "roc_curve_comparison_improved.png"
    pr_path = RESULTS_DIR / "pr_curve_comparison_improved.png"
    with col1:
        if roc_path.exists():
            try:
                st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curves", width="stretch")
            except TypeError:
                st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
    with col2:
        if pr_path.exists():
            try:
                st.image(str(pr_path), caption="Precision-Recall Curves (Critical for Imbalanced Data)", width="stretch")
            except TypeError:
                st.image(str(pr_path), caption="Precision-Recall Curves (Critical for Imbalanced Data)", use_container_width=True)

# 5. Confusion Matrices
elif menu == "Confusion Matrices":
    st.title("🎯 Confusion Matrices (Held-Out Test Set: 20,153 Encounters)")
    cm_path = RESULTS_DIR / "confusion_matrices_improved.png"
    if cm_path.exists():
        try:
            st.image(str(cm_path), caption="Confusion Matrices at Optimal F1 Thresholds", width="stretch")
        except TypeError:
            st.image(str(cm_path), caption="Confusion Matrices at Optimal F1 Thresholds", use_container_width=True)

    st.markdown("### Confusion Matrix Breakdown")
    cms = ART["confusion_matrices"]
    rows = []
    for model_name, matrix in cms.items():
        tn, fp, fn, tp = matrix.ravel()
        rows.append({
            "Model": model_name,
            "True Negatives": f"{tn:,}",
            "False Positives (False Alarms)": f"{fp:,}",
            "False Negatives (Missed Cases)": f"{fn:,}",
            "True Positives (Caught Cases)": f"{tp:,}",
            "Sensitivity / Recall": f"{tp/(tp+fn):.2%}",
            "Specificity": f"{tn/(tn+fp):.2%}"
        })
    try:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    except TypeError:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# 6. Feature Importance
elif menu == "Feature Importance":
    st.title("💡 Feature Importance Analysis")
    col1, col2 = st.columns(2)
    xgb_imp = RESULTS_DIR / "feature_importance_xgb_improved.png"
    rf_imp = RESULTS_DIR / "feature_importance_rf_improved.png"
    with col1:
        if xgb_imp.exists():
            try:
                st.image(str(xgb_imp), caption="XGBoost Top 20 Predictors (Gain)", width="stretch")
            except TypeError:
                st.image(str(xgb_imp), caption="XGBoost Top 20 Predictors (Gain)", use_container_width=True)
    with col2:
        if rf_imp.exists():
            try:
                st.image(str(rf_imp), caption="Random Forest Top 20 Predictors (Gini Impurity)", width="stretch")
            except TypeError:
                st.image(str(rf_imp), caption="Random Forest Top 20 Predictors (Gini Impurity)", use_container_width=True)

# 7. Class Imbalance Analysis
elif menu == "Class Imbalance Analysis":
    st.title("⚖️ Class Imbalance: Cost-Sensitive Weights vs SMOTE")
    smote_path = RESULTS_DIR / "smote_vs_class_weights.csv"
    if smote_path.exists():
        smote_df = pd.read_csv(smote_path)
        try:
            st.dataframe(smote_df, width="stretch")
        except TypeError:
            st.dataframe(smote_df, use_container_width=True)

    st.markdown("""
    ### Why Cost-Sensitive Weighting Outperforms SMOTE:
    1. **Discrete Clinical Coordinates:** In healthcare data with binary indicators and categorical one-hot variables, SMOTE creates synthetic interpolations that generate non-existent fractional states.
    2. **ROC-AUC & PR-AUC:** Empirical cross-validation showed `scale_pos_weight` achieves higher ROC-AUC (0.6823 vs 0.6738) and PR-AUC (0.2206 vs 0.2112).
    3. **Training Efficiency:** Class-weighting avoids artificial sample inflation, running 3x faster without risk of synthetic noise overfitting.
    """)

# 8. Patient Readmission Screener
elif menu == "Patient Readmission Screener":
    st.title("🩺 Patient Readmission Risk Screener")
    st.markdown("Simulate patient admission profiles and evaluate real-time readmission risk.")

    model_choice = st.selectbox("Select Model", list(ART["models"].keys()), index=3)
    chosen_model = ART["models"][model_choice]
    thresh = ART["thresholds"][model_choice]

    col1, col2, col3 = st.columns(3)
    with col1:
        age_mid = st.slider("Patient Age", 10, 95, 65, 5)
        time_hosp = st.slider("Hospital Stay (Days)", 1, 14, 4)
        num_inpatient = st.number_input("Prior Inpatient Visits", 0, 20, 1)
        num_emergency = st.number_input("Prior Emergency Visits", 0, 20, 0)
        num_outpatient = st.number_input("Prior Outpatient Visits", 0, 20, 0)

    with col2:
        num_meds = st.slider("Number of Medications", 1, 60, 15)
        num_labs = st.slider("Number of Lab Procedures", 1, 120, 45)
        num_procs = st.slider("Number of Procedures", 0, 6, 1)
        num_diags = st.slider("Number of Diagnoses", 1, 16, 7)
        primary_diag = st.selectbox("Primary Diagnosis Group", ["Circulatory", "Diabetes", "Respiratory", "Genitourinary", "Digestive", "Injury", "Musculoskeletal", "Neoplasms", "Other"])

    with col3:
        discharge_disp = st.selectbox("Discharge Destination", ["1 (Discharged to Home)", "3 (SNF)", "6 (Home with Home Health)", "2 (Short Term Hospital)", "22 (Rehab Facility)", "7 (Left AMA)", "11 (Expired)", "Other"])
        insulin_usage = st.selectbox("Insulin Treatment", ["No", "Steady", "Up", "Down"])
        med_change = st.selectbox("Diabetes Medication Changed during encounter?", ["No", "Ch"])
        a1c_res = st.selectbox("HbA1c Test Result", ["Not Tested", "Norm", ">7", ">8"])

    if st.button("Assess Readmission Risk", type="primary"):
        inp_df = pd.DataFrame(index=[0])
        for col in ART["metadata"]["numerical_columns"] + ART["metadata"]["binary_columns"]:
            inp_df[col] = ART["metadata"]["defaults"].get(col, 0.0)
        for col in ART["metadata"]["categorical_columns"]:
            inp_df[col] = ART["metadata"]["defaults"].get(col, "Unknown")

        inp_df["age_midpoint"] = float(age_mid)
        inp_df["is_senior"] = int(age_mid >= 65)
        inp_df["time_in_hospital"] = float(time_hosp)
        inp_df["number_inpatient"] = float(num_inpatient)
        inp_df["number_emergency"] = float(num_emergency)
        inp_df["number_outpatient"] = float(num_outpatient)
        inp_df["num_medications"] = float(num_meds)
        inp_df["num_lab_procedures"] = float(num_labs)
        inp_df["num_procedures"] = float(num_procs)
        inp_df["number_diagnoses"] = float(num_diags)

        tot_prior = num_inpatient + num_emergency + num_outpatient
        inp_df["total_prior_visits"] = float(tot_prior)
        inp_df["has_prior_inpatient"] = int(num_inpatient > 0)
        inp_df["has_prior_emergency"] = int(num_emergency > 0)
        inp_df["has_prior_outpatient"] = int(num_outpatient > 0)
        inp_df["inpatient_visit_ratio"] = float(num_inpatient / (tot_prior + 1.0))
        inp_df["emergency_visit_ratio"] = float(num_emergency / (tot_prior + 1.0))
        inp_df["prior_visit_intensity"] = float(tot_prior / (time_hosp + 1.0))
        inp_df["lab_procedures_per_day"] = float(num_labs / time_hosp)
        inp_df["procedures_per_day"] = float(num_procs / time_hosp)
        inp_df["medications_per_day"] = float(num_meds / time_hosp)
        inp_df["clinical_complexity"] = float(num_procs + (num_labs/10.0) + (num_meds/5.0) + num_diags)

        inp_df["diag_1_group"] = primary_diag
        inp_df["has_diag_diabetes"] = int(primary_diag == "Diabetes")
        inp_df["has_diag_circulatory"] = int(primary_diag == "Circulatory")
        inp_df["has_diag_respiratory"] = int(primary_diag == "Respiratory")
        inp_df["has_diag_genitourinary"] = int(primary_diag == "Genitourinary")

        disp_id = discharge_disp.split()[0]
        inp_df["discharge_disposition_id"] = str(disp_id)
        inp_df["insulin"] = insulin_usage
        inp_df["has_insulin"] = int(insulin_usage != "No")
        inp_df["change"] = med_change
        inp_df["med_changed_flag"] = int(med_change == "Ch")
        inp_df["A1Cresult"] = a1c_res
        inp_df["a1c_tested"] = int(a1c_res != "Not Tested")
        inp_df["a1c_high"] = int(a1c_res in [">7", ">8"])

        processed_input = ART["preprocessor"].transform(inp_df[ART["input_features"]])
        risk_prob = chosen_model.predict_proba(processed_input)[0, 1]
        is_readmitted = risk_prob >= thresh

        st.markdown("---")
        st.subheader("Assessment Outcome")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.metric("Predicted 30-Day Readmission Probability", f"{risk_prob*100:.2f}%")
            st.caption(f"Decision Threshold for {model_choice}: {thresh*100:.2f}%")
        with r_col2:
            if is_readmitted:
                st.error("⚠️ HIGH RISK OF 30-DAY READMISSION")
                st.write("Recommended Action: Arrange post-discharge clinical follow-up, home health support, or pharmacist medication reconciliation.")
            else:
                st.success("✅ LOW / STANDARD READMISSION RISK")
                st.write("Standard post-discharge instructions and scheduled primary care follow-up.")
