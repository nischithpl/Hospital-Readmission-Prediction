"""
CarePredict | Hospital Readmission Analytics & Prediction Dashboard
===================================================================
Undergraduate Machine Learning Project
Dataset: HLT-005 Synthetic Hospital Admission Dataset
Models: Logistic Regression, Decision Tree, Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="CarePredict | Hospital Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

# CarePredict classic styling
st.markdown("""
<style>
/* ============================================================ */
/* BASE THEME & LAYOUT                                          */
/* ============================================================ */

.stApp, [data-testid="stAppViewContainer"] {
    background-color: #F4F7FB !important;
    color: #263B50 !important;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}

/* Headings in main area */
h1, h2, h3, h4, h5, h6 {
    color: #153B5B !important;
    font-weight: 700 !important;
}

p, span {
    color: #34495E;
}

/* ============================================================ */
/* SIDEBAR (Dark Navy Background with White Text)               */
/* ============================================================ */

section[data-testid="stSidebar"] {
    background-color: #123650 !important;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: #214A63 !important;
    border-color: #45677D !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] * {
    color: #FFFFFF !important;
}

/* ============================================================ */
/* HERO BANNER                                                  */
/* ============================================================ */

.hero {
    background: linear-gradient(110deg, #123650, #176B87, #23988D);
    padding: 26px 30px;
    border-radius: 14px;
    margin-bottom: 22px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 30px;
    margin: 0;
    padding-bottom: 6px;
}

.hero p {
    color: #E5F3F6 !important;
    margin: 0;
    font-size: 15px;
}

/* Cards & Metrics in main view */
.card, div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #DFE7EF !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #153B5B !important;
}

/* ============================================================ */
/* PATIENT READMISSION PREDICTION FORM & LABELS (DARK & VISIBLE)*/
/* ============================================================ */

/* White Form Card */
[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #DCE6EF !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 2px 8px rgba(18, 54, 80, 0.04) !important;
}

/* Form header */
div[data-testid="stForm"] h5 {
    color: #153B5B !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
}

/* Force ALL widget labels to be dark navy and bold */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] *,
label[data-testid="stWidgetLabel"] p,
div[data-testid="stForm"] label,
div[data-testid="stForm"] label *,
div[data-testid="stForm"] label p,
div[data-testid="stForm"] span {
    color: #153B5B !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Number input fields and text inputs */
div[data-baseweb="input"] input,
div[data-testid="stNumberInput"] input {
    color: #123650 !important;
    background-color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

/* Input borders and backgrounds */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #C9D7E5 !important;
    border-radius: 8px !important;
}

/* Dropdown select option text */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #123650 !important;
    font-weight: 500 !important;
}

/* Number input step (+/-) buttons */
div[data-testid="stNumberInput"] button {
    background-color: #F4F7FB !important;
    border: 1px solid #C9D7E5 !important;
}

div[data-testid="stNumberInput"] button svg {
    fill: #153B5B !important;
    color: #153B5B !important;
}

/* Help tooltip icon */
[data-testid="stTooltipHoverTarget"] svg {
    fill: #176B87 !important;
}

/* ============================================================ */
/* BUTTONS & TABS                                               */
/* ============================================================ */

.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    background: #176B87 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 9px 20px !important;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #12546B !important;
    color: #FFFFFF !important;
}

/* Tab Headers */
button[data-baseweb="tab"] {
    color: #153B5B !important;
    font-weight: 600 !important;
    font-size: 14.5px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #176B87 !important;
    border-bottom-color: #176B87 !important;
}

/* Expanders & DataFrames */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE6EF !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] * {
    color: #153B5B !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #718493;
    font-size: 13px;
    padding: 24px 0 10px 0;
}
</style>""", unsafe_allow_html=True)

# ============================================================
# 2. LOAD TRAINED ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    try:
        art = {}
        art["models"] = {
            "Random Forest": joblib.load(MODEL_DIR / "random_forest_model.pkl"),
            "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression_model.pkl"),
            "Decision Tree": joblib.load(MODEL_DIR / "decision_tree_model.pkl")
        }
        art["preprocessor"] = joblib.load(MODEL_DIR / "preprocessor.pkl")
        art["evaluation"] = joblib.load(MODEL_DIR / "model_evaluation.pkl")
        art["thresholds"] = joblib.load(MODEL_DIR / "model_thresholds.pkl")
        art["confusion_matrices"] = joblib.load(MODEL_DIR / "confusion_matrices.pkl")
        art["input_features"] = joblib.load(MODEL_DIR / "input_features.pkl")
        art["metadata"] = joblib.load(MODEL_DIR / "feature_metadata.pkl")
        art["dataset_summary"] = joblib.load(MODEL_DIR / "dataset_summary.pkl")
        return art
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please ensure that `notebooks/train_pipeline.py` has been executed to generate all model files in `models/`.")
        st.stop()

ART = load_artifacts()
models = ART["models"]
preprocessor = ART["preprocessor"]
evaluation_df = ART["evaluation"]
thresholds = ART["thresholds"]
confusion_matrices = ART["confusion_matrices"]
input_features = ART["input_features"]
metadata = ART["metadata"]
dataset_summary = ART["dataset_summary"]

defaults = metadata["defaults"]
category_options = metadata["categorical_options"]

# ============================================================
# 3. SIDEBAR NAVIGATION & MODEL SELECTOR
# ============================================================

with st.sidebar:
    st.markdown("## 🏥 CarePredict")
    st.caption("Hospital Readmission Analytics Platform")
    st.markdown("---")
    
    st.markdown("### NAVIGATION")
    nav_selection = st.radio(
        "Select Section",
        [
            "Patient Readmission Prediction",
            "Model Performance Dashboard",
            "Dataset & Exploratory Analysis"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### MODEL SELECTION")
    selected_model_name = st.selectbox(
        "Active Prediction Model",
        list(models.keys()),
        index=0,
        help="Select the classification algorithm used for real-time patient risk evaluation."
    )
    
    model_threshold = thresholds.get(selected_model_name, 0.50)
    st.metric("Classification Threshold", f"{model_threshold:.2f}")
    
    st.caption("Academic ML Project — HLT-005 Synthetic Dataset.")

# ============================================================
# 4. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>CarePredict</h1>
    <p>Predictive Healthcare Analytics & 30-Day Hospital Readmission Risk Stratification</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION A: PATIENT READMISSION PREDICTION
# ============================================================

if nav_selection == "Patient Readmission Prediction":
    st.markdown("### 📋 Patient Readmission Risk Assessment")
    st.markdown(
        "Enter clinical indicators for the patient prior to discharge to evaluate the likelihood of "
        "an unplanned **30-day hospital readmission**. All inputs are processed through the trained "
        "scikit-learn preprocessing pipeline."
    )
    
    # Session state for form reset
    if "reset_trigger" not in st.session_state:
        st.session_state.reset_trigger = 0

    col_btn_reset, col_info = st.columns([1, 4])
    with col_btn_reset:
        if st.button("🔄 Reset to Default Values"):
            st.session_state.reset_trigger += 1
            st.rerun()
    with col_info:
        st.caption("Active Model: **" + selected_model_name + "** | Preprocessing: Standard Scaler + One-Hot Encoding")

    suffix = f"_{st.session_state.reset_trigger}"
    
    with st.form("patient_prediction_form"):
        st.markdown("##### Clinical Patient Indicators (7 Core Features)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(
                "1. Patient Age (Years)",
                min_value=0, max_value=120,
                value=int(defaults["age"]),
                step=1,
                help="Age in years at admission",
                key=f"age{suffix}"
            )
            los_days = st.number_input(
                "4. Length of Stay (Days)",
                min_value=0.5, max_value=90.0,
                value=float(defaults["los_days"]),
                step=0.5,
                help="Duration of current inpatient hospital stay",
                key=f"los{suffix}"
            )
            icu_admission = st.selectbox(
                "7. ICU Admission",
                options=["No", "Yes"],
                index=0 if defaults["icu_flag_str"] == "No" else 1,
                help="Was the patient admitted to the Intensive Care Unit during this stay?",
                key=f"icu{suffix}"
            )
            
        with c2:
            prior_admits = st.number_input(
                "2. Prior Admissions (Last 12 Mo)",
                min_value=0, max_value=20,
                value=int(defaults["prior_admits_12mo"]),
                step=1,
                help="Number of inpatient admissions in previous 12 months",
                key=f"prior_adm{suffix}"
            )
            cci_score = st.number_input(
                "5. Comorbidity Score (CCI)",
                min_value=0, max_value=20,
                value=int(defaults["cci_score"]),
                step=1,
                help="Charlson Comorbidity Index measuring chronic disease burden (0=low, 5+=high)",
                key=f"cci{suffix}"
            )
            
        with c3:
            prior_ed = st.number_input(
                "3. Prior ED Visits (Last 6-12 Mo)",
                min_value=0, max_value=20,
                value=int(defaults["prior_ed_visits_6mo"]),
                step=1,
                help="Emergency department visits in previous 6-12 months",
                key=f"prior_ed{suffix}"
            )
            diagnosis_cat = st.selectbox(
                "6. Primary Diagnosis Category",
                options=category_options["diagnosis_category"],
                index=category_options["diagnosis_category"].index(defaults["diagnosis_category"])
                if defaults["diagnosis_category"] in category_options["diagnosis_category"] else 0,
                help="Clinical diagnosis classification for current hospital admission",
                key=f"dx{suffix}"
            )

        submit_btn = st.form_submit_button("🔍 Predict Readmission Risk")

    if submit_btn:
        # Build DataFrame with exact feature names matching the pipeline
        input_data = pd.DataFrame([{
            "age": age,
            "prior_admits_12mo": prior_admits,
            "prior_ed_visits_6mo": prior_ed,
            "los_days": los_days,
            "cci_score": cci_score,
            "diagnosis_category": diagnosis_cat,
            "icu_flag_str": icu_admission
        }])[input_features]

        # Transform using saved preprocessor
        try:
            X_transformed = preprocessor.transform(input_data)
            active_model = models[selected_model_name]
            
            # Predict probability and class
            prob = float(active_model.predict_proba(X_transformed)[0, 1])
            is_likely = prob >= model_threshold
            
            # Display prominent outcome banner
            banner_color = "#E74C3C" if is_likely else "#27AE60"
            status_text = "⚠️ Readmission Likely" if is_likely else "✅ Readmission Not Likely"
            badge_text = "HIGH RISK" if is_likely else "LOW RISK"
            
            st.markdown(f"""
            <div style="background:{banner_color}; color:white; padding:22px 26px; border-radius:12px; margin:18px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="color:white !important; margin:0; font-size:26px;">{status_text}</h2>
                    <span style="background:rgba(255,255,255,0.25); padding:4px 14px; border-radius:20px; font-weight:bold; font-size:14px;">{badge_text}</span>
                </div>
                <p style="margin:10px 0 0 0; font-size:16px; color:#FDFEFE !important;">
                    Estimated 30-Day Readmission Probability: <b>{prob * 100:.1f}%</b> &nbsp;|&nbsp;
                    Decision Threshold: <b>{model_threshold * 100:.1f}%</b> &nbsp;|&nbsp;
                    Evaluator: <b>{selected_model_name}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key drivers & Clinical explanation
            st.markdown("#### Clinical Interpretation & Key Risk Drivers")
            drivers = []
            if prior_admits >= 2:
                drivers.append(f"• **Frequent Inpatient History**: {prior_admits} prior admissions in the last 12 months is a strong predictor of recurring healthcare utilization.")
            if cci_score >= 4:
                drivers.append(f"• **Elevated Comorbidity Burden**: Charlson Comorbidity Index of {cci_score} indicates multiple underlying chronic conditions.")
            if los_days >= 7.0:
                drivers.append(f"• **Prolonged Inpatient Stay**: Length of stay of {los_days:.1f} days reflects acute illness severity during hospitalization.")
            if icu_admission == "Yes":
                drivers.append("• **Critical Care Utilization**: Patient required ICU admission, reflecting higher clinical vulnerability.")
            if prior_ed >= 2:
                drivers.append(f"• **Emergency Department Reliance**: {prior_ed} prior ED visits indicate high ambulatory care needs.")
            if diagnosis_cat in ["Infection/Sepsis", "Cardiovascular", "Renal"]:
                drivers.append(f"• **High-Risk Clinical Category**: Primary diagnosis of *{diagnosis_cat}* carries higher average 30-day readmission risk.")

            if drivers:
                for d in drivers:
                    st.markdown(d)
            else:
                st.markdown("• **Low Acuity Profile**: Short hospital stay, low prior healthcare utilization, and manageable comorbidity burden place this patient in a lower risk category.")

            if is_likely:
                st.info("💡 **Recommended Action**: Flag for comprehensive post-discharge transition planning, medication reconciliation, and scheduled primary care follow-up within 7 days.")
            else:
                st.success("💡 **Recommended Action**: Proceed with standard discharge instructions and routine primary care follow-up.")

            # Downloadable prediction report
            st.markdown("---")
            report_df = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model": selected_model_name,
                "Prediction": "Readmission Likely (1)" if is_likely else "Readmission Not Likely (0)",
                "Readmission_Probability_Pct": round(prob * 100, 2),
                "Threshold": model_threshold,
                "Age": age,
                "Prior_Admissions_12mo": prior_admits,
                "Prior_ED_Visits_6mo": prior_ed,
                "Length_of_Stay_Days": los_days,
                "CCI_Score": cci_score,
                "Diagnosis_Category": diagnosis_cat,
                "ICU_Admission": icu_admission
            }])
            
            st.download_button(
                "📥 Download Prediction Summary (CSV)",
                report_df.to_csv(index=False),
                file_name=f"readmission_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check that all 7 input feature values match the expected format.")

# ============================================================
# SECTION B: MODEL PERFORMANCE & VISUALIZATION DASHBOARD
# ============================================================

elif nav_selection == "Model Performance Dashboard":
    st.markdown("### 📊 Model Performance & Evaluation Dashboard")
    st.markdown(
        "Side-by-side comparative evaluation of all three machine learning algorithms evaluated on the "
        "held-out test set (1,008 encounters) using patient-isolated splitting (`StratifiedGroupKFold`) "
        "to prevent data leakage."
    )
    
    # 7. Dataset Overview Cards
    st.markdown("#### 1. Dataset Overview (HLT-005)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Encounters", f"{dataset_summary['total_records']:,}")
    c2.metric("Unique Patients", f"{dataset_summary['unique_patients']:,}")
    c3.metric("30-Day Readmissions", f"{dataset_summary['readmitted_count']:,} ({dataset_summary['readmission_rate_pct']}%)")
    c4.metric("Non-Readmitted", f"{dataset_summary['not_readmitted_count']:,} ({100 - dataset_summary['readmission_rate_pct']:.1f}%)")
    
    st.markdown("---")
    
    # Visualizations organized cleanly into Presentation Tabs
    tab_cmp, tab_cm, tab_curves, tab_feat = st.tabs([
        "📈 Model Comparison & Metrics",
        "🔲 Confusion Matrices",
        "🎯 ROC & Precision-Recall Curves",
        "⭐ Feature Importance"
    ])
    
    # Tab 1: Comparison Table & Grouped Bar Chart
    with tab_cmp:
        st.subheader("Model Comparison Metrics Table")
        st.markdown(
            "Evaluation performed on the held-out test partition with balanced class weighting. "
            "**Recall** is particularly emphasized to capture as many preventable hospital readmissions as possible."
        )
        
        # Display formatted comparison table
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
        st.dataframe(
            evaluation_df[display_cols].style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}"
            }).highlight_max(subset=["Recall", "F1-score", "ROC-AUC"], color="#D4EFDF"),
            width="stretch"
        )
        
        st.subheader("Grouped Performance Bar Chart")
        cmp_barchart_img = RESULTS_DIR / "model_comparison_barchart.png"
        if cmp_barchart_img.exists():
            st.image(str(cmp_barchart_img), caption="Figure 1: Cross-Algorithm Metric Comparison", width="stretch")
        else:
            st.warning("Comparison bar chart image not found in results/ folder.")
            
        st.info(
            "**Key Insight**: **Random Forest** achieved the strongest overall performance with an ROC-AUC of **0.6228**, "
            "Recall of **55.25%**, and F1-score of **0.3503**, outperforming both Logistic Regression and Decision Tree "
            "by capturing complex non-linear interactions across length of stay, comorbidities, and diagnosis."
        )

    # Tab 2: Confusion Matrices
    with tab_cm:
        st.subheader("Confusion Matrix Analysis")
        st.markdown("Select a specific model from the dropdown to inspect its True Positives, False Positives, False Negatives, and True Negatives.")
        
        cm_option = st.selectbox(
            "Select View",
            ["All 3 Models (Side-by-Side)", "Logistic Regression", "Decision Tree", "Random Forest"]
        )
        
        if cm_option == "All 3 Models (Side-by-Side)":
            cm_all_img = RESULTS_DIR / "confusion_matrices.png"
            if cm_all_img.exists():
                st.image(str(cm_all_img), caption="Figure 2: Confusion Matrices for All Three Models", width="stretch")
        else:
            slug = cm_option.lower().replace(" ", "_")
            cm_single_img = RESULTS_DIR / f"confusion_matrix_{slug}.png"
            if cm_single_img.exists():
                col_cm_left, col_cm_right = st.columns([1, 1])
                with col_cm_left:
                    st.image(str(cm_single_img), caption=f"Confusion Matrix — {cm_option}", width="stretch")
                with col_cm_right:
                    cm = confusion_matrices[cm_option]
                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"##### Detailed Breakdown — {cm_option}")
                    st.write(f"• **True Negatives (TN)**: `{tn:,}` (Correctly identified non-readmitted)")
                    st.write(f"• **False Positives (FP)**: `{fp:,}` (Non-readmitted flagged as high risk)")
                    st.write(f"• **False Negatives (FN)**: `{fn:,}` (Missed readmissions — high clinical cost)")
                    st.write(f"• **True Positives (TP)**: `{tp:,}` (Successfully caught readmissions)")
                    st.markdown(f"**Sensitivity / Recall**: `{tp / (tp + fn) * 100:.2f}%`")
                    st.markdown(f"**Specificity**: `{tn / (tn + fp) * 100:.2f}%`")
                    st.markdown(f"**Positive Predictive Value (Precision)**: `{tp / (tp + fp) * 100:.2f}%`")

    # Tab 3: ROC & PR Curves
    with tab_curves:
        st.subheader("Discrimination & Imbalance Curves")
        st.markdown(
            "ROC curves evaluate sensitivity vs 1-specificity across all thresholds, while Precision-Recall curves "
            "specifically measure positive predictive value in the presence of class imbalance (~17.3% positive rate)."
        )
        
        c_roc, c_pr = st.columns(2)
        with c_roc:
            roc_img = RESULTS_DIR / "roc_curve_comparison.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="Figure 3: Receiver Operating Characteristic (ROC) Comparison", width="stretch")
            else:
                st.warning("ROC curve image not found.")
                
        with c_pr:
            pr_img = RESULTS_DIR / "pr_curve_comparison.png"
            if pr_img.exists():
                st.image(str(pr_img), caption="Figure 4: Precision-Recall (PR) Curve Comparison", width="stretch")
            else:
                st.warning("PR curve image not found.")
                
        st.caption(
            "Notice how the PR curve baseline sits at 17.3% (the true positive incidence rate). All three models "
            "substantially outperform the horizontal no-skill baseline, demonstrating genuine discriminative utility."
        )

    # Tab 4: Feature Importance
    with tab_feat:
        st.subheader("Random Forest Feature Importance")
        st.markdown("Relative feature importances derived from Gini impurity reduction across all 150 decision trees:")
        
        rf_imp_img = RESULTS_DIR / "feature_importance_rf.png"
        if rf_imp_img.exists():
            st.image(str(rf_imp_img), caption="Figure 5: Random Forest Gini Feature Importance", width="stretch")
        else:
            st.warning("Feature importance image not found.")
            
        st.markdown("""
        **Clinical Takeaways from Feature Ranking**:
        1. **Length of Hospital Stay (`los_days`)**: Single strongest predictor; prolonged inpatient care reflects complications or severe medical frailty.
        2. **Patient Age (`age`)**: Older demographics have reduced physiological resilience and higher readmission propensity.
        3. **Comorbidity Score (`cci_score`)**: Multi-system chronic disease burden significantly compounds post-discharge risk.
        4. **Prior Inpatient Admissions (`prior_admits_12mo`)**: Prior utilization patterns are reliable markers of chronic disease exacerbation.
        """)

# ============================================================
# SECTION C: DATASET & EXPLORATORY ANALYSIS (EDA)
# ============================================================

elif nav_selection == "Dataset & Exploratory Analysis":
    st.markdown("### 🔬 Exploratory Data Analysis (EDA)")
    st.markdown(
        "Visualizations and distributional analysis performed on the **HLT-005 Synthetic Hospital Admission Dataset** "
        "(5,000 encounters, 76 attributes) prior to model training."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Class Distribution")
        target_img = RESULTS_DIR / "target_class_distribution.png"
        if target_img.exists():
            st.image(str(target_img), caption="Figure 6: 30-Day Readmission Distribution (17.3% Positive Rate)", width="stretch")
    with col2:
        st.subheader("Patient Age Distribution")
        age_img = RESULTS_DIR / "age_distribution.png"
        if age_img.exists():
            st.image(str(age_img), caption="Figure 7: Age Distribution Stratified by Readmission", width="stretch")
            
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Length of Stay Distribution")
        los_img = RESULTS_DIR / "length_of_stay_distribution.png"
        if los_img.exists():
            st.image(str(los_img), caption="Figure 8: Length of Hospital Stay by Readmission Status", width="stretch")
    with col4:
        st.subheader("Clinical Correlation Heatmap")
        corr_img = RESULTS_DIR / "correlation_heatmap.png"
        if corr_img.exists():
            st.image(str(corr_img), caption="Figure 9: Pearson Correlation Matrix of Modeling Features", width="stretch")

    st.subheader("Multi-Panel Predictor Distributions")
    feat_dist_img = RESULTS_DIR / "feature_distributions.png"
    if feat_dist_img.exists():
        st.image(str(feat_dist_img), caption="Figure 10: Distributions of Admissions, ED Visits, CCI, and ICU Flags", width="stretch")

    with st.expander("🔍 View Missing Values Audit"):
        miss_img = RESULTS_DIR / "missing_values.png"
        if miss_img.exists():
            st.image(str(miss_img), caption="Figure 11: Missing Values Across All Columns", width="stretch")
            st.caption("Note: All 7 features selected for machine learning modeling have 0% missing values (100% complete data).")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <b>CarePredict | Hospital Readmission Analytics & Prediction</b><br>
    Machine Learning Engineering Project &nbsp;|&nbsp; HLT-005 Synthetic Dataset &nbsp;|&nbsp; IEEE Two-Column Alignment
</div>
""", unsafe_allow_html=True)
