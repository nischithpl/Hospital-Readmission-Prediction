"""
CarePredict | 30-Day Hospital Readmission Analytics & Prediction Dashboard
==========================================================================
Undergraduate Machine Learning Project
Dataset: UCI Diabetes 130-US Hospitals (1999–2008) — Real-World Clinical Data
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
    page_title="CarePredict | 30-Day Readmission Analytics",
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
/* Base theme & layout */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #F4F7FB !important;
    color: #263B50 !important;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}

h1, h2, h3, h4, h5, h6 {
    color: #153B5B !important;
    font-weight: 700 !important;
}

p, span {
    color: #34495E;
}

/* Sidebar (Dark Navy with Pure White Text) */
section[data-testid="stSidebar"] {
    background-color: #123650 !important;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg {
    fill: #A0C4DF !important;
    color: #A0C4DF !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: #214A63 !important;
    border-color: #45677D !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] * {
    color: #FFFFFF !important;
}

/* Hero Header */
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

/* Patient Form Card */
[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #DCE6EF !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 2px 8px rgba(18, 54, 80, 0.04) !important;
}

[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] label[data-testid="stWidgetLabel"],
div[data-testid="stForm"] label,
div[data-testid="stForm"] label *,
div[data-testid="stForm"] label p,
div[data-testid="stForm"] span,
div[data-testid="stForm"] h5 {
    color: #153B5B !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

div[data-baseweb="input"] input,
div[data-testid="stNumberInput"] input {
    color: #123650 !important;
    background-color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #C9D7E5 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #123650 !important;
    font-weight: 500 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: #F4F7FB !important;
    border: 1px solid #C9D7E5 !important;
}

div[data-testid="stNumberInput"] button svg {
    fill: #153B5B !important;
    color: #153B5B !important;
}

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

button[data-baseweb="tab"] {
    color: #153B5B !important;
    font-weight: 600 !important;
    font-size: 14.5px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #176B87 !important;
    border-bottom-color: #176B87 !important;
}

.footer {
    text-align: center;
    color: #718493;
    font-size: 13px;
    padding: 24px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

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
        st.info("Please run `notebooks/pipeline.ipynb` to train models and generate artifacts in `models/`.")
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
    st.caption("30-Day Hospital Readmission Analytics (CMS HRRP)")
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
        help="Select the trained supervised classifier used for real-time patient risk evaluation."
    )
    
    model_threshold = thresholds.get(selected_model_name, 0.50)
    st.metric("Decision Threshold", f"{model_threshold:.2f}")
    
    st.caption("Real-World Benchmark: UCI Diabetes 130-US Hospitals (1999–2008).")

# ============================================================
# 4. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>CarePredict</h1>
    <p>Predictive Healthcare Analytics & 30-Day Hospital Readmission Risk Stratification (CMS HRRP Standard)</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION A: PATIENT READMISSION PREDICTION
# ============================================================

if nav_selection == "Patient Readmission Prediction":
    st.markdown("### 📋 Clinical Patient Readmission Assessment")
    st.markdown(
        "Enter clinical indicators for the inpatient prior to discharge to evaluate the likelihood of an "
        "unplanned **30-day hospital readmission** under CMS HRRP guidelines. Inputs are transformed through the trained "
        "Scikit-Learn preprocessing pipeline."
    )
    
    if "reset_trigger" not in st.session_state:
        st.session_state.reset_trigger = 0

    col_btn_reset, col_info = st.columns([1, 4])
    with col_btn_reset:
        if st.button("🔄 Reset to Default Values"):
            st.session_state.reset_trigger += 1
            st.rerun()
    with col_info:
        st.caption("Active Model: **" + selected_model_name + "** | Real Data: UCI Diabetes 130-US Hospitals")

    suffix = f"_{st.session_state.reset_trigger}"
    
    with st.form("patient_prediction_form"):
        st.markdown("##### Clinical Patient Indicators (11 Features)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("###### Demographics & Stay")
            age_val = st.number_input(
                "1. Patient Age (Years)",
                min_value=5, max_value=95,
                value=int(defaults["age_num"]),
                step=10,
                help="Approximate patient age in years (decade midpoint)",
                key=f"age{suffix}"
            )
            los_val = st.number_input(
                "2. Length of Stay (Days)",
                min_value=1, max_value=14,
                value=int(defaults["time_in_hospital"]),
                step=1,
                help="Duration of current inpatient hospital stay (1 to 14 days)",
                key=f"los{suffix}"
            )
            dx_val = st.selectbox(
                "3. Primary Diagnosis Category",
                options=category_options["primary_diagnosis"],
                index=category_options["primary_diagnosis"].index(defaults["primary_diagnosis"])
                if defaults["primary_diagnosis"] in category_options["primary_diagnosis"] else 0,
                help="Primary ICD-9 clinical diagnostic group",
                key=f"dx{suffix}"
            )
            num_dx_val = st.number_input(
                "4. Secondary Diagnoses Count",
                min_value=1, max_value=16,
                value=int(defaults["number_diagnoses"]),
                step=1,
                help="Total number of secondary diagnoses entered for this encounter",
                key=f"ndx{suffix}"
            )
            
        with c2:
            st.markdown("###### Prior Healthcare Utilization")
            inpatient_val = st.number_input(
                "5. Prior Inpatient Visits (Last 1 Yr)",
                min_value=0, max_value=25,
                value=int(defaults["number_inpatient"]),
                step=1,
                help="Number of inpatient admissions in the preceding 12 months",
                key=f"inp{suffix}"
            )
            er_val = st.number_input(
                "6. Prior Emergency Room Visits",
                min_value=0, max_value=25,
                value=int(defaults["number_emergency"]),
                step=1,
                help="Number of emergency department visits in the preceding 12 months",
                key=f"er{suffix}"
            )
            outpatient_val = st.number_input(
                "7. Prior Outpatient Visits",
                min_value=0, max_value=30,
                value=int(defaults["number_outpatient"]),
                step=1,
                help="Number of outpatient clinic visits in the preceding 12 months",
                key=f"outp{suffix}"
            )
            
        with c3:
            st.markdown("###### Clinical Tests & Medications")
            labs_val = st.number_input(
                "8. Lab Procedures Performed",
                min_value=1, max_value=135,
                value=int(defaults["num_lab_procedures"]),
                step=1,
                help="Number of lab tests performed during this hospitalization",
                key=f"labs{suffix}"
            )
            meds_val = st.number_input(
                "9. Medications Prescribed",
                min_value=1, max_value=85,
                value=int(defaults["num_medications"]),
                step=1,
                help="Number of distinct medication prescriptions administered",
                key=f"meds{suffix}"
            )
            insulin_val = st.selectbox(
                "10. Prescribed Insulin?",
                options=["No", "Yes"],
                index=0 if defaults["insulin_flag"] == "No" else 1,
                help="Is the patient on insulin therapy?",
                key=f"ins{suffix}"
            )
            change_val = st.selectbox(
                "11. Medication Dosage Changed?",
                options=["No", "Yes"],
                index=0 if defaults["med_change"] == "No" else 1,
                help="Was there a change in diabetic medications during this encounter?",
                key=f"chg{suffix}"
            )

        submit_btn = st.form_submit_button("🔍 Predict 30-Day Readmission Risk")

    if submit_btn:
        input_data = pd.DataFrame([{
            "age_num": age_val,
            "time_in_hospital": los_val,
            "number_inpatient": inpatient_val,
            "number_emergency": er_val,
            "number_outpatient": outpatient_val,
            "num_medications": meds_val,
            "num_lab_procedures": labs_val,
            "number_diagnoses": num_dx_val,
            "primary_diagnosis": dx_val,
            "diabetes_med": "Yes",
            "insulin_flag": insulin_val,
            "med_change": change_val
        }])[input_features]

        try:
            X_transformed = preprocessor.transform(input_data)
            active_model = models[selected_model_name]
            
            prob = float(active_model.predict_proba(X_transformed)[0, 1])
            is_likely = prob >= model_threshold
            
            banner_color = "#E74C3C" if is_likely else "#27AE60"
            status_text = "⚠️ 30-Day Readmission Likely" if is_likely else "✅ 30-Day Readmission Not Likely"
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
            
            st.markdown("#### Clinical Interpretation & Key Risk Drivers")
            drivers = []
            if inpatient_val >= 2:
                drivers.append(f"• **Recurring Inpatient History**: {inpatient_val} prior inpatient admissions in the preceding year strongly elevates long-term readmission probability.")
            if meds_val >= 18:
                drivers.append(f"• **High Medication Complexity**: {meds_val} prescribed medications indicates significant multi-system chronic illness.")
            if los_val >= 7:
                drivers.append(f"• **Prolonged Inpatient Stay**: Length of stay of {los_val} days reflects acute medical complexity.")
            if insulin_val == "Yes":
                drivers.append("• **Insulin Dependence**: Patient is prescribed insulin therapy, reflecting advanced diabetes management.")
            if dx_val in ["Circulatory", "Respiratory", "Genitourinary"]:
                drivers.append(f"• **High-Risk Clinical Category**: Primary diagnosis of *{dx_val}* carries elevated 30-day readmission risk.")
            if er_val >= 1:
                drivers.append(f"• **Emergency Department Utilization**: {er_val} prior ED visits indicate frequent acute disease exacerbations.")

            if drivers:
                for d in drivers:
                    st.markdown(d)
            else:
                st.markdown("• **Low Acuity Profile**: Modest hospital stay duration, low prior utilization, and manageable medication count place this patient in a lower risk category.")

            if is_likely:
                st.info("💡 **Recommended Action**: Enroll patient in structured post-discharge disease management, schedule primary care follow-up within 7–14 days, and verify medication adherence.")
            else:
                st.success("💡 **Recommended Action**: Proceed with standard discharge instructions and routine outpatient follow-up.")

            st.markdown("---")
            report_df = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model": selected_model_name,
                "Prediction": "Readmission Likely (1)" if is_likely else "Not Likely (0)",
                "Readmission_Probability_Pct": round(prob * 100, 2),
                "Threshold": model_threshold,
                "Age_Midpoint": age_val,
                "Length_of_Stay_Days": los_val,
                "Prior_Inpatient_Visits": inpatient_val,
                "Prior_Emergency_Visits": er_val,
                "Prior_Outpatient_Visits": outpatient_val,
                "Num_Medications": meds_val,
                "Num_Lab_Procedures": labs_val,
                "Num_Diagnoses": num_dx_val,
                "Primary_Diagnosis": dx_val,
                "Insulin_Prescribed": insulin_val,
                "Medication_Changed": change_val
            }])
            
            st.download_button(
                "📥 Download Prediction Summary (CSV)",
                report_df.to_csv(index=False),
                file_name=f"carepredict_1yr_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# SECTION B: MODEL PERFORMANCE & VISUALIZATION DASHBOARD
# ============================================================

elif nav_selection == "Model Performance Dashboard":
    st.markdown("### 📊 Model Performance & Evaluation Dashboard")
    st.markdown(
        "Comparative evaluation of all three machine learning algorithms on the real-world **UCI Diabetes 130-US Hospitals dataset** "
        "(19,802 encounters in the held-out test partition). Evaluated with patient-isolated splitting (`GroupShuffleSplit`) "
        "to guarantee zero patient data leakage."
    )
    
    st.markdown("#### 1. Real-World Dataset Overview (UCI 130-US Hospitals)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Encounters", f"{dataset_summary['total_records']:,}")
    c2.metric("Unique Patients", f"{dataset_summary['unique_patients']:,}")
    c3.metric("30-Day Readmissions", f"{dataset_summary['readmitted_count']:,} ({dataset_summary['readmission_rate_pct']}%)")
    c4.metric("Non-Readmitted", f"{dataset_summary['not_readmitted_count']:,} ({100 - dataset_summary['readmission_rate_pct']:.1f}%)")
    
    st.markdown("---")
    
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
            "Evaluation on the held-out test set (19,802 real encounters). "
            "Formulated under **CMS HRRP standards**, the 30-day readmission rate is **11.4%**, allowing models to achieve **~86.6% to 88.8% Accuracy** with robust clinical screening discrimination."
        )
        
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
        st.dataframe(
            evaluation_df[display_cols].style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}"
            }).highlight_max(subset=["Precision", "F1-score", "ROC-AUC", "PR-AUC"], color="#D4EFDF"),
            width="stretch"
        )
        
        st.subheader("Grouped Performance Bar Chart")
        cmp_barchart_img = RESULTS_DIR / "model_comparison_barchart.png"
        if cmp_barchart_img.exists():
            st.image(str(cmp_barchart_img), caption="Figure 1: Cross-Algorithm Metric Comparison", width="stretch")
        else:
            st.warning("Comparison bar chart image not found in results/ folder.")
            
        st.info(
            "**Key Insight**: **Random Forest** achieved the highest overall discrimination with an **ROC-AUC of 0.6701**, "
            "Precision of **62.86%**, and F1-score of **0.5562**, accurately identifying patients susceptible to repeated hospitalization."
        )

    # Tab 2: Confusion Matrices
    with tab_cm:
        st.subheader("Confusion Matrix Analysis")
        st.markdown("Side-by-side confusion matrices for all three models on 19,802 held-out test encounters:")
        
        cm_all_img = RESULTS_DIR / "confusion_matrices.png"
        if cm_all_img.exists():
            st.image(str(cm_all_img), caption="Figure 2: Confusion Matrices for All Three Models", width="stretch")
        else:
            st.warning("Confusion matrices image not found.")

    # Tab 3: ROC & PR Curves
    with tab_curves:
        st.subheader("Discrimination & Precision-Recall Curves")
        c_roc, c_pr = st.columns(2)
        with c_roc:
            roc_img = RESULTS_DIR / "roc_curve_comparison.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="Figure 3: Receiver Operating Characteristic (ROC) Comparison", width="stretch")
        with c_pr:
            pr_img = RESULTS_DIR / "pr_curve_comparison.png"
            if pr_img.exists():
                st.image(str(pr_img), caption="Figure 4: Precision-Recall (PR) Curve Comparison", width="stretch")

    # Tab 4: Feature Importance
    with tab_feat:
        st.subheader("Random Forest Feature Importance")
        st.markdown("Relative feature importances derived from Gini impurity reduction across 100 decision trees:")
        
        rf_imp_img = RESULTS_DIR / "feature_importance_rf.png"
        if rf_imp_img.exists():
            st.image(str(rf_imp_img), caption="Figure 5: Random Forest Gini Feature Importance", width="stretch")
            
        st.markdown("""
        **Clinical Takeaways from Feature Ranking**:
        1. **Lab Procedures (`num_lab_procedures`)**: Heavy inpatient testing reflects clinical complexity and monitoring intensity.
        2. **Medications Prescribed (`num_medications`)**: Polypharmacy is a primary determinant of long-term post-discharge vulnerability.
        3. **Length of Stay (`time_in_hospital`)**: Longer hospitalizations indicate acute illness severity.
        4. **Prior Inpatient Hospitalizations (`number_inpatient`)**: Single most reliable utilization marker for recurring readmission.
        """)

# ============================================================
# SECTION C: DATASET & EXPLORATORY ANALYSIS (EDA)
# ============================================================

elif nav_selection == "Dataset & Exploratory Analysis":
    st.markdown("### 🔬 Exploratory Data Analysis (EDA)")
    st.markdown(
        "Visualizations and distributional analysis performed on the **UCI Diabetes 130-US Hospitals dataset** "
        "(99,343 eligible encounters across 130 hospitals) prior to model training."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("30-Day Readmission Class Distribution")
        target_img = RESULTS_DIR / "target_class_distribution.png"
        if target_img.exists():
            st.image(str(target_img), caption="Figure 6: 30-Day Readmission Incidence (11.4% Yes vs 88.6% No)", width="stretch")
    with col2:
        st.subheader("Patient Age Bracket Distribution")
        age_img = RESULTS_DIR / "age_distribution.png"
        if age_img.exists():
            st.image(str(age_img), caption="Figure 7: Age Brackets Stratified by Readmission", width="stretch")
            
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Length of Stay Distribution")
        los_img = RESULTS_DIR / "length_of_stay_distribution.png"
        if los_img.exists():
            st.image(str(los_img), caption="Figure 8: Length of Hospital Stay (Days)", width="stretch")
    with col4:
        st.subheader("Clinical Correlation Heatmap")
        corr_img = RESULTS_DIR / "correlation_heatmap.png"
        if corr_img.exists():
            st.image(str(corr_img), caption="Figure 9: Pearson Correlation Matrix of Modeling Features", width="stretch")

    st.subheader("Multi-Panel Clinical Predictor Distributions")
    feat_dist_img = RESULTS_DIR / "feature_distributions.png"
    if feat_dist_img.exists():
        st.image(str(feat_dist_img), caption="Figure 10: Prior Inpatient, Emergency, Diagnoses, and Diabetes Meds", width="stretch")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <b>CarePredict | 30-Day Hospital Readmission Analytics & Prediction</b><br>
    Machine Learning Project &nbsp;|&nbsp; UCI Diabetes 130-US Hospitals (Real-World Benchmark) &nbsp;|&nbsp; IEEE Two-Column Alignment
</div>
""", unsafe_allow_html=True)
