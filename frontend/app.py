
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


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CarePredict | Hospital Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
IMPROVED_MODEL_DIR = ROOT / "models" / "improved"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "datasets" / "dataset1"

st.markdown("""
<style>

.stApp {
    background: #F4F7FB;
    color: #263B50;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #153B5B !important;
}

p, label, span {
    color: #34495E;
}

section[data-testid="stSidebar"] {
    background: #123650;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(110deg, #123650, #176B87, #23988D);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 22px;
}

.hero h1 {
    color: white !important;
    font-size: 32px;
}

.hero p {
    color: #E5F3F6 !important;
}

.card {
    background: white;
    border: 1px solid #DFE7EF;
    padding: 18px;
    border-radius: 13px;
    margin-bottom: 12px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DFE7EF;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #153B5B !important;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: #214A63;
    border-color: #45677D;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] * {
    color: white !important;
}

button[data-baseweb="tab"] {
    color: #153B5B !important;
}

.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    background: #176B87;
    color: white !important;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #12546B;
    color: white !important;
}

[data-testid="stWidgetLabel"] * {
    color: #263B50 !important;
}

/* Light, readable form controls and chart containers */
[data-testid="stAppViewContainer"] { background: #F4F7FB; }
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #FFFFFF !important;
    border: 1px solid #C9D7E5 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] *,
div[data-baseweb="input"] input,
[data-testid="stNumberInput"] input {
    color: #20384D !important;
}
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DCE6EF;
    border-radius: 14px;
    padding: 22px;
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DCE6EF;
    border-radius: 10px;
}
[data-testid="stDataFrame"] { border: 1px solid #DCE6EF; border-radius: 10px; }
[data-testid="stAlert"] { border-radius: 10px; }

.footer {
    text-align: center;
    color: #718493;
    font-size: 12px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOAD MODELS AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    artifacts = {}

    artifacts["models"] = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression_model.pkl"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest_model.pkl"
        ),
        "XGBoost": joblib.load(
            MODEL_DIR / "xgboost_model.pkl"
        )
    }

    artifacts["preprocessor"] = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    artifacts["thresholds"] = joblib.load(
        MODEL_DIR / "model_thresholds.pkl"
    )

    artifacts["input_features"] = joblib.load(
        MODEL_DIR / "input_features.pkl"
    )

    artifacts["metadata"] = joblib.load(
        MODEL_DIR / "feature_metadata.pkl"
    )

    artifacts["evaluation"] = joblib.load(
        MODEL_DIR / "model_evaluation.pkl"
    )

    artifacts["confusion_matrices"] = joblib.load(
        MODEL_DIR / "confusion_matrices.pkl"
    )

    # Optional: generated from the untouched test set in the training notebook.
    roc_path = MODEL_DIR / "roc_curves.pkl"
    artifacts["roc_curves"] = joblib.load(roc_path) if roc_path.exists() else None

    return artifacts


try:
    ART = load_artifacts()

except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info(
        "Check that all three model files, the preprocessing "
        "file and evaluation files exist inside the models folder."
    )
    st.stop()


models = ART["models"]
preprocessor = ART["preprocessor"]
thresholds = ART["thresholds"]
input_features = ART["input_features"]
metadata = ART["metadata"]
evaluation_df = ART["evaluation"]
confusion_matrices = ART["confusion_matrices"]
roc_curves = ART.get("roc_curves")

num_cols = metadata["numerical_columns"]
cat_cols = metadata["categorical_columns"]
defaults = metadata["defaults"]
category_options = metadata["categorical_options"]
# ============================================================
# 2b. LOAD IMPROVED PIPELINE ARTIFACTS
# ============================================================

@st.cache_resource
def load_improved_artifacts():
    if not IMPROVED_MODEL_DIR.exists():
        return None
    try:
        imp = {}
        imp["models"] = {
            "Logistic Regression": joblib.load(IMPROVED_MODEL_DIR / "logistic_regression_model.pkl"),
            "Decision Tree":       joblib.load(IMPROVED_MODEL_DIR / "decision_tree_model.pkl"),
            "Random Forest":       joblib.load(IMPROVED_MODEL_DIR / "random_forest_model.pkl"),
            "XGBoost":             joblib.load(IMPROVED_MODEL_DIR / "xgboost_model.pkl"),
        }
        imp["preprocessor"]    = joblib.load(IMPROVED_MODEL_DIR / "preprocessor.pkl")
        imp["thresholds"]      = joblib.load(IMPROVED_MODEL_DIR / "model_thresholds.pkl")
        imp["input_features"]  = joblib.load(IMPROVED_MODEL_DIR / "input_features.pkl")
        imp["metadata"]        = joblib.load(IMPROVED_MODEL_DIR / "feature_metadata.pkl")
        imp["evaluation"]      = joblib.load(IMPROVED_MODEL_DIR / "model_evaluation.pkl")
        imp["confusion_matrices"] = joblib.load(IMPROVED_MODEL_DIR / "confusion_matrices.pkl")
        return imp
    except Exception:
        return None

IMP = load_improved_artifacts()




# ============================================================
# 3. LOAD ORIGINAL DATASET
# ============================================================

@st.cache_data
def load_dataset():

    candidates = [
        DATA_DIR / "diabetic_data.csv",
        ROOT / "diabetic_data.csv"
    ]

    dataset_path = next(
        (p for p in candidates if p.exists()),
        None
    )

    if dataset_path is None:
        return None

    df = pd.read_csv(
        dataset_path,
        na_values=["?"]
    )

    df["readmitted"] = df["readmitted"].astype(str)

    df["readmission_30"] = (
        df["readmitted"] == "<30"
    ).astype(int)

    return df


df = load_dataset()


# ============================================================
# 4. COMMON FUNCTIONS
# ============================================================

def section(title, subtitle=None):

    st.markdown(f"## {title}")

    if subtitle:
        st.caption(subtitle)


def chart_theme(fig):
    """Apply a consistent light chart theme with high-contrast black labels."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            color="#000000",
            size=14
        ),
        title_font=dict(color="#000000", size=18),
        legend=dict(
            title=dict(font=dict(color="#000000", size=13)),
            font=dict(color="#000000", size=12),
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=35, r=25, t=55, b=45),
        legend_title_text=""
    )

    # Axis tick labels and axis titles must remain black on white backgrounds.
    fig.update_xaxes(
        tickfont=dict(color="#000000", size=12),
        title_font=dict(color="#000000", size=13),
        showline=True,
        linecolor="#333333",
        gridcolor="#E5E7EB",
        zerolinecolor="#9CA3AF"
    )
    fig.update_yaxes(
        tickfont=dict(color="#000000", size=12),
        title_font=dict(color="#000000", size=13),
        showline=True,
        linecolor="#333333",
        gridcolor="#E5E7EB",
        zerolinecolor="#9CA3AF"
    )

    # Text printed on bars, heatmaps, annotations and color scales.
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(color="#000000", size=12)
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color="#000000"),
            title_font=dict(color="#000000")
        ),
        uniformtext=dict(minsize=10, mode="hide")
    )
    for annotation in fig.layout.annotations or []:
        annotation.font = dict(color="#000000", size=12)

    return fig


def show_missing_dataset():

    st.warning(
        "The original diabetic_data.csv file was not found. "
        "Place it inside datasets/dataset1/ to enable "
        "hospital analytics."
    )


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏥 CarePredict")

    st.caption("Hospital Readmission Analytics")

    st.markdown("---")

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select section",
        [
            "Hospital Overview",
            "Patient Analytics",
            "Admission Analytics",
            "Diagnosis Analytics",
            "Correlation Analysis",
            "Model Comparison",
            "Model Evaluation",
            "Feature Importance",
            "Patient Assessment",
            "Prediction Results",
            "── Improved Pipeline ──",
            "Improved Model Comparison",
            "Improved Feature Importance",
            "Improved Patient Assessment",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### MODEL INFORMATION")

    selected_model = st.selectbox(
        "Select prediction model",
        list(models.keys()),
        index=2,
        key="selected_model"
    )

    st.metric(
        "Selected threshold",
        f"{thresholds[selected_model]:.4f}"
    )

    st.caption(
        "Academic prototype — not clinically validated."
    )


# ============================================================
# 6. HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>CarePredict</h1>
    <p>
        Hospital Readmission Prediction and Analytics Platform
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 7. HOSPITAL OVERVIEW
# ============================================================

if page == "Hospital Overview":

    section(
        "Hospital Overview",
        "A summary of hospital encounters and 30-day readmission patterns."
    )

    if df is None:

        show_missing_dataset()

    else:

        total = len(df)

        readmitted_30 = int(
            df["readmission_30"].sum()
        )

        rate = (
            readmitted_30 / total * 100
        )

        avg_stay = df["time_in_hospital"].mean()

        avg_meds = df["num_medications"].mean()

        unique_patients = df["patient_nbr"].nunique()

        a, b, c, d, e = st.columns(5)

        a.metric("Total encounters", f"{total:,}")

        b.metric("30-day readmissions", f"{readmitted_30:,}")

        c.metric("30-day readmission rate", f"{rate:.2f}%")

        d.metric("Average stay", f"{avg_stay:.2f} days")

        e.metric("Unique patients", f"{unique_patients:,}")


        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            section("Readmission Distribution")

            counts = (
                df["readmitted"]
                .value_counts()
                .reindex(["NO", ">30", "<30"])
                .fillna(0)
                .reset_index()
            )

            counts.columns = ["Category", "Count"]

            fig = px.pie(
                counts,
                names="Category",
                values="Count",
                hole=0.5,
                color="Category",
                color_discrete_map={
                    "NO": "#23988D",
                    ">30": "#F2B84B",
                    "<30": "#176B87"
                }
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )

        with col2:

            section("Readmission Class Distribution")

            class_counts = (
                df["readmitted"]
                .value_counts()
                .reindex(["NO", ">30", "<30"])
                .fillna(0)
                .reset_index()
            )

            class_counts.columns = [
                "Readmission category",
                "Encounters"
            ]

            fig = px.bar(
                class_counts,
                x="Readmission category",
                y="Encounters",
                color="Readmission category",
                text_auto=True
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )


        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            section("Hospital Stay Distribution")

            fig = px.histogram(
                df,
                x="time_in_hospital",
                color="readmitted",
                barmode="overlay",
                opacity=0.75,
                labels={
                    "time_in_hospital": "Hospital stay (days)",
                    "readmitted": "Readmission"
                }
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )

        with col2:

            section("Average Medication Count")

            med = (
                df.groupby("readmitted")["num_medications"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                med,
                x="readmitted",
                y="num_medications",
                color="readmitted",
                text_auto=".2f",
                labels={
                    "readmitted": "Readmission",
                    "num_medications": "Average medications"
                }
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )


# ============================================================
# 8. PATIENT ANALYTICS
# ============================================================

elif page == "Patient Analytics":

    section(
        "Patient Analytics",
        "Explore demographic patterns across hospital encounters."
    )

    if df is None:

        show_missing_dataset()

    else:

        col1, col2 = st.columns(2)

        with col1:

            section("Age Group Distribution")

            age_data = (
                df["age"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            age_data.columns = ["Age group", "Encounters"]

            fig = px.bar(
                age_data,
                x="Age group",
                y="Encounters",
                color="Encounters",
                text_auto=True
            )

            fig.update_xaxes(tickangle=0)

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )

        with col2:

            section("Gender Distribution")

            gender_data = (
                df["gender"]
                .value_counts()
                .reset_index()
            )

            gender_data.columns = ["Gender", "Encounters"]

            fig = px.pie(
                gender_data,
                names="Gender",
                values="Encounters",
                hole=0.45
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )


        col1, col2 = st.columns(2)

        with col1:

            section("Readmission by Gender")

            gender_readmission = (
                df.groupby(["gender", "readmitted"])
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                gender_readmission,
                x="gender",
                y="Count",
                color="readmitted",
                barmode="group"
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )

        with col2:

            section("Readmission by Age")

            age_readmission = (
                df.groupby(["age", "readmitted"])
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                age_readmission,
                x="age",
                y="Count",
                color="readmitted",
                barmode="stack"
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )


        section("Readmission Rate by Race")

        race_data = (
            df.groupby("race")["readmission_30"]
            .mean()
            .mul(100)
            .reset_index()
        )

        race_data.columns = ["Race", "Readmission rate (%)"]

        fig = px.bar(
            race_data,
            x="Race",
            y="Readmission rate (%)",
            color="Readmission rate (%)",
            text_auto=".2f"
        )

        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )


# ============================================================
# 9. ADMISSION ANALYTICS
# ============================================================

elif page == "Admission Analytics":

    section(
        "Admission Analytics",
        "Understand admission types, hospital stays and discharge patterns."
    )

    if df is None:

        show_missing_dataset()

    else:

        col1, col2 = st.columns(2)

        with col1:

            section("Admission Type")

            admission = (
                df["admission_type_id"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            admission.columns = ["Admission type", "Count"]

            fig = px.bar(
                admission,
                x="Admission type",
                y="Count",
                text_auto=True
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )

        with col2:

            section("Discharge Disposition")

            discharge = (
                df["discharge_disposition_id"]
                .value_counts()
                .head(12)
                .reset_index()
            )

            discharge.columns = ["Disposition ID", "Count"]

            fig = px.bar(
                discharge,
                x="Count",
                y="Disposition ID",
                orientation="h",
                text_auto=True
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )


        section("Hospital Stay vs Readmission")

        fig = px.box(
            df,
            x="readmitted",
            y="time_in_hospital",
            color="readmitted",
            points=False,
            labels={
                "readmitted": "Readmission",
                "time_in_hospital": "Hospital stay (days)"
            }
        )

        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )


        section("Previous Inpatient Visits vs Readmission")

        fig = px.box(
            df,
            x="readmitted",
            y="number_inpatient",
            color="readmitted",
            points=False
        )

        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )


        section("Hospital Stay vs Number of Procedures")

        sample = df.sample(
            min(5000, len(df)),
            random_state=42
        )

        fig = px.scatter(
            sample,
            x="time_in_hospital",
            y="num_procedures",
            color="readmitted",
            opacity=0.55,
            hover_data=["num_medications"]
        )

        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )


# ============================================================
# 10. DIAGNOSIS ANALYTICS
# ============================================================

elif page == "Diagnosis Analytics":

    section(
        "Diagnosis Analytics",
        "Explore diagnosis codes, medical specialties and treatment patterns."
    )

    if df is None:

        show_missing_dataset()

    else:

        col1, col2 = st.columns(2)

        with col1:

            section("Top Primary Diagnoses")

            diagnosis = (
                df["diag_1"]
                .value_counts()
                .head(12)
                .reset_index()
            )

            diagnosis.columns = ["Diagnosis", "Count"]

            fig = px.bar(
                diagnosis,
                x="Count",
                y="Diagnosis",
                orientation="h",
                text_auto=True
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )

        with col2:

            section("Medical Specialty Distribution")

            specialty = (
                df["medical_specialty"]
                .fillna("Unknown")
                .value_counts()
                .head(12)
                .reset_index()
            )

            specialty.columns = ["Specialty", "Count"]

            fig = px.bar(
                specialty,
                x="Count",
                y="Specialty",
                orientation="h"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )


        section("Readmission by Medical Specialty")

        top_specialties = (
            df["medical_specialty"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .index
        )

        specialty_data = df[
            df["medical_specialty"]
            .fillna("Unknown")
            .isin(top_specialties)
        ].copy()

        specialty_data["medical_specialty"] = (
            specialty_data["medical_specialty"].fillna("Unknown")
        )

        grouped = (
            specialty_data.groupby(
                ["medical_specialty", "readmitted"]
            )
            .size()
            .reset_index(name="Count")
        )

        fig = px.bar(
            grouped,
            x="medical_specialty",
            y="Count",
            color="readmitted",
            barmode="stack"
        )

        fig.update_xaxes(tickangle=-35)

        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )


        section("Medication Usage")

        medication_cols = [
            "metformin",
            "repaglinide",
            "glimepiride",
            "glipizide",
            "glyburide",
            "pioglitazone",
            "rosiglitazone",
            "insulin"
        ]

        medication_cols = [
            c for c in medication_cols if c in df.columns
        ]

        med_data = []

        for col in medication_cols:

            counts = df[col].value_counts()

            med_data.append({
                "Medication": col,
                "Steady": counts.get("Steady", 0),
                "Up": counts.get("Up", 0),
                "Down": counts.get("Down", 0),
                "No": counts.get("No", 0)
            })

        med_df = pd.DataFrame(med_data)

        if not med_df.empty:

            fig = px.bar(
                med_df,
                x="Medication",
                y=["Steady", "Up", "Down", "No"],
                barmode="stack"
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

elif page == "Correlation Analysis":

    section(
        "Correlation Analysis",
        "Examine relationships between numerical encounter attributes."
    )

    if df is None:

        show_missing_dataset()

    else:

        numerical = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "readmission_30"
        ]

        numerical = [
            c for c in numerical if c in df.columns
        ]

        corr = df[numerical].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )

        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )


        st.markdown("---")

        section("Numerical Feature Distributions")

        feature = st.selectbox(
            "Select numerical feature",
            numerical[:-1]
        )

        fig = px.histogram(
            df,
            x=feature,
            color="readmitted",
            marginal="box",
            opacity=0.7
        )

        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    section(
        "Model Comparison",
        "Performance comparison using the same untouched test dataset."
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    comparison = evaluation_df.copy()

    st.dataframe(
        comparison.style.format({
            "Threshold": "{:.4f}",
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        width="stretch",
        hide_index=True
    )


    section("Performance Metrics Comparison")

    metric = st.selectbox(
        "Choose evaluation metric",
        metrics,
        index=3
    )

    fig = px.bar(
        comparison,
        x="Model",
        y=metric,
        color="Model",
        text=metric
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(
        chart_theme(fig),
        width="stretch"
    )


    section("Compare All Metrics")

    melted = comparison.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".3f"
    )

    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(
        chart_theme(fig),
        width="stretch"
    )


    best_f1 = comparison.loc[
        comparison["F1 Score"].idxmax()
    ]

    best_recall = comparison.loc[
        comparison["Recall"].idxmax()
    ]

    best_auc = comparison.loc[
        comparison["ROC-AUC"].idxmax()
    ]

    a, b, c = st.columns(3)

    a.metric(
        "Best F1-score",
        best_f1["Model"],
        f"{best_f1['F1 Score']:.4f}"
    )

    b.metric(
        "Best Recall",
        best_recall["Model"],
        f"{best_recall['Recall']:.4f}"
    )

    c.metric(
        "Best ROC-AUC",
        best_auc["Model"],
        f"{best_auc['ROC-AUC']:.4f}"
    )

    st.markdown("---")
    section("ROC Curve Comparison", "ROC curves are calculated from prediction scores on the same untouched test set.")
    if roc_curves:
        roc_fig = go.Figure()
        for model_name in models:
            curve = roc_curves.get(model_name)
            if curve and "fpr" in curve and "tpr" in curve:
                auc_value = float(curve.get("auc", comparison.loc[comparison["Model"] == model_name, "ROC-AUC"].iloc[0]))
                roc_fig.add_trace(go.Scatter(
                    x=curve["fpr"], y=curve["tpr"], mode="lines",
                    name=f"{model_name} (AUC = {auc_value:.3f})"
                ))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Random classifier", line=dict(dash="dash", color="#94A3B8")))
        roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(chart_theme(roc_fig), width="stretch")
    else:
        st.info("ROC curve data has not been saved yet. Add the short notebook cell provided with this updated app to create models/roc_curves.pkl. The existing ROC-AUC scores above are unchanged.")



# ============================================================
# 13. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    section(
        "Model Evaluation",
        "Inspect confusion matrices for the three trained classifiers."
    )

    chosen = st.selectbox(
        "Choose model",
        list(models.keys()),
        key="evaluation_model"
    )

    cm = confusion_matrices[chosen]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        color_continuous_scale="Blues",
        aspect="auto"
    )

    fig.update_layout(
        title=f"{chosen} — Confusion Matrix"
    )

    st.plotly_chart(
        chart_theme(fig),
        width="stretch"
    )

    tn, fp, fn, tp = cm.ravel()

    a, b, c, d = st.columns(4)

    a.metric("True Negatives", f"{tn:,}")
    b.metric("False Positives", f"{fp:,}")
    c.metric("False Negatives", f"{fn:,}")
    d.metric("True Positives", f"{tp:,}")

    row = evaluation_df[
        evaluation_df["Model"] == chosen
    ].iloc[0]

    st.markdown("---")

    st.markdown("### Classification Metrics")

    a, b, c = st.columns(3)

    a.metric("Accuracy", f"{row['Accuracy']:.2%}")
    b.metric("Precision", f"{row['Precision']:.2%}")
    c.metric("Recall", f"{row['Recall']:.2%}")

    a, b = st.columns(2)

    a.metric("F1-score", f"{row['F1 Score']:.4f}")
    b.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")

    st.info(
        "The confusion matrix uses the validation-selected "
        "decision threshold for the selected model."
    )

    # ROC Curve Comparison
    st.markdown("---")

    st.subheader("ROC Curve Comparison")

    roc_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "roc_curve_comparison.png"
    )


    if os.path.exists(roc_path):
        st.image(
            roc_path,
            caption="ROC Curve Comparison of Three ML Models",
            width="stretch"
        )
    else:
        st.warning("ROC curve image not found.")


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    section(
        "Feature Importance",
        "Explore the features contributing to the trained XGBoost model."
    )

    xgb = models["XGBoost"]

    if not hasattr(xgb, "feature_importances_"):

        st.warning(
            "Feature importance is unavailable for this model."
        )

    else:

        try:

            feature_names = preprocessor.get_feature_names_out()

            importance = xgb.feature_importances_

            feature_df = pd.DataFrame({

                "Feature": feature_names,

                "Importance": importance

            })

            feature_df = feature_df.sort_values(
                "Importance",
                ascending=False
            ).head(20)

            fig = px.bar(
                feature_df.sort_values(
                    "Importance",
                    ascending=True
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance"
            )

            fig.update_layout(
                height=650
            )

            st.plotly_chart(
                chart_theme(fig),
                width="stretch"
            )

            st.markdown("### Top 20 Important Features")

            st.dataframe(
                feature_df,
                width="stretch",
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Unable to display feature importance: {e}"
            )


# ============================================================
# 15. PATIENT ASSESSMENT
# ============================================================

elif page == "Patient Assessment":

    section(
        "Patient Assessment",
        "Enter encounter details to generate a prediction."
    )

    st.info(
        f"Selected model: {selected_model} | "
        f"Decision threshold: {thresholds[selected_model]:.4f}"
    )
    st.caption("Enter the encounter information available at the time of discharge. Fields not shown in the form use the training-set defaults saved in feature_metadata.pkl.")

    values = {}

    with st.form("patient_assessment_form"):

        st.markdown("### 01 · Patient Demographics")

        demographic = [
            c for c in ["race", "gender", "age"]
            if c in cat_cols
        ]

        cols = st.columns(2)

        for i, name in enumerate(demographic):

            options = [
                str(x) for x in category_options[name]
            ]

            default = str(
                defaults.get(name, options[0])
            )

            index = (
                options.index(default)
                if default in options
                else 0
            )

            with cols[i % 2]:

                values[name] = st.selectbox(
                    name.replace("_", " ").title(),
                    options,
                    index=index
                )


        st.markdown("---")

        st.markdown("### 02 · Admission Details")

        admission_fields = [
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "payer_code",
            "medical_specialty"
        ]

        admission_fields = [
            c for c in admission_fields if c in cat_cols
        ]

        cols = st.columns(2)

        for i, name in enumerate(admission_fields):

            options = [
                str(x) for x in category_options[name]
            ]

            default = str(
                defaults.get(name, options[0])
            )

            index = (
                options.index(default)
                if default in options
                else 0
            )

            with cols[i % 2]:

                values[name] = st.selectbox(
                    name.replace("_", " ").title(),
                    options,
                    index=index
                )


        st.markdown("---")

        st.markdown("### 03 · Hospital Stay and Previous Visits")

        numeric_labels = {

            "time_in_hospital": "Hospital stay (days)",

            "num_lab_procedures": "Number of lab procedures",

            "num_procedures": "Number of procedures",

            "num_medications": "Number of medications",

            "number_outpatient": "Previous outpatient visits",

            "number_emergency": "Previous emergency visits",

            "number_inpatient": "Previous inpatient visits",

            "number_diagnoses": "Number of diagnoses"

        }

        cols = st.columns(2)

        for i, name in enumerate(num_cols):

            default = max(
                0,
                int(round(float(defaults.get(name, 0))))
            )

            with cols[i % 2]:

                values[name] = st.number_input(
                    numeric_labels.get(
                        name,
                        name.replace("_", " ").title()
                    ),
                    min_value=0,
                    max_value=500,
                    value=default
                )


        st.markdown("---")

        st.markdown("### 04 · Diagnosis and Laboratory")

        diagnosis_fields = [
            "diag_1",
            "diag_2",
            "diag_3",
            "max_glu_serum",
            "A1Cresult"
        ]

        diagnosis_fields = [
            c for c in diagnosis_fields if c in cat_cols
        ]

        cols = st.columns(2)

        for i, name in enumerate(diagnosis_fields):

            options = [
                str(x) for x in category_options[name]
            ]

            default = str(
                defaults.get(name, options[0])
            )

            index = (
                options.index(default)
                if default in options
                else 0
            )

            with cols[i % 2]:

                values[name] = st.selectbox(
                    name.replace("_", " ").title(),
                    options,
                    index=index
                )


        st.markdown("---")

        medication_fields = [
            c for c in cat_cols
            if c not in demographic
            and c not in admission_fields
            and c not in diagnosis_fields
        ]

        with st.expander(
            "Medication Details",
            expanded=False
        ):

            cols = st.columns(2)

            for i, name in enumerate(medication_fields):

                options = [
                    str(x) for x in category_options[name]
                ]

                default = str(
                    defaults.get(name, options[0])
                )

                index = (
                    options.index(default)
                    if default in options
                    else 0
                )

                with cols[i % 2]:

                    values[name] = st.selectbox(
                        name.replace("_", " ").title(),
                        options,
                        index=index
                    )


        submit = st.form_submit_button(
            "Generate Prediction",
            width="stretch"
        )


    if submit:

        try:

            row = {}

            for name in input_features:

                if name in values:

                    row[name] = values[name]

                elif name in defaults:

                    row[name] = defaults[name]

                else:

                    raise ValueError(
                        f"Missing feature: {name}. "
                        "Check feature_metadata.pkl."
                    )


            patient_df = pd.DataFrame(
                [row],
                columns=input_features
            )


            for name in cat_cols:

                patient_df[name] = (
                    patient_df[name].astype(str)
                )


            for name in num_cols:

                patient_df[name] = pd.to_numeric(
                    patient_df[name],
                    errors="raise"
                )


            transformed = preprocessor.transform(
                patient_df
            )


            model = models[selected_model]

            score = float(
                model.predict_proba(transformed)[0, 1]
            )

            threshold = float(
                thresholds[selected_model]
            )

            prediction = int(
                score >= threshold
            )


            st.session_state["prediction_result"] = {

                "model": selected_model,

                "score": score,

                "threshold": threshold,

                "prediction": prediction,

                "timestamp": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )

            }


            st.success(
                "Prediction generated successfully. "
                "Open Prediction Results."
            )


        except Exception as e:

            st.error(f"Prediction error: {e}")


# ============================================================
# 16. PREDICTION RESULTS
# ============================================================

elif page == "Prediction Results":

    section(
        "Prediction Results",
        "Review the latest patient-level prediction."
    )

    result = st.session_state.get(
        "prediction_result"
    )

    if result is None:

        st.info(
            "No prediction available. "
            "Go to Patient Assessment and generate a prediction."
        )

    else:

        score = result["score"]

        threshold = result["threshold"]

        prediction = result["prediction"]

        model_name = result["model"]


        if prediction == 1:

            st.error(
                "Model classification: Readmission within 30 days"
            )

        else:

            st.success(
                "Model classification: Not classified as "
                "readmitted within 30 days"
            )


        a, b, c = st.columns(3)

        a.metric(
            "Selected model",
            model_name
        )

        b.metric(
            "Model score",
            f"{score:.2%}"
        )

        c.metric(
            "Decision threshold",
            f"{threshold:.2%}"
        )

        if prediction == 1:
            st.markdown("#### Model flag: Higher predicted readmission risk")
            st.write("The model score is at or above the selected decision threshold. This is a model classification, not a diagnosis or a clinical decision.")
        else:
            st.markdown("#### Model flag: Below the readmission threshold")
            st.write("The model score is below the selected decision threshold. This does not guarantee that readmission will not occur.")

        section("Model Score vs Decision Threshold")

        chart_data = pd.DataFrame({

            "Metric": [
                "Model score",
                "Decision threshold"
            ],

            "Percentage": [
                score * 100,
                threshold * 100
            ]

        })

        fig = px.bar(
            chart_data,
            x="Percentage",
            y="Metric",
            orientation="h",
            text_auto=".2f",
            color="Metric",
            range_x=[0, 100]
        )

        fig.update_layout(showlegend=False, xaxis_title="Score (%)", yaxis_title="")
        st.plotly_chart(
            chart_theme(fig),
            width="stretch"
        )

        st.caption(
            f"Prediction generated: {result['timestamp']}"
        )


        report = pd.DataFrame([{

            "Model": model_name,

            "Model score": score,

            "Threshold": threshold,

            "Prediction": prediction,

            "Generated at": result["timestamp"]

        }])


        st.download_button(
            "Download Prediction Report",
            report.to_csv(index=False),
            file_name="carepredict_report.csv",
            mime="text/csv"
        )


        st.warning(
            "This is an educational machine-learning prototype. "
            "The score is not necessarily a calibrated clinical "
            "probability and must not be used for patient-care decisions."
        )



# ============================================================
# 18. IMPROVED MODEL COMPARISON
# ============================================================

elif page == "Improved Model Comparison":

    section(
        "Improved Model Comparison",
        "Side-by-side performance of all 4 models from the improved pipeline (5-fold StratifiedGroupKFold CV)."
    )

    if IMP is None:
        st.warning("Improved model artifacts not found. Run notebooks/train_improved_pipeline.py first.")
    else:
        imp_eval = IMP["evaluation"]

        # Metrics table
        st.subheader("Performance Metrics (held-out test set)")
        display_cols = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        available = [c for c in display_cols if c in imp_eval.columns]
        st.dataframe(imp_eval[available].set_index("Model").style.format("{:.4f}"), width="stretch")

        # Bar chart comparison
        st.subheader("ROC-AUC & PR-AUC Comparison")
        if "ROC-AUC" in imp_eval.columns and "PR-AUC" in imp_eval.columns:
            melted = imp_eval[["Model", "ROC-AUC", "PR-AUC"]].melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig = px.bar(
                melted, x="Model", y="Score", color="Metric", barmode="group",
                color_discrete_sequence=["#0D9488", "#1E3A8A"],
                title="ROC-AUC vs PR-AUC per Model"
            )
            fig.update_layout(height=400, yaxis_range=[0, 1])
            st.plotly_chart(fig, width="stretch")

        # ROC & PR curve images from results/
        col1, col2 = st.columns(2)
        roc_img = RESULTS_DIR / "roc_curve_comparison_improved.png"
        pr_img  = RESULTS_DIR / "pr_curve_comparison_improved.png"
        with col1:
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curves – Improved Pipeline", width="stretch")
            else:
                st.info("ROC curve image not generated yet.")
        with col2:
            if pr_img.exists():
                st.image(str(pr_img), caption="PR Curves – Improved Pipeline", width="stretch")
            else:
                st.info("PR curve image not generated yet.")

        # Confusion matrices
        st.subheader("Confusion Matrices")
        cm_img = RESULTS_DIR / "confusion_matrices_improved.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrices – All 4 Improved Models", width="stretch")
        else:
            st.info("Confusion matrix image not generated yet.")

        # Threshold optimisation
        st.subheader("Threshold Optimisation")
        thr_img = RESULTS_DIR / "threshold_optimization_curves.png"
        if thr_img.exists():
            st.image(str(thr_img), caption="Precision / Recall / F1 vs Threshold", width="stretch")

        # Baseline vs improved comparison
        st.subheader("Baseline vs Improved Pipeline")
        cmp_img = RESULTS_DIR / "baseline_vs_improved_comparison.png"
        if cmp_img.exists():
            st.image(str(cmp_img), caption="Original vs Improved – Key Metrics", width="stretch")


# ============================================================
# 19. IMPROVED FEATURE IMPORTANCE
# ============================================================

elif page == "Improved Feature Importance":

    section(
        "Improved Feature Importance",
        "Top clinical features ranked by the improved XGBoost and Random Forest models."
    )

    if IMP is None:
        st.warning("Improved model artifacts not found. Run notebooks/train_improved_pipeline.py first.")
    else:
        imp_xgb_img = RESULTS_DIR / "feature_importance_xgb_improved.png"
        imp_rf_img  = RESULTS_DIR / "feature_importance_rf_improved.png"

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("XGBoost – Top 20 Features")
            if imp_xgb_img.exists():
                st.image(str(imp_xgb_img), caption="XGBoost Feature Importance", width="stretch")
            else:
                st.info("XGBoost importance image not generated yet.")
        with col2:
            st.subheader("Random Forest – Top 20 Features")
            if imp_rf_img.exists():
                st.image(str(imp_rf_img), caption="Random Forest Feature Importance", width="stretch")
            else:
                st.info("Random Forest importance image not generated yet.")

        # Interactive Plotly chart from loaded model
        st.subheader("Interactive Feature Importance (XGBoost)")
        try:
            imp_xgb_model = IMP["models"]["XGBoost"]
            imp_prep      = IMP["preprocessor"]
            feat_names    = imp_prep.get_feature_names_out()
            importances   = imp_xgb_model.feature_importances_
            feat_df = (
                __import__("pandas").DataFrame({"Feature": feat_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(25)
            )
            fig = px.bar(
                feat_df.sort_values("Importance"),
                x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="teal",
                title="Top 25 Features – Improved XGBoost"
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, width="stretch")
        except Exception as ex:
            st.warning(f"Could not build interactive chart: {ex}")


# ============================================================
# 20. IMPROVED PATIENT ASSESSMENT
# ============================================================

elif page == "Improved Patient Assessment":

    section(
        "Improved Patient Assessment",
        "Run risk prediction using the improved 4-model pipeline with optimised thresholds."
    )

    if IMP is None:
        st.warning("Improved model artifacts not found. Run notebooks/train_improved_pipeline.py first.")
    else:
        imp_models     = IMP["models"]
        imp_prep       = IMP["preprocessor"]
        imp_thresholds = IMP["thresholds"]
        imp_features   = IMP["input_features"]
        imp_meta       = IMP["metadata"]

        imp_num_cols  = imp_meta["numerical_columns"]
        imp_cat_cols  = imp_meta["categorical_columns"]
        imp_defaults  = imp_meta["defaults"]
        imp_cat_opts  = imp_meta["categorical_options"]

        imp_sel_model = st.sidebar.selectbox(
            "Improved model",
            list(imp_models.keys()),
            index=3,
            key="imp_sel_model"
        )

        st.subheader("Enter Patient Information")

        # imp_features are the engineered columns the preprocessor expects.
        # Build the form directly from that list so the DataFrame matches exactly.
        with st.form("imp_patient_form"):
            cols = st.columns(3)
            patient_data = {}
            for i, feat in enumerate(imp_features):
                col = cols[i % 3]
                default = imp_defaults.get(feat, 0)
                # Binary/flag features (0/1) → number input clamped 0-1
                # Categorical features listed in imp_cat_opts → selectbox
                if feat in imp_cat_opts:
                    opts = imp_cat_opts[feat]
                    patient_data[feat] = col.selectbox(feat, opts, key=f"imp_{feat}")
                else:
                    patient_data[feat] = col.number_input(
                        feat,
                        value=float(default),
                        key=f"imp_{feat}"
                    )

            submitted = st.form_submit_button("Predict Readmission Risk")

        if submitted:
            import pandas as _pd
            # DataFrame columns already match imp_features — no extra filtering needed
            row = _pd.DataFrame([patient_data])
            X_t = imp_prep.transform(row)
            model_obj = imp_models[imp_sel_model]
            prob = float(model_obj.predict_proba(X_t)[0, 1])
            thr  = imp_thresholds.get(imp_sel_model, 0.5)
            pred = "HIGH RISK – Likely Readmission" if prob >= thr else "LOW RISK – Unlikely Readmission"
            colour = "#DC2626" if prob >= thr else "#16A34A"

            st.markdown(f"""
<div style="background:{colour};color:white;padding:20px;border-radius:12px;margin-top:14px;">
  <h2 style="color:white;margin:0;">{pred}</h2>
  <p style="margin:6px 0 0 0;font-size:1.1rem;">
    Risk score: <strong>{prob:.4f}</strong> &nbsp;|&nbsp; Threshold: <strong>{thr:.4f}</strong>
    &nbsp;|&nbsp; Model: <strong>{imp_sel_model}</strong>
  </p>
</div>
""", unsafe_allow_html=True)

            report_df = _pd.DataFrame([{
                "Model": imp_sel_model,
                "Risk Score": prob,
                "Threshold": thr,
                "Prediction": pred,
                "Generated at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            st.download_button(
                "Download Prediction Report",
                report_df.to_csv(index=False),
                file_name="carepredict_improved_report.csv",
                mime="text/csv"
            )
            st.warning(
                "This is an educational ML prototype – not clinically validated. "
                "Do not use for patient-care decisions."
            )

elif page == "── Improved Pipeline ──":
    st.info("👈 Select one of the **Improved Pipeline** pages from the sidebar.")

# ============================================================
# 17. FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    CarePredict | Hospital Readmission Prediction
    <br>
    Academic Machine Learning Project
    <br>
    UCI Diabetes 130-US Hospitals Dataset
</div>
""", unsafe_allow_html=True)