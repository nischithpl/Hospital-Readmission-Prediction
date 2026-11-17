"""
End-to-End Training & Evaluation Pipeline for Hospital Readmission Prediction
=============================================================================
Dataset: HLT-005 Synthetic Hospital Admission Dataset (admissions.csv)
Target: readmit_flag_30d (0 = Not Readmitted, 1 = Readmitted within 30 days)

Features (7 clinical fields matching the patient frontend):
1. age: Patient age in years
2. prior_admits_12mo: Inpatient admissions in prior 12 months
3. prior_ed_visits_6mo: Emergency department visits in prior 6-12 months
4. los_days: Length of hospital stay in days
5. cci_score: Charlson Comorbidity Index score
6. diagnosis_category: Primary diagnosis clinical category (mapped from ms_drg_label)
7. icu_flag_str: ICU admission indicator ('No' / 'Yes')

Models:
1. Logistic Regression (class_weight='balanced')
2. Decision Tree Classifier (class_weight='balanced')
3. Random Forest Classifier (class_weight='balanced')

Splitting:
StratifiedGroupKFold (5-fold, grouped by patient_id)
Prevents patient-level data leakage across multiple admissions.
"""

import os
import sys
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "admissions.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

print("=" * 70)
print("HOSPITAL READMISSION PREDICTION — ML PIPELINE")
print("Dataset: HLT-005 Synthetic Hospital Admission Dataset")
print("=" * 70)

# Set visualization style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# -------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------
df_raw = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
print(f"Unique patients: {df_raw['patient_id'].nunique():,}")

# -------------------------------------------------------------
# 2. Exploratory Data Analysis & Visualizations
# -------------------------------------------------------------
print("\n--- Generating EDA Visualizations ---")

# (A) Target Class Distribution
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
counts = df_raw["readmit_flag_30d"].value_counts().sort_index()
total_n = len(df_raw)
bars = ax.bar(["Not Readmitted (0)", "Readmitted 30d (1)"], counts.values, color=["#176B87", "#E74C3C"], width=0.55, edgecolor="#123650", linewidth=1.2)
for bar in bars:
    height = bar.get_height()
    pct = height / total_n * 100
    ax.annotate(f"{height:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Target Class Distribution: 30-Day Hospital Readmission", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Number of Encounters", fontsize=12)
ax.set_ylim(0, max(counts.values) * 1.18)
plt.tight_layout()
target_plot_path = RESULTS_DIR / "target_class_distribution.png"
plt.savefig(target_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {target_plot_path.name}")

# (B) Age Distribution by Readmission Status
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
sns.histplot(data=df_raw, x="age", hue="readmit_flag_30d", multiple="stack", bins=30,
             palette={0: "#176B87", 1: "#E74C3C"}, ax=ax, edgecolor="white", alpha=0.85)
ax.set_title("Age Distribution by 30-Day Readmission Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Patient Age (Years)", fontsize=12)
ax.set_ylabel("Number of Encounters", fontsize=12)
legend = ax.get_legend()
if legend:
    legend.set_title("Status")
    for t, label in zip(legend.texts, ["Not Readmitted (0)", "Readmitted (1)"]):
        t.set_text(label)
plt.tight_layout()
age_plot_path = RESULTS_DIR / "age_distribution.png"
plt.savefig(age_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {age_plot_path.name}")

# (C) Length of Stay Distribution
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
sns.histplot(data=df_raw, x="los_days", hue="readmit_flag_30d", bins=30, kde=True,
             palette={0: "#176B87", 1: "#E74C3C"}, ax=ax, element="step", common_norm=False)
ax.set_title("Length of Hospital Stay (Days) by Readmission Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Length of Stay (Days)", fontsize=12)
ax.set_ylabel("Density / Count", fontsize=12)
legend = ax.get_legend()
if legend:
    legend.set_title("Status")
    for t, label in zip(legend.texts, ["Not Readmitted (0)", "Readmitted (1)"]):
        t.set_text(label)
plt.tight_layout()
los_plot_path = RESULTS_DIR / "length_of_stay_distribution.png"
plt.savefig(los_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {los_plot_path.name}")

# (D) Missing Values Visualization
null_counts = df_raw.isnull().sum()
null_cols = null_counts[null_counts > 0].sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
if len(null_cols) > 0:
    bars = ax.barh(null_cols.index, null_cols.values, color="#F39C12", edgecolor="#B9770E")
    for bar in bars:
        w = bar.get_width()
        pct = w / total_n * 100
        ax.annotate(f"{w:,} ({pct:.1f}%)", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, total_n * 1.15)
else:
    ax.text(0.5, 0.5, "No Missing Values in Modeling Features (100% Complete)", ha="center", va="center", fontsize=13)
ax.set_title("Missing Values Audit (All 76 Columns)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Number of Missing Values", fontsize=12)
plt.tight_layout()
missing_plot_path = RESULTS_DIR / "missing_values.png"
plt.savefig(missing_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {missing_plot_path.name}")

# (E) Correlation Heatmap
corr_features = ["readmit_flag_30d", "age", "prior_admits_12mo", "prior_ed_visits_6mo", "los_days", "cci_score", "icu_flag"]
corr_matrix = df_raw[corr_features].corr()
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="Blues", cbar=True, ax=ax,
            linewidths=0.5, linecolor="#DFE7EF", annot_kws={"size": 10, "weight": "bold"})
ax.set_title("Correlation Heatmap: Clinical Predictors & Readmission", fontsize=14, fontweight="bold", pad=12)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
corr_plot_path = RESULTS_DIR / "correlation_heatmap.png"
plt.savefig(corr_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {corr_plot_path.name}")

# (F) Multi-panel Feature Distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
# Prior admits
sns.countplot(data=df_raw, x="prior_admits_12mo", hue="readmit_flag_30d", palette={0: "#176B87", 1: "#E74C3C"}, ax=axes[0, 0])
axes[0, 0].set_title("Prior Admissions (Last 12 Mo)", fontweight="bold")
axes[0, 0].set_xlabel("Admissions Count")
axes[0, 0].set_ylabel("Count")
# Prior ED visits
sns.countplot(data=df_raw, x="prior_ed_visits_6mo", hue="readmit_flag_30d", palette={0: "#176B87", 1: "#E74C3C"}, ax=axes[0, 1])
axes[0, 1].set_title("Prior Emergency Visits (Last 6-12 Mo)", fontweight="bold")
axes[0, 1].set_xlabel("Emergency Visits Count")
axes[0, 1].set_ylabel("Count")
# CCI Score
sns.countplot(data=df_raw, x="cci_score", hue="readmit_flag_30d", palette={0: "#176B87", 1: "#E74C3C"}, ax=axes[1, 0])
axes[1, 0].set_title("Charlson Comorbidity Index (CCI)", fontweight="bold")
axes[1, 0].set_xlabel("Comorbidity Score")
axes[1, 0].set_ylabel("Count")
# ICU Flag
sns.countplot(data=df_raw, x="icu_flag", hue="readmit_flag_30d", palette={0: "#176B87", 1: "#E74C3C"}, ax=axes[1, 1])
axes[1, 1].set_title("ICU Admission (0=No, 1=Yes)", fontweight="bold")
axes[1, 1].set_xlabel("ICU Indicator")
axes[1, 1].set_ylabel("Count")

for ax in axes.flat:
    legend = ax.get_legend()
    if legend:
        legend.set_title("Status")
        for t, label in zip(legend.texts, ["Not Readmit", "Readmitted"]):
            t.set_text(label)

fig.suptitle("Key Clinical Predictor Distributions by Readmission Status", fontsize=15, fontweight="bold", y=1.00)
plt.tight_layout()
feat_dist_path = RESULTS_DIR / "feature_distributions.png"
plt.savefig(feat_dist_path, bbox_inches="tight")
plt.close()
print(f"Saved: {feat_dist_path.name}")

# -------------------------------------------------------------
# 3. Clinical Feature Mapping & Selection
# -------------------------------------------------------------
print("\n--- Clinical Feature Engineering & Mapping ---")

def map_diagnosis(drg_label):
    """Maps 25 granular MS-DRG labels to 10 high-level clinical categories."""
    drg = str(drg_label).lower()
    if "septicemia" in drg or "sepsis" in drg:
        return "Infection/Sepsis"
    elif "heart" in drg or "cardio" in drg or "chest pain" in drg:
        return "Cardiovascular"
    elif "pneumonia" in drg or "respiratory" in drg:
        return "Respiratory"
    elif "hemorrhage" in drg or "stroke" in drg or "seizure" in drg:
        return "Neurological"
    elif "joint" in drg or "spinal" in drg or "fracture" in drg or "deran" in drg:
        return "Orthopedic/Musculoskeletal"
    elif "renal" in drg:
        return "Renal"
    elif "esophagitis" in drg or "gi" in drg:
        return "Digestive/GI"
    elif "delivery" in drg or "newborn" in drg:
        return "Obstetric/Newborn"
    elif "psychoses" in drg or "alcohol" in drg or "drug" in drg:
        return "Psychiatric/Substance"
    else:
        return "Other/Symptoms"

df = df_raw.copy()
df["diagnosis_category"] = df["ms_drg_label"].apply(map_diagnosis)
df["icu_flag_str"] = df["icu_flag"].map({0: "No", 1: "Yes"})

# Define the exact 7 modeling features
input_features = [
    "age",
    "prior_admits_12mo",
    "prior_ed_visits_6mo",
    "los_days",
    "cci_score",
    "diagnosis_category",
    "icu_flag_str"
]

num_cols = ["age", "prior_admits_12mo", "prior_ed_visits_6mo", "los_days", "cci_score"]
cat_cols = ["diagnosis_category", "icu_flag_str"]
target_col = "readmit_flag_30d"

print("Selected features (1:1 matching frontend input form):")
for f in input_features:
    print(f"  • {f}")

# -------------------------------------------------------------
# 4. Leakage-Free Train / Test Split (Patient-Level Grouping)
# -------------------------------------------------------------
print("\n--- Data Splitting (StratifiedGroupKFold) ---")
# Using StratifiedGroupKFold ensures that all records of any given patient
# reside strictly in EITHER train OR test (never split across both).
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
train_idx, test_idx = next(sgkf.split(df, df[target_col], groups=df["patient_id"]))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

overlap = len(set(train_df["patient_id"]).intersection(set(test_df["patient_id"])))
print(f"Training encounters: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%) | Readmission rate: {train_df[target_col].mean()*100:.2f}%")
print(f"Testing encounters:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%) | Readmission rate: {test_df[target_col].mean()*100:.2f}%")
print(f"Patient overlap between Train & Test: {overlap} (ZERO leakage confirmed ✅)")

X_train = train_df[input_features]
y_train = train_df[target_col]
X_test = test_df[input_features]
y_test = test_df[target_col]

# -------------------------------------------------------------
# 5. Preprocessing Pipeline Construction
# -------------------------------------------------------------
print("\n--- Building Scikit-Learn Preprocessing Pipeline ---")

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# Fit on training data ONLY
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Extract transformed feature names
encoded_cat_names = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_cols).tolist()
all_transformed_feature_names = num_cols + encoded_cat_names
print(f"Transformed feature space: {len(all_transformed_feature_names)} dense features")

# -------------------------------------------------------------
# 6. Model Training & Evaluation (3 Required Algorithms)
# -------------------------------------------------------------
print("\n--- Training Machine Learning Models ---")

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        C=0.1,
        max_iter=1000,
        random_state=RANDOM_SEED
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=4,
        min_samples_leaf=20,
        random_state=RANDOM_SEED
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=15,
        random_state=RANDOM_SEED
    )
}

eval_records = []
confusion_matrices_dict = {}
roc_data = {}
pr_data = {}
thresholds_dict = {}

for name, clf in models.items():
    print(f"\nTraining: {name}...")
    clf.fit(X_train_proc, y_train)
    
    y_pred = clf.predict(X_test_proc)
    y_proba = clf.predict_proba(X_test_proc)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    
    roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
    pr_data[name] = {"precision": precision_vals.tolist(), "recall": recall_vals.tolist(), "ap": float(pr_auc)}
    confusion_matrices_dict[name] = cm
    thresholds_dict[name] = 0.50
    
    eval_records.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

eval_df = pd.DataFrame(eval_records)
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 70)
print(eval_df.to_string(index=False))

# Save metrics CSV
comparison_csv_path = RESULTS_DIR / "model_comparison.csv"
eval_df.to_csv(comparison_csv_path, index=False)
print(f"\nSaved comparison metrics: {comparison_csv_path.name}")

# -------------------------------------------------------------
# 7. Model Evaluation Visualizations
# -------------------------------------------------------------
print("\n--- Generating Evaluation Visualizations ---")

# (A) Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
melted_df = eval_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

palette = ["#176B87", "#23988D", "#F39C12", "#E74C3C", "#2C3E50"]
sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette="Blues_r", ax=ax, edgecolor="#123650")
ax.set_title("Model Comparison: Accuracy, Precision, Recall, F1-score & ROC-AUC", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(title="Model", loc="upper right", frameon=True)

# Add value labels
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{height:.2f}",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
cmp_barchart_path = RESULTS_DIR / "model_comparison_barchart.png"
plt.savefig(cmp_barchart_path, bbox_inches="tight")
plt.close()
print(f"Saved: {cmp_barchart_path.name}")

# (B) Side-by-Side Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
for idx, (name, cm) in enumerate(confusion_matrices_dict.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xticklabels(["Not Readmit (0)", "Readmitted (1)"])
    ax.set_yticklabels(["Not Readmit (0)", "Readmitted (1)"])

plt.tight_layout()
cm_plot_path = RESULTS_DIR / "confusion_matrices.png"
plt.savefig(cm_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {cm_plot_path.name}")

# Individual confusion matrix plots
for name, cm in confusion_matrices_dict.items():
    slug = name.lower().replace(" ", "_")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"{name} — Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xticklabels(["Not Readmit (0)", "Readmitted (1)"])
    ax.set_yticklabels(["Not Readmit (0)", "Readmitted (1)"])
    plt.tight_layout()
    p_path = RESULTS_DIR / f"confusion_matrix_{slug}.png"
    plt.savefig(p_path, bbox_inches="tight")
    plt.close()

# (C) ROC Curve Comparison
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
colors = {"Logistic Regression": "#176B87", "Decision Tree": "#F39C12", "Random Forest": "#E74C3C"}

for name, clf in models.items():
    y_proba = clf.predict_proba(X_test_proc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[name], linewidth=2.2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
ax.legend(loc="lower right", frameon=True, fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
plt.tight_layout()
roc_plot_path = RESULTS_DIR / "roc_curve_comparison.png"
plt.savefig(roc_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {roc_plot_path.name}")

# (D) Precision-Recall Curve Comparison
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
no_skill = y_test.mean()

for name, clf in models.items():
    y_proba = clf.predict_proba(X_test_proc)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    ap_val = average_precision_score(y_test, y_proba)
    ax.plot(recall_vals, precision_vals, label=f"{name} (PR-AUC = {ap_val:.3f})", color=colors[name], linewidth=2.2)

ax.plot([0, 1], [no_skill, no_skill], "k--", alpha=0.5, label=f"Baseline Rate ({no_skill*100:.1f}%)")
ax.set_title("Precision-Recall (PR) Curves", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
ax.legend(loc="upper right", frameon=True, fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
plt.tight_layout()
pr_plot_path = RESULTS_DIR / "pr_curve_comparison.png"
plt.savefig(pr_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {pr_plot_path.name}")

# (E) Random Forest Feature Importance Graph
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_

# Clean readable names for display
clean_names = []
for f in all_transformed_feature_names:
    name_clean = f.replace("cat__diagnosis_category_", "Dx: ")
    name_clean = name_clean.replace("cat__icu_flag_str_Yes", "ICU Admission: Yes")
    name_clean = name_clean.replace("num__", "")
    name_clean = name_clean.replace("age", "Age")
    name_clean = name_clean.replace("prior_admits_12mo", "Prior Admissions (12mo)")
    name_clean = name_clean.replace("prior_ed_visits_6mo", "Prior ED Visits (6mo)")
    name_clean = name_clean.replace("los_days", "Length of Stay (Days)")
    name_clean = name_clean.replace("cci_score", "Comorbidity Score (CCI)")
    clean_names.append(name_clean)

feat_imp_df = pd.DataFrame({"Feature": clean_names, "Importance": importances}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
bars = ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color="#176B87", edgecolor="#123650", height=0.65)
for bar in bars:
    w = bar.get_width()
    ax.annotate(f"{w*100:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, fontweight="bold")

ax.set_title("Random Forest — Feature Importance", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Relative Importance (Gini Importance)", fontsize=12)
ax.set_xlim(0, max(importances) * 1.25)
plt.tight_layout()
rf_imp_plot_path = RESULTS_DIR / "feature_importance_rf.png"
plt.savefig(rf_imp_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved: {rf_imp_plot_path.name}")

# -------------------------------------------------------------
# 8. Save Model Artifacts
# -------------------------------------------------------------
print("\n--- Saving Model Artifacts for Production & Frontend ---")

# Save each trained model
joblib.dump(models["Logistic Regression"], MODELS_DIR / "logistic_regression_model.pkl")
joblib.dump(models["Decision Tree"], MODELS_DIR / "decision_tree_model.pkl")
joblib.dump(models["Random Forest"], MODELS_DIR / "random_forest_model.pkl")

# Save preprocessor
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

# Save evaluation DataFrame
joblib.dump(eval_df, MODELS_DIR / "model_evaluation.pkl")

# Save thresholds
joblib.dump(thresholds_dict, MODELS_DIR / "model_thresholds.pkl")

# Save confusion matrices
joblib.dump(confusion_matrices_dict, MODELS_DIR / "confusion_matrices.pkl")

# Save input features
joblib.dump(input_features, MODELS_DIR / "input_features.pkl")

# Save feature metadata & defaults for frontend
feature_metadata = {
    "input_features": input_features,
    "numerical_columns": num_cols,
    "categorical_columns": cat_cols,
    "defaults": {
        "age": int(df_raw["age"].median()),
        "prior_admits_12mo": int(df_raw["prior_admits_12mo"].median()),
        "prior_ed_visits_6mo": int(df_raw["prior_ed_visits_6mo"].median()),
        "los_days": round(float(df_raw["los_days"].median()), 1),
        "cci_score": int(df_raw["cci_score"].median()),
        "diagnosis_category": "Cardiovascular",
        "icu_flag_str": "No"
    },
    "categorical_options": {
        "diagnosis_category": sorted(df["diagnosis_category"].unique().tolist()),
        "icu_flag_str": ["No", "Yes"]
    }
}
joblib.dump(feature_metadata, MODELS_DIR / "feature_metadata.pkl")

# Save dataset summary for frontend overview
dataset_summary = {
    "total_records": int(len(df_raw)),
    "unique_patients": int(df_raw["patient_id"].nunique()),
    "readmitted_count": int(df_raw["readmit_flag_30d"].sum()),
    "not_readmitted_count": int((df_raw["readmit_flag_30d"] == 0).sum()),
    "readmission_rate_pct": round(float(df_raw["readmit_flag_30d"].mean() * 100), 2)
}
joblib.dump(dataset_summary, MODELS_DIR / "dataset_summary.pkl")

print("All model artifacts saved successfully in models/ folder!")
print("Pipeline execution completed successfully. ✅")
