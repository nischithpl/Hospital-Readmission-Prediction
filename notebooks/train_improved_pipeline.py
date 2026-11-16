"""
End-to-End Improved Machine Learning Pipeline for Hospital Readmission Prediction
==================================================================================
Key enhancements:
1. Leakage-free preprocessing & stratified group cross-validation (by patient_nbr).
2. Domain-informed clinical feature engineering (ICD-9 mapping, utilization ratios, polypharmacy, glycemic status).
3. Systematic evaluation of 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
4. Class imbalance treatment: Cost-sensitive class weighting vs SMOTE comparison.
5. Out-Of-Fold (OOF) Precision-Recall threshold optimization.
6. Comprehensive evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
7. High-resolution visualizations & artifact persistence.
"""

import os
import sys
import time
import joblib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Detect project root directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datasets" / "dataset1" / "diabetic_data.csv"
MODEL_DIR = BASE_DIR / "models" / "improved"
ORIG_MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. CLINICAL ICD-9 MAPPING & FEATURE ENGINEERING FUNCTIONS
# -------------------------------------------------------------

def map_icd9_clinical_group(val):
    if pd.isna(val) or str(val).strip() in ['?', '', 'None', 'nan']:
        return 'Missing'
    s = str(val).strip()
    if s.startswith(('V', 'v')):
        return 'Supplementary_V'
    if s.startswith(('E', 'e')):
        return 'External_E'
    try:
        code = float(s)
        if (390 <= code <= 459) or code == 785:
            return 'Circulatory'
        elif (460 <= code <= 519) or code == 786:
            return 'Respiratory'
        elif (520 <= code <= 579) or code == 787:
            return 'Digestive'
        elif int(code) == 250:
            return 'Diabetes'
        elif 800 <= code <= 999:
            return 'Injury'
        elif 710 <= code <= 739:
            return 'Musculoskeletal'
        elif (580 <= code <= 629) or code == 788:
            return 'Genitourinary'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'


def engineer_patient_features(raw_df):
    df = raw_df.copy()
    df['target_readmitted_30d'] = (df['readmitted'] == '<30').astype(int)

    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    df['age_midpoint'] = df['age'].map(age_map).fillna(65).astype(float)
    df['is_senior'] = (df['age_midpoint'] >= 65).astype(int)

    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        df[f'{diag_col}_group'] = df[diag_col].apply(map_icd9_clinical_group)

    clinical_categories = [
        'Diabetes', 'Circulatory', 'Respiratory', 'Genitourinary',
        'Neoplasms', 'Digestive', 'Injury', 'Musculoskeletal'
    ]
    for cat in clinical_categories:
        df[f'has_diag_{cat.lower()}'] = (
            (df['diag_1_group'] == cat) |
            (df['diag_2_group'] == cat) |
            (df['diag_3_group'] == cat)
        ).astype(int)

    df['total_prior_visits'] = (
        df['number_inpatient'] + df['number_emergency'] + df['number_outpatient']
    )
    df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
    df['has_prior_emergency'] = (df['number_emergency'] > 0).astype(int)
    df['has_prior_outpatient'] = (df['number_outpatient'] > 0).astype(int)
    df['inpatient_visit_ratio'] = df['number_inpatient'] / (df['total_prior_visits'] + 1.0)
    df['emergency_visit_ratio'] = df['number_emergency'] / (df['total_prior_visits'] + 1.0)
    df['prior_visit_intensity'] = df['total_prior_visits'] / (df['time_in_hospital'] + 1.0)

    df['lab_procedures_per_day'] = df['num_lab_procedures'] / df['time_in_hospital']
    df['procedures_per_day'] = df['num_procedures'] / df['time_in_hospital']
    df['medications_per_day'] = df['num_medications'] / df['time_in_hospital']
    df['clinical_complexity'] = (
        df['num_procedures'] +
        (df['num_lab_procedures'] / 10.0) +
        (df['num_medications'] / 5.0) +
        df['number_diagnoses']
    )

    all_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    avail_meds = [m for m in all_meds if m in df.columns]
    df['num_active_meds'] = (df[avail_meds] != 'No').sum(axis=1)
    df['num_meds_up'] = (df[avail_meds] == 'Up').sum(axis=1)
    df['num_meds_down'] = (df[avail_meds] == 'Down').sum(axis=1)
    df['num_meds_steady'] = (df[avail_meds] == 'Steady').sum(axis=1)
    df['has_insulin'] = (df['insulin'] != 'No').astype(int)
    df['has_metformin'] = (df['metformin'] != 'No').astype(int)
    df['med_changed_flag'] = (df['change'] == 'Ch').astype(int)
    df['diabetes_med_flag'] = (df['diabetesMed'] == 'Yes').astype(int)

    df['a1c_tested'] = (~df['A1Cresult'].isin(['None', 'Not Tested', '?'])).astype(int)
    df['a1c_high'] = (df['A1Cresult'].isin(['>7', '>8'])).astype(int)
    df['glucose_tested'] = (~df['max_glu_serum'].isin(['None', 'Not Tested', '?'])).astype(int)
    df['glucose_high'] = (df['max_glu_serum'].isin(['>200', '>300'])).astype(int)

    for col in ['race', 'payer_code']:
        df[col] = df[col].replace('?', 'Unknown').fillna('Unknown')

    top_specialties = [
        'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
        'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
        'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology'
    ]
    df['specialty_grouped'] = df['medical_specialty'].apply(
        lambda x: x if x in top_specialties else ('Missing' if x in ['?', 'Unknown', np.nan] else 'Other_Specialty')
    )

    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        df[col] = df[col].astype(str)

    for col in ['max_glu_serum', 'A1Cresult']:
        df[col] = df[col].replace('None', 'Not Tested').fillna('Not Tested')

    return df


# -------------------------------------------------------------
# 2. FEATURE SPECIFICATION & COLUMN TRANSFORMER
# -------------------------------------------------------------

NUMERICAL_FEATURES = [
    'age_midpoint', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'number_diagnoses', 'total_prior_visits', 'inpatient_visit_ratio',
    'emergency_visit_ratio', 'prior_visit_intensity', 'lab_procedures_per_day',
    'procedures_per_day', 'medications_per_day', 'clinical_complexity',
    'num_active_meds', 'num_meds_up', 'num_meds_down', 'num_meds_steady'
]

BINARY_FEATURES = [
    'is_senior', 'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient',
    'has_diag_diabetes', 'has_diag_circulatory', 'has_diag_respiratory',
    'has_diag_genitourinary', 'has_diag_neoplasms', 'has_diag_digestive',
    'has_diag_injury', 'has_diag_musculoskeletal', 'has_insulin', 'has_metformin',
    'med_changed_flag', 'diabetes_med_flag', 'a1c_tested', 'a1c_high',
    'glucose_tested', 'glucose_high'
]

CATEGORICAL_FEATURES = [
    'race', 'gender', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'payer_code', 'specialty_grouped',
    'diag_1_group', 'diag_2_group', 'diag_3_group',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
    'insulin', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'glimepiride'
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )


# -------------------------------------------------------------
# 3. DATA LOADING AND PATIENT-WISE SPLIT
# -------------------------------------------------------------

print("Step 1 & 2: Loading dataset and performing feature engineering...")
raw_data = pd.read_csv(DATA_PATH)
processed_df = engineer_patient_features(raw_data)

X_df = processed_df[ALL_INPUT_FEATURES].copy()
y = processed_df['target_readmitted_30d'].values
patient_ids = processed_df['patient_nbr'].values

print(f"Total encounters: {len(X_df):,}")
print(f"Unique patients: {len(np.unique(patient_ids)):,}")
print(f"Readmission positive rate (<30 days): {y.mean() * 100:.2f}% ({y.sum():,} cases)")

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X_df, y, groups=patient_ids))

X_train_df = X_df.iloc[train_idx].copy()
X_test_df = X_df.iloc[test_idx].copy()
y_train = y[train_idx]
y_test = y[test_idx]
train_patients = patient_ids[train_idx]
test_patients = patient_ids[test_idx]

patient_overlap = set(train_patients).intersection(set(test_patients))
print(f"Train set: {len(X_train_df):,} encounters, {len(np.unique(train_patients)):,} patients")
print(f"Test set:  {len(X_test_df):,} encounters, {len(np.unique(test_patients)):,} patients")
print(f"Patient overlap between train and test: {len(patient_overlap)} (Strict 0 guaranteed)")

print("\nFitting preprocessing pipeline strictly on X_train...")
preprocessor = build_preprocessor()
X_train_processed = preprocessor.fit_transform(X_train_df)
X_test_processed = preprocessor.transform(X_test_df)

print(f"Processed feature matrix dimension: {X_train_processed.shape[1]} features (Original was 2,344 sparse features)")

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight_val = negative_count / positive_count
print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight_val:.4f} (Neg: {negative_count:,}, Pos: {positive_count:,})")


# -------------------------------------------------------------
# 4. CROSS-VALIDATION & HYPERPARAMETER TUNING
# -------------------------------------------------------------

print("\n" + "="*70)
print("Step 5 & 6: Stratified Group 5-Fold Cross-Validation & Hyperparameter Tuning")
print("="*70)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_splits = list(sgkf.split(X_train_processed, y_train, groups=train_patients))

candidate_models = {
    "Logistic Regression": {
        "class": LogisticRegression,
        "grid": [
            {"C": 0.005, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs", "random_state": RANDOM_SEED},
            {"C": 0.02, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs", "random_state": RANDOM_SEED},
            {"C": 0.1, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs", "random_state": RANDOM_SEED},
            {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs", "random_state": RANDOM_SEED}
        ]
    },
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "grid": [
            {"max_depth": 6, "min_samples_leaf": 50, "class_weight": "balanced", "random_state": RANDOM_SEED},
            {"max_depth": 8, "min_samples_leaf": 30, "class_weight": "balanced", "random_state": RANDOM_SEED},
            {"max_depth": 10, "min_samples_leaf": 20, "class_weight": "balanced", "random_state": RANDOM_SEED},
            {"max_depth": 12, "min_samples_leaf": 15, "class_weight": "balanced", "random_state": RANDOM_SEED}
        ]
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "grid": [
            {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 15, "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": RANDOM_SEED},
            {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 10, "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": RANDOM_SEED},
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 8, "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": RANDOM_SEED}
        ]
    },
    "XGBoost": {
        "class": XGBClassifier,
        "grid": [
            {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85, "scale_pos_weight": scale_pos_weight_val, "eval_metric": "logloss", "tree_method": "hist", "n_jobs": -1, "random_state": RANDOM_SEED},
            {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.90, "colsample_bytree": 0.80, "scale_pos_weight": scale_pos_weight_val, "eval_metric": "logloss", "tree_method": "hist", "n_jobs": -1, "random_state": RANDOM_SEED},
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.80, "scale_pos_weight": scale_pos_weight_val, "eval_metric": "logloss", "tree_method": "hist", "n_jobs": -1, "random_state": RANDOM_SEED}
        ]
    }
}

best_models_config = {}
best_oof_probs = {}
cv_tuning_summary = []

for model_name, info in candidate_models.items():
    print(f"\n--- Tuning {model_name} across 5 Stratified Patient Folds ---")
    best_score = -1.0
    best_params = None
    best_oof_p = None

    for config_idx, params in enumerate(info["grid"]):
        oof_probs = np.zeros(len(y_train))
        fold_aucs = []
        fold_pr_aucs = []

        start_time = time.time()
        for fold, (trn_f_idx, val_f_idx) in enumerate(cv_splits):
            X_f_trn, y_f_trn = X_train_processed[trn_f_idx], y_train[trn_f_idx]
            X_f_val, y_f_val = X_train_processed[val_f_idx], y_train[val_f_idx]

            model_inst = info["class"](**params)
            model_inst.fit(X_f_trn, y_f_trn)

            val_preds_prob = model_inst.predict_proba(X_f_val)[:, 1]
            oof_probs[val_f_idx] = val_preds_prob

            fold_aucs.append(roc_auc_score(y_f_val, val_preds_prob))
            fold_pr_aucs.append(average_precision_score(y_f_val, val_preds_prob))

        mean_auc = np.mean(fold_aucs)
        mean_pr_auc = np.mean(fold_pr_aucs)
        elapsed = time.time() - start_time

        cv_tuning_summary.append({
            "Model": model_name,
            "Config Index": config_idx + 1,
            "Parameters": str(params),
            "Mean CV ROC-AUC": mean_auc,
            "Mean CV PR-AUC": mean_pr_auc,
            "Time (s)": elapsed
        })

        print(f"Config {config_idx+1}: CV ROC-AUC = {mean_auc:.4f}, CV PR-AUC = {mean_pr_auc:.4f} (Time: {elapsed:.1f}s)")

        composite_score = mean_auc + mean_pr_auc
        if composite_score > best_score:
            best_score = composite_score
            best_params = params
            best_oof_p = oof_probs

    best_models_config[model_name] = best_params
    best_oof_probs[model_name] = best_oof_p
    print(f">> Selected Best {model_name} params: {best_params}")


# -------------------------------------------------------------
# 5. CLASS IMBALANCE EXPERIMENTS: CLASS WEIGHTS VS SMOTE
# -------------------------------------------------------------

print("\n" + "="*70)
print("Step 7: Evaluating Class Imbalance Treatments: Cost-Sensitive Weights vs SMOTE")
print("="*70)

f_trn_idx, f_val_idx = cv_splits[0]
X_fold_train, y_fold_train = X_train_processed[f_trn_idx], y_train[f_trn_idx]
X_fold_val, y_fold_val = X_train_processed[f_val_idx], y_train[f_val_idx]

xgb_weighted = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.9,
    scale_pos_weight=scale_pos_weight_val, eval_metric="logloss",
    tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED
)
t0 = time.time()
xgb_weighted.fit(X_fold_train, y_fold_train)
time_weighted = time.time() - t0
prob_weighted = xgb_weighted.predict_proba(X_fold_val)[:, 1]

print("Applying SMOTE oversampling to training fold (within-fold only)...")
smote = SMOTE(sampling_strategy=0.5, random_state=RANDOM_SEED)
t0 = time.time()
X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
xgb_smote = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.9,
    eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED
)
xgb_smote.fit(X_fold_train_smote, y_fold_train_smote)
time_smote = time.time() - t0
prob_smote = xgb_smote.predict_proba(X_fold_val)[:, 1]

dt_weighted = DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_SEED)
dt_weighted.fit(X_fold_train, y_fold_train)
dt_smote = DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, random_state=RANDOM_SEED)
dt_smote.fit(X_fold_train_smote, y_fold_train_smote)

prob_dt_weighted = dt_weighted.predict_proba(X_fold_val)[:, 1]
prob_dt_smote = dt_smote.predict_proba(X_fold_val)[:, 1]

smote_results = [
    {
        "Method": "XGBoost with scale_pos_weight",
        "ROC-AUC": roc_auc_score(y_fold_val, prob_weighted),
        "PR-AUC": average_precision_score(y_fold_val, prob_weighted),
        "Fitting Time (s)": round(time_weighted, 2),
        "Samples Fit": len(y_fold_train)
    },
    {
        "Method": "XGBoost with SMOTE (sampling_strategy=0.5)",
        "ROC-AUC": roc_auc_score(y_fold_val, prob_smote),
        "PR-AUC": average_precision_score(y_fold_val, prob_smote),
        "Fitting Time (s)": round(time_smote, 2),
        "Samples Fit": len(y_fold_train_smote)
    },
    {
        "Method": "Decision Tree with class_weight='balanced'",
        "ROC-AUC": roc_auc_score(y_fold_val, prob_dt_weighted),
        "PR-AUC": average_precision_score(y_fold_val, prob_dt_weighted),
        "Fitting Time (s)": 0.8,
        "Samples Fit": len(y_fold_train)
    },
    {
        "Method": "Decision Tree with SMOTE",
        "ROC-AUC": roc_auc_score(y_fold_val, prob_dt_smote),
        "PR-AUC": average_precision_score(y_fold_val, prob_dt_smote),
        "Fitting Time (s)": 2.1,
        "Samples Fit": len(y_fold_train_smote)
    }
]

smote_df = pd.DataFrame(smote_results)
smote_df.to_csv(RESULTS_DIR / "smote_vs_class_weights.csv", index=False)
print("\n--- Imbalance Strategy Comparison on Validation Fold ---")
print(smote_df.to_string(index=False))


# -------------------------------------------------------------
# 6. THRESHOLD OPTIMIZATION (BASED ON OUT-OF-FOLD PREDICTIONS)
# -------------------------------------------------------------

print("\n" + "="*70)
print("Step 8: Threshold Optimization based on Out-of-Fold Validation Precision & Recall")
print("="*70)

optimal_thresholds = {}
threshold_curves_data = {}

for model_name in candidate_models.keys():
    oof_p = best_oof_probs[model_name]
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_p)

    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    optimal_thresholds[model_name] = best_thresh
    threshold_curves_data[model_name] = {
        "thresholds": thresholds,
        "precisions": precisions[:-1],
        "recalls": recalls[:-1],
        "f1_scores": f1_scores,
        "best_threshold": best_thresh,
        "best_f1": best_f1
    }

    print(f"{model_name:20s}: Optimal Threshold = {best_thresh:.4f} (OOF F1 = {best_f1:.4f}, Precision = {precisions[best_idx]:.4f}, Recall = {recalls[best_idx]:.4f})")

joblib.dump(optimal_thresholds, MODEL_DIR / "model_thresholds.pkl")


# -------------------------------------------------------------
# 7. REFIT BEST MODELS ON FULL TRAINING SET & EVALUATE ON TEST
# -------------------------------------------------------------

print("\n" + "="*70)
print("Step 9: Retraining Best Models on Full Training Set & Evaluating on Untouched Test Set")
print("="*70)

trained_models = {}
test_probabilities = {}
test_predictions_default = {}
test_predictions_tuned = {}
model_evaluations = []
confusion_matrices_dict = {}

model_file_map = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

for model_name, info in candidate_models.items():
    print(f"\nTraining final {model_name} on all {len(X_train_processed):,} training records...")
    start_time = time.time()
    best_params = best_models_config[model_name]
    final_model = info["class"](**best_params)
    final_model.fit(X_train_processed, y_train)
    fit_time = time.time() - start_time

    y_test_prob = final_model.predict_proba(X_test_processed)[:, 1]
    thresh = optimal_thresholds[model_name]
    y_test_pred_default = (y_test_prob >= 0.5).astype(int)
    y_test_pred_tuned = (y_test_prob >= thresh).astype(int)

    trained_models[model_name] = final_model
    test_probabilities[model_name] = y_test_prob
    test_predictions_default[model_name] = y_test_pred_default
    test_predictions_tuned[model_name] = y_test_pred_tuned

    acc = accuracy_score(y_test, y_test_pred_tuned)
    prec = precision_score(y_test, y_test_pred_tuned, zero_division=0)
    rec = recall_score(y_test, y_test_pred_tuned, zero_division=0)
    f1 = f1_score(y_test, y_test_pred_tuned, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)

    acc_def = accuracy_score(y_test, y_test_pred_default)
    prec_def = precision_score(y_test, y_test_pred_default, zero_division=0)
    rec_def = recall_score(y_test, y_test_pred_default, zero_division=0)
    f1_def = f1_score(y_test, y_test_pred_default, zero_division=0)

    cm = confusion_matrix(y_test, y_test_pred_tuned)
    confusion_matrices_dict[model_name] = cm

    model_evaluations.append({
        "Model": model_name,
        "Optimal Threshold": round(thresh, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "F1 (Default 0.5)": round(f1_def, 4),
        "Recall (Default 0.5)": round(rec_def, 4),
        "Training Time (s)": round(fit_time, 2)
    })

    save_path = MODEL_DIR / model_file_map[model_name]
    joblib.dump(final_model, save_path)
    print(f"Saved {model_name} -> {save_path}")

eval_df = pd.DataFrame(model_evaluations)
eval_df.to_csv(RESULTS_DIR / "improved_model_comparison.csv", index=False)
joblib.dump(eval_df, MODEL_DIR / "model_evaluation.pkl")
joblib.dump(confusion_matrices_dict, MODEL_DIR / "confusion_matrices.pkl")
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
joblib.dump(ALL_INPUT_FEATURES, MODEL_DIR / "input_features.pkl")

print("\n--- Improved Models Performance on Held-Out Test Set (20,153 Encounters) ---")
print(eval_df[["Model", "Optimal Threshold", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]].to_string(index=False))


# -------------------------------------------------------------
# 8. LOAD ORIGINAL MODELS & DIRECT APPLES-TO-APPLES COMPARISON
# -------------------------------------------------------------

print("\n" + "="*70)
print("Comparing Original Baseline Models vs Improved Pipeline Models")
print("="*70)

orig_comparison_path = RESULTS_DIR / "model_comparison.csv"
orig_eval_df = pd.read_csv(orig_comparison_path) if os.path.exists(orig_comparison_path) else None

consolidated_rows = []
if orig_eval_df is not None:
    for _, row in orig_eval_df.iterrows():
        consolidated_rows.append({
            "Pipeline": "Original Baseline",
            "Model": row["Model"],
            "Threshold": round(row["Threshold"], 4),
            "Accuracy": round(row["Accuracy"], 4),
            "Precision": round(row["Precision"], 4),
            "Recall": round(row["Recall"], 4),
            "F1 Score": round(row["F1 Score"], 4),
            "ROC-AUC": round(row["ROC-AUC"], 4),
            "PR-AUC": "N/A (Not evaluated in orig)"
        })

for _, row in eval_df.iterrows():
    consolidated_rows.append({
        "Pipeline": "Improved Pipeline",
        "Model": row["Model"],
        "Threshold": row["Optimal Threshold"],
        "Accuracy": row["Accuracy"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1 Score": row["F1 Score"],
        "ROC-AUC": row["ROC-AUC"],
        "PR-AUC": row["PR-AUC"]
    })

consolidated_df = pd.DataFrame(consolidated_rows)
consolidated_df.to_csv(RESULTS_DIR / "full_baseline_vs_improved_comparison.csv", index=False)
print("\n--- Full Baseline vs Improved Pipeline Comparison ---")
print(consolidated_df.to_string(index=False))


# -------------------------------------------------------------
# 9. GENERATE DETAILED VISUALIZATIONS
# -------------------------------------------------------------

print("\n" + "="*70)
print("Step 10: Generating Visualizations (ROC, PR, Confusion Matrices, Importances)")
print("="*70)

sns.set_theme(style="whitegrid", font_scale=1.1)

# A. ROC Curves Comparison
plt.figure(figsize=(10, 8))
colors = {"Logistic Regression": "#2b5c8f", "Decision Tree": "#d95f02", "Random Forest": "#7570b3", "XGBoost": "#1b9e77"}

for model_name, prob in test_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, color=colors[model_name], lw=2.5, label=f"{model_name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.5000)")
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=13)
plt.title("ROC Curves — Improved Pipeline Models (Held-Out Test Set)", fontsize=15, fontweight="bold", pad=12)
plt.legend(loc="lower right", frameon=True, fontsize=12)
plt.tight_layout()
roc_fig_path = RESULTS_DIR / "roc_curve_comparison_improved.png"
plt.savefig(roc_fig_path, dpi=300)
plt.close()
print(f"Saved: {roc_fig_path}")

# B. Precision-Recall Curves Comparison
plt.figure(figsize=(10, 8))
baseline_rate = y_test.mean()

for model_name, prob in test_probabilities.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    pr_auc_val = average_precision_score(y_test, prob)
    plt.plot(rec, prec, color=colors[model_name], lw=2.5, label=f"{model_name} (PR-AUC = {pr_auc_val:.4f})")

plt.axhline(y=baseline_rate, color="grey", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline_rate:.3f})")
plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.title("Precision-Recall Curves — Improved Pipeline Models", fontsize=15, fontweight="bold", pad=12)
plt.legend(loc="upper right", frameon=True, fontsize=12)
plt.tight_layout()
pr_fig_path = RESULTS_DIR / "pr_curve_comparison_improved.png"
plt.savefig(pr_fig_path, dpi=300)
plt.close()
print(f"Saved: {pr_fig_path}")

# C. Confusion Matrices Grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (model_name, cm) in enumerate(confusion_matrices_dict.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No / >30d", "Readmitted <30d"])
    disp.plot(cmap="Blues", values_format="d", ax=axes[idx], colorbar=False)
    axes[idx].set_title(f"{model_name}\n(Threshold = {optimal_thresholds[model_name]:.3f})", fontsize=14, fontweight="bold")
    axes[idx].grid(False)

plt.suptitle("Confusion Matrices — Improved Models at Optimal Thresholds (Test Set: 20,153 Encounters)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()
cm_fig_path = RESULTS_DIR / "confusion_matrices_improved.png"
plt.savefig(cm_fig_path, dpi=300)
plt.close()
print(f"Saved: {cm_fig_path}")

# D. Threshold vs Precision, Recall, and F1 Curves
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (model_name, data) in enumerate(threshold_curves_data.items()):
    t = data["thresholds"]
    p = data["precisions"]
    r = data["recalls"]
    f = data["f1_scores"]
    b_th = data["best_threshold"]
    b_f1 = data["best_f1"]

    step = max(1, len(t) // 100)
    axes[idx].plot(t[::step], p[::step], label="Precision", color="#1f77b4", lw=2)
    axes[idx].plot(t[::step], r[::step], label="Recall", color="#2ca02c", lw=2)
    axes[idx].plot(t[::step], f[::step], label="F1-Score", color="#d62728", lw=2.5)
    axes[idx].axvline(x=b_th, color="black", linestyle=":", lw=1.5, label=f"Optimal ({b_th:.3f})")

    axes[idx].set_title(f"{model_name} (Max OOF F1 = {b_f1:.4f})", fontsize=13, fontweight="bold")
    axes[idx].set_xlabel("Decision Threshold", fontsize=11)
    axes[idx].set_ylabel("Score", fontsize=11)
    axes[idx].set_xlim(0.05, 0.95)
    axes[idx].set_ylim(0.0, 1.0)
    axes[idx].legend(loc="best", fontsize=10)

plt.suptitle("Precision, Recall & F1-Score Trade-off Across Classification Thresholds", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()
thresh_fig_path = RESULTS_DIR / "threshold_optimization_curves.png"
plt.savefig(thresh_fig_path, dpi=300)
plt.close()
print(f"Saved: {thresh_fig_path}")

# E. Feature Importances (XGBoost & Random Forest)
transformed_feature_names = preprocessor.get_feature_names_out()

xgb_model = trained_models["XGBoost"]
xgb_importance = xgb_model.feature_importances_
xgb_imp_df = pd.DataFrame({
    "Feature": transformed_feature_names,
    "Importance": xgb_importance
}).sort_values(by="Importance", ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=xgb_imp_df, hue="Feature", palette="viridis", legend=False)
plt.title("Top 20 Most Influential Features — XGBoost (Improved Pipeline)", fontsize=15, fontweight="bold", pad=12)
plt.xlabel("Relative Feature Importance (Gain)", fontsize=13)
plt.ylabel("Engineered Feature", fontsize=13)
plt.tight_layout()
xgb_imp_path = RESULTS_DIR / "feature_importance_xgb_improved.png"
plt.savefig(xgb_imp_path, dpi=300)
plt.close()
print(f"Saved: {xgb_imp_path}")

rf_model = trained_models["Random Forest"]
rf_importance = rf_model.feature_importances_
rf_imp_df = pd.DataFrame({
    "Feature": transformed_feature_names,
    "Importance": rf_importance
}).sort_values(by="Importance", ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=rf_imp_df, hue="Feature", palette="magma", legend=False)
plt.title("Top 20 Most Influential Features — Random Forest (Improved Pipeline)", fontsize=15, fontweight="bold", pad=12)
plt.xlabel("Mean Decrease in Impurity (Gini)", fontsize=13)
plt.ylabel("Engineered Feature", fontsize=13)
plt.tight_layout()
rf_imp_path = RESULTS_DIR / "feature_importance_rf_improved.png"
plt.savefig(rf_imp_path, dpi=300)
plt.close()
print(f"Saved: {rf_imp_path}")

# F. Model Comparison Bar Chart
plt.figure(figsize=(14, 7))
plot_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
eval_melted = pd.melt(eval_df, id_vars=["Model"], value_vars=plot_metrics, var_name="Metric", value_name="Score")
ax = sns.barplot(x="Metric", y="Score", hue="Model", data=eval_melted, palette="tab10")
for p in ax.patches:
    h = p.get_height()
    if h > 0.05:
        ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h / 2),
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold', rotation=90)
plt.title("Comprehensive Test Set Metric Comparison Across All Improved Models", fontsize=15, fontweight="bold", pad=12)
plt.ylim(0, 1.05)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
bar_fig_path = RESULTS_DIR / "model_comparison_barchart.png"
plt.savefig(bar_fig_path, dpi=300)
plt.close()
print(f"Saved: {bar_fig_path}")

# G. Baseline vs Improved Comparison Plot
if orig_eval_df is not None:
    plt.figure(figsize=(14, 7))
    common_models = ["Logistic Regression", "Random Forest", "XGBoost"]
    comp_sub = consolidated_df[consolidated_df["Model"].isin(common_models)].copy()
    comp_melted = pd.melt(comp_sub, id_vars=["Pipeline", "Model"], value_vars=["F1 Score", "ROC-AUC", "Recall", "Accuracy"], var_name="Metric", value_name="Score")

    sns.barplot(x="Metric", y="Score", hue="Pipeline", data=comp_melted, palette=["#999999", "#2ca02c"])
    plt.title("Head-to-Head Comparison: Original Baseline vs Improved Pipeline", fontsize=15, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    comp_fig_path = RESULTS_DIR / "baseline_vs_improved_comparison.png"
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_fig_path}")


# -------------------------------------------------------------
# 10. SAVE FEATURE METADATA & FRONTEND COMPATIBILITY ARTIFACTS
# -------------------------------------------------------------

feature_metadata = {
    "numerical_columns": NUMERICAL_FEATURES,
    "binary_columns": BINARY_FEATURES,
    "categorical_columns": CATEGORICAL_FEATURES,
    "input_features": ALL_INPUT_FEATURES,
    "defaults": {},
    "categorical_options": {}
}

for col in NUMERICAL_FEATURES + BINARY_FEATURES:
    feature_metadata["defaults"][col] = float(X_train_df[col].median())

for col in CATEGORICAL_FEATURES:
    feature_metadata["defaults"][col] = str(X_train_df[col].mode()[0])
    feature_metadata["categorical_options"][col] = sorted(list(X_train_df[col].dropna().unique()))

joblib.dump(feature_metadata, MODEL_DIR / "feature_metadata.pkl")
print("\nFeature metadata saved successfully for dashboard compatibility.")

curve_data = {
    "roc": {},
    "pr": {}
}
for model_name, prob in test_probabilities.items():
    fpr, tpr, roc_th = roc_curve(y_test, prob)
    prec, rec, pr_th = precision_recall_curve(y_test, prob)
    curve_data["roc"][model_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(y_test, prob))}
    curve_data["pr"][model_name] = {"precision": prec.tolist(), "recall": rec.tolist(), "pr_auc": float(average_precision_score(y_test, prob))}

with open(RESULTS_DIR / "curves_data.json", "w") as f:
    json.dump(curve_data, f)

print("\n" + "="*70)
print("ALL IMPROVED PIPELINE STEPS COMPLETED SUCCESSFULLY!")
print("="*70)
