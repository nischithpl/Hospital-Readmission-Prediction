# 🏥 CarePredict — 30-Day Hospital Readmission Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipelines-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-CarePredict%20Dashboard-red?logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-HLT--005%20Synthetic-teal)
![Status](https://img.shields.io/badge/Status-Undergraduate%20Engineering%20Project-green)

A machine learning project that predicts whether an inpatient will experience an unplanned **30-day hospital readmission** following discharge. Built on the **HLT-005 Synthetic Hospital Admission Dataset**, this project implements a leakage-free preprocessing pipeline, patient-level cross-validation (`StratifiedGroupKFold`), and three supervised classification algorithms (**Logistic Regression**, **Decision Tree**, and **Random Forest**).

The system features **CarePredict**, an interactive Streamlit clinical dashboard providing real-time patient readmission risk assessment, transparent decision factors, and an academic model performance dashboard.

> ⚠️ **Disclaimer & Notice:** This project uses a **synthetic clinical dataset (HLT-005)** designed for healthcare machine learning education and benchmarking. It does **not** contain real protected health information (PHI) and is an academic engineering prototype not intended for direct clinical decision-making.

---

## 📑 Table of Contents

1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Problem Statement](#-problem-statement)
3. [Dataset Description (HLT-005)](#-dataset-description-hlt-005)
4. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
5. [Data Preprocessing & Leakage Prevention](#-data-preprocessing--leakage-prevention)
6. [Feature Selection & Clinical Mapping](#-feature-selection--clinical-mapping)
7. [Machine Learning Algorithms](#-machine-learning-algorithms)
8. [Experimental Results & Evaluation](#-experimental-results--evaluation)
9. [CarePredict Web Dashboard](#-carepredict-web-dashboard)
10. [Repository Structure](#-repository-structure)
11. [How to Run the Project](#-how-to-run-the-project)
12. [Limitations of Synthetic Data](#-limitations-of-synthetic-data)

---

## 🎯 Project Overview & Objectives

Under the Hospital Readmissions Reduction Program (HRRP), acute care hospitals face substantial financial penalties for higher-than-expected 30-day readmission rates. Unplanned readmissions represent adverse clinical outcomes, patient distress, and avoidable healthcare spending.

### Core Objectives:
1. **Analyze** inpatient admission patterns, comorbidities, and demographic risk factors in the HLT-005 synthetic dataset.
2. **Engineer a Leakage-Free Pipeline**: Implement patient-level group splitting (`StratifiedGroupKFold`) so patients with multiple hospital admissions never leak between train and test sets.
3. **Select 7 Clinically Meaningful Features**: Focus on predictors accessible prior to discharge (Age, Prior Admissions, Prior ED Visits, Length of Stay, Comorbidity Score, Diagnosis Category, ICU Admission).
4. **Train & Benchmark 3 Classifiers**: Compare Logistic Regression, Decision Tree, and Random Forest using balanced class weighting to handle target imbalance (17.3% readmission rate).
5. **Prioritize Recall & Discrimination**: Optimize models to catch high-risk patients who would benefit most from post-discharge care coordination.
6. **Deliver an Interactive Dashboard**: Connect the exact trained pipeline directly to a Streamlit interface for live demonstration and college evaluation.

---

## 🔍 Problem Statement

Given a patient's clinical indicators upon admission and throughout their inpatient stay, predict whether the patient will be readmitted to the hospital within **30 days** of discharge:

$$\text{Target: } y \in \{0, 1\}$$
- $y = 0$: No readmission within 30 days
- $y = 1$: Readmitted within 30 days of discharge

Because only **17.26%** of hospital stays result in 30-day readmission, the dataset exhibits class imbalance. Standard accuracy is an insufficient metric; models must be evaluated on **Recall (Sensitivity)**, **Precision**, **F1-score**, **ROC-AUC**, and **PR-AUC**.

---

## 🗄️ Dataset Description (HLT-005)

- **Source**: [HLT-005 – Synthetic Hospital Admission Dataset (Hugging Face)](https://huggingface.co/datasets/xpertsystems/hlt005-sample)
- **File**: `data/admissions.csv`
- **Total Records**: 5,000 hospital encounters
- **Unique Patients**: 1,580 individual patients (`patient_id`)
- **Total Attributes**: 76 raw columns covering demographics, triage vitals, DRG diagnostic codes, severity scores, and utilization history.
- **Target Column**: `readmit_flag_30d` (0 = Not Readmitted: 4,137 | 1 = Readmitted: 863)

### Data Leakage Columns Explicitly Excluded:
To ensure genuine predictive validity, downstream outcome metrics and target-derived risk scores were strictly removed prior to training:
- `lace_plus_score`, `readmit_risk_30d`, `readmit_risk_60d`, `readmit_risk_90d`
- `readmit_risk_category`, `readmit_cause_category`
- `discharge_date`, `discharge_disposition`

---

## 📊 Exploratory Data Analysis (EDA)

All exploratory visualizations are automatically generated and saved in `results/`:

| Visualization | Description |
|---|---|
| `results/target_class_distribution.png` | Target class breakdown: 82.7% non-readmitted vs 17.3% 30-day readmissions |
| `results/age_distribution.png` | Stacked age distribution stratified by readmission status |
| `results/length_of_stay_distribution.png` | Histogram & KDE of hospital stay duration (`los_days`) |
| `results/correlation_heatmap.png` | Pearson correlation matrix across numerical clinical predictors |
| `results/feature_distributions.png` | 2x2 multi-panel distribution of prior admits, prior ED visits, CCI, and ICU status |
| `results/missing_values.png` | Full audit of missing entries across all 76 columns |

### Key EDA Takeaways:
- **Missing Values**: The 7 selected modeling features contain **0% missing values (100% complete data)**. Missing entries in the broader dataset are confined to conditional downstream attributes (`readmit_cause_category`, `hac_type`, `age_group`).
- **Length of Stay**: Patients with extended hospitalizations (>7 days) exhibit elevated readmission probability.
- **Comorbidity & Prior Utilization**: Patients with $\ge 2$ prior admissions in the last 12 months show significantly higher readmission likelihood.

---

## 🛠️ Data Preprocessing & Leakage Prevention

1. **Patient-Level Isolation (`StratifiedGroupKFold`)**:
   - The 5,000 encounters belong to 1,580 unique patients (many patients have 2–6 admissions).
   - Random splitting would place encounters from the **same patient** into both training and test sets, leading to severe patient data leakage.
   - Using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `patient_id` guarantees **ZERO patient overlap** between train (3,992 encounters, 79.8%) and held-out test (1,008 encounters, 20.2%).
2. **Numerical Pipeline**:
   - `SimpleImputer(strategy='median')`: Robust against outliers.
   - `StandardScaler()`: Standardizes features to mean 0, unit variance.
3. **Categorical Pipeline**:
   - `SimpleImputer(strategy='most_frequent')`: Handles potential missing categories.
   - `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`: Converts categorical strings into dense binary indicators.
4. **Pipeline Encapsulation**:
   - All transformations are bundled into a `ColumnTransformer` fitted **strictly on training data**, preventing lookahead leakage.

---

## 🧬 Feature Selection & Clinical Mapping

To keep the clinical input form intuitive and suitable for presentation while maintaining a 1:1 match with the model preprocessor, **7 core features** were selected:

| Feature Name | Type | Range / Options | Clinical Significance |
|---|---|---|---|
| `age` | Numeric | 0 – 120 years | Older patients face higher post-acute vulnerability |
| `prior_admits_12mo` | Numeric | 0 – 20 admissions | Recurring inpatient history indicates chronic instability |
| `prior_ed_visits_6mo` | Numeric | 0 – 20 visits | Frequent emergency visits reflect ambulatory care gaps |
| `los_days` | Numeric | 0.5 – 90.0 days | Acute illness severity and complex recovery trajectory |
| `cci_score` | Numeric | 0 – 20 points | Charlson Comorbidity Index measuring chronic disease burden |
| `diagnosis_category` | Categorical | 10 Clinical Groups | Primary diagnosis mapped from MS-DRG codes |
| `icu_flag_str` | Categorical | "No" / "Yes" | Critical care stay during hospitalization |

### Clinical Diagnosis Categorization (from 25 MS-DRGs):
1. **Infection/Sepsis** (Septicemia, Severe Sepsis)
2. **Cardiovascular** (Heart Failure, Chest Pain, Percutaneous Cardiac Stents)
3. **Orthopedic/Musculoskeletal** (Major Joint Replacement, Spinal Fusion, Fractures)
4. **Respiratory** (Pneumonia, Respiratory Infections)
5. **Neurological** (Stroke, Intracranial Hemorrhage, Seizures)
6. **Digestive/GI** (Esophagitis, Gastrointestinal Disorders)
7. **Renal** (Acute & Chronic Kidney Failure)
8. **Psychiatric/Substance** (Psychoses, Substance Abuse Rehabilitation)
9. **Obstetric/Newborn** (Vaginal Delivery, Normal Newborn)
10. **Other/Symptoms** (General Symptoms & Other Categories)

---

## 🤖 Machine Learning Algorithms

Three supervised classification algorithms were implemented and trained using cost-sensitive class weighting (`class_weight='balanced'`) to penalize false negatives:

1. **Logistic Regression**:
   - Serves as the parametric, interpretable clinical baseline.
   - Regularized with $L_2$ penalty ($C=0.1$) to prevent overfitting on collinear indicators.
2. **Decision Tree Classifier**:
   - White-box rule-based model.
   - Regularized with `max_depth=4` and `min_samples_leaf=20` to prevent deep branch memorization and ensure high interpretability.
3. **Random Forest Classifier**:
   - Non-parametric ensemble consisting of 150 bootstrapped decision trees.
   - Constrained to `max_depth=5` and `min_samples_leaf=15` with random feature subspacing to reduce variance and capture non-linear clinical risk interactions.

---

## 📈 Experimental Results & Evaluation

All models were evaluated on the held-out test partition (1,008 encounters, 181 readmissions) with fixed random seeds for strict reproducibility:

### Official Model Comparison Table:
| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 59.72% | 23.02% | 53.04% | 0.3211 | 0.6139 | 0.2462 |
| **Decision Tree** | 63.69% | 23.50% | 45.30% | 0.3094 | 0.6094 | 0.2421 |
| **Random Forest (Best)** | **63.19%** | **25.64%** | **55.25%** | **0.3503** | **0.6228** | **0.2444** |

### Confusion Matrices (Held-out Test Set, N=1,008):
- **Random Forest**: True Negatives = 537 | False Positives = 290 | False Negatives = 81 | **True Positives = 100**
- **Logistic Regression**: True Negatives = 506 | False Positives = 321 | False Negatives = 85 | **True Positives = 96**
- **Decision Tree**: True Negatives = 560 | False Positives = 267 | False Negatives = 99 | **True Positives = 82**

### Performance Analysis & Selection Justification:
- **Random Forest is the Champion Model**: It achieves the highest **ROC-AUC (0.6228)**, highest **Recall (55.25%)**, and highest **F1-Score (0.3503)**.
- **Clinical Priority on Recall**: In hospital readmission prevention, the cost of a False Negative (missing a patient who will be readmitted and penalized under HRRP) is dramatically higher than a False Positive (offering low-cost post-discharge follow-up to a patient who did not need it). Random Forest successfully identifies over 55% of all readmissions.
- **Discrimination**: With a positive prevalence of 17.3%, the baseline no-skill PR-AUC is 0.173. All three models score ~0.245, confirming genuine predictive signal over random chance.

---

## 💻 CarePredict Web Dashboard

The frontend is implemented in **Streamlit** (`frontend/app.py`), retaining the classic CarePredict theme (deep navy `#123650`, teal `#176B87`, and crisp white metric cards):

### Section 1: Patient Readmission Prediction
- 7 simple form fields matching the exact trained pipeline.
- Instant risk prediction: **⚠️ Readmission Likely** or **✅ Readmission Not Likely**.
- Estimated probability percentage and model selection dropdown.
- Automated clinical risk driver highlights.
- Reset button to quickly restore defaults.
- Downloadable prediction report (CSV).

### Section 2: Model Performance Dashboard
- **Dataset Overview**: Encounters, unique patients, and readmission counts.
- **Model Comparison Table & Grouped Bar Chart**: Live evaluation metrics.
- **Interactive Confusion Matrix**: Dropdown selector to view individual and side-by-side matrices with TN/FP/FN/TP breakdowns.
- **ROC & Precision-Recall Curves**: High-resolution curves with baseline comparisons.
- **Random Forest Feature Importance**: Gini ranking of predictors.

### Section 3: Exploratory Data Analysis (EDA)
- Displays all 6 exploratory figures for immediate reference during presentation questions.

---

## 📁 Repository Structure

```
Hospital-Readmission-Prediction/
├── data/
│   └── admissions.csv                     # HLT-005 synthetic hospital admission dataset (5,000 rows)
├── frontend/
│   └── app.py                             # CarePredict Streamlit web application
├── models/
│   ├── logistic_regression_model.pkl      # Trained Logistic Regression model
│   ├── decision_tree_model.pkl            # Trained Decision Tree model
│   ├── random_forest_model.pkl            # Trained Random Forest model (champion)
│   ├── preprocessor.pkl                   # Fitted Scikit-Learn ColumnTransformer
│   ├── model_evaluation.pkl               # Test set evaluation metrics DataFrame
│   ├── model_thresholds.pkl               # Classification thresholds
│   ├── confusion_matrices.pkl             # Confusion matrices dict
│   ├── input_features.pkl                 # Exact 7 feature names list
│   ├── feature_metadata.pkl               # Feature types, options, and defaults
│   └── dataset_summary.pkl                # Summary statistics for dashboard cards
├── notebooks/
│   ├── pipeline.ipynb                     # Interactive step-by-step Jupyter Notebook
│   └── train_pipeline.py                  # Standalone executable pipeline script
├── results/
│   ├── target_class_distribution.png      # Target distribution plot
│   ├── age_distribution.png               # Age distribution plot
│   ├── length_of_stay_distribution.png    # Length of stay plot
│   ├── missing_values.png                 # Missing values audit plot
│   ├── correlation_heatmap.png            # Feature correlation matrix plot
│   ├── feature_distributions.png          # 2x2 multi-panel clinical predictors plot
│   ├── model_comparison_barchart.png      # Grouped bar chart comparing all 3 models
│   ├── confusion_matrices.png             # Side-by-side confusion matrices plot
│   ├── roc_curve_comparison.png           # Multi-model ROC curves
│   ├── pr_curve_comparison.png            # Multi-model Precision-Recall curves
│   ├── feature_importance_rf.png          # Random Forest Gini feature importances
│   └── model_comparison.csv               # Comparison metrics CSV
└── README.md                              # Comprehensive project documentation
```

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure Python 3.10+ is installed with standard data science packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib requests
```

### 2. Re-train Models & Generate Visualizations (Optional)
To execute the complete end-to-end pipeline and regenerate all models, results, and figures:
```bash
python notebooks/train_pipeline.py
```

### 3. Launch the CarePredict Web Dashboard
To start the interactive web application:
```bash
streamlit run frontend/app.py
```
The application will open automatically at `http://localhost:8501`.

---

## ⚠️ Limitations of Synthetic Data

While synthetic healthcare datasets provide an accessible, privacy-preserving framework for machine learning development, several inherent limitations must be acknowledged:

1. **Simulated Clinical Correlations**: Synthetic generators model statistical distributions and probabilistic correlations, but they may lack rare physiological nuances and complex multi-morbid interactions found in real Electronic Health Records (EHR).
2. **Simplified Diagnosis Hierarchy**: Real hospital billing data contains thousands of ICD-10-CM diagnosis and procedure codes with intricate temporal sequences, whereas synthetic benchmarks simplify diagnoses into coarse DRG clusters.
3. **Unmeasured Social Determinants of Health (SDoH)**: In real healthcare environments, housing stability, caregiver support, and prescription drug affordability strongly influence 30-day readmissions. These dimensions are largely approximated or absent in synthetic benchmarks.
4. **Generalizability**: Performance achieved on synthetic data serves as a proof-of-concept and cannot be extrapolated directly to clinical deployment without validation on prospectively collected real patient cohorts under Institutional Review Board (IRB) oversight.
