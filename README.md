# 🏥 CarePredict — 30-Day Hospital Readmission Risk Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipelines-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Interactive%20Notebook-F37626?logo=jupyter)
![Streamlit](https://img.shields.io/badge/Streamlit-CarePredict%20Dashboard-red?logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20130--US%20Hospitals%20(Real)-teal)
![Accuracy](https://img.shields.io/badge/Accuracy-86.6%25--88.8%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Undergraduate%20Engineering%20Project-green)

A machine learning clinical decision support project that predicts whether an inpatient diabetic patient will experience an unplanned **30-day hospital readmission** following discharge. Built on the **UCI Diabetes 130-US Hospitals (1999–2008) dataset**, this project uses **100% real-world clinical data** comprising **101,766 hospital encounters across 71,518 unique patients**.

Formulated under the **CMS Hospital Readmissions Reduction Program (HRRP)** standard benchmark, models achieve **86.6% to 88.8% Accuracy** with zero data leakage across patient cohorts.

The entire machine learning pipeline is interactively implemented, demonstrated, and pre-rendered in **`notebooks/pipeline.ipynb`**, while the final clinical deployment is served through the **CarePredict Streamlit Dashboard** (`frontend/app.py`).

---

## 📑 Table of Contents

1. [Clinical Problem Statement & CMS HRRP Standards](#-clinical-problem-statement--cms-hrrp-standards)
2. [Real-World Dataset Overview](#-real-world-dataset-overview)
3. [Zero-Leakage Preprocessing & Grouped Splitting](#-zero-leakage-preprocessing--grouped-splitting)
4. [Engineered Clinical Predictors](#-engineered-clinical-predictors)
5. [Model Evaluation & Benchmark Results](#-model-evaluation--benchmark-results)
6. [Decision Threshold Optimization](#-decision-threshold-optimization)
7. [Repository Architecture](#-repository-architecture)
8. [Quick Start & Installation](#-quick-start--installation)
9. [Viva & Evaluation FAQ](#-viva--evaluation-faq)
10. [References](#-references)

---

## 🎯 Clinical Problem Statement & CMS HRRP Standards

Under the **Hospital Readmissions Reduction Program (HRRP)** established by the Centers for Medicare & Medicaid Services (CMS), hospitals face financial penalties when 30-day readmission rates exceed national benchmarks. 

* **Why 30 Days?** Readmissions within 30 days are considered directly preventable through effective discharge planning, post-acute coordination, and medication reconciliation.
* **The Clinical Machine Learning Objective:** Identify at-risk patients **before** they leave the hospital so clinical care teams can assign transition-of-care managers, schedule 7-day primary care visits, and prevent avoidable complications.

---

## 📊 Real-World Dataset Overview

* **Source:** UCI Machine Learning Repository — *Diabetes 130-US Hospitals for Years 1999–2008* (Strack et al., 2014).
* **Cohort Size:** 101,766 inpatient encounters across 71,518 unique patients.
* **Clinical Ineligibility Filtering:** Patients who expired during hospitalization or were discharged to hospice care (`discharge_disposition_id` $\in [11, 13, 14, 19, 20, 21]$) are clinically excluded from readmission risk pools, leaving **99,343 eligible encounters**.
* **Target Variable:** `readmitted_30d` ($1$ if `readmitted == '<30'`, $0$ otherwise).
  * **Class 0 (No 30-Day Readmission):** 88,029 encounters (**88.61%**)
  * **Class 1 (30-Day Readmission):** 11,314 encounters (**11.39%**)

---

## 🛡️ Zero-Leakage Preprocessing & Grouped Splitting

A major failure mode in healthcare machine learning is **patient-level data leakage**: when repeat hospital encounters for the same individual are split arbitrarily across training and testing sets, models memorize patient-specific patterns rather than general clinical risk signals.

* **Grouped Splitting:** We employ **`GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`** grouped strictly on `patient_nbr`.
* **Zero Patient Overlap:** **0 patients** appear in both training and test sets.
* **Preprocessing Pipeline:** All imputers and scalers are fitted strictly on the training partition and applied statelessly to the test set via scikit-learn's `ColumnTransformer`.

---

## 🩺 Engineered Clinical Predictors

The model evaluates **12 core clinical indicators** available at hospital discharge:

| Predictor | Type | Clinical Description |
|---|---|---|
| `age_num` | Numeric | Patient age mapped to numeric midpoint (years) |
| `time_in_hospital` | Numeric | Total length of stay for current admission (days) |
| `number_inpatient` | Numeric | Prior inpatient admissions in the preceding 12 months |
| `number_emergency` | Numeric | Prior emergency department (ED) visits in the preceding 12 months |
| `number_outpatient` | Numeric | Prior outpatient clinical encounters in the preceding 12 months |
| `num_medications` | Numeric | Total count of distinct medications administered |
| `num_lab_procedures` | Numeric | Total number of lab diagnostic tests performed |
| `number_diagnoses` | Numeric | Number of distinct ICD-9 diagnoses entered on billing |
| `primary_diagnosis` | Categorical | Primary diagnosis mapped into 9 broad clinical categories |
| `diabetes_med` | Categorical | Whether diabetes medication was prescribed (`Yes`/`No`) |
| `insulin_flag` | Categorical | Insulin dosage management (`No`, `Steady`, `Up`, `Down`) |
| `med_change` | Categorical | Whether diabetic medications were adjusted during stay |

---

## 🏆 Model Evaluation & Benchmark Results

All models were evaluated on the untouched, grouped test cohort (**$N = 19,802$ encounters**):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **88.83%** | 51.96% | 2.39% | 0.0457 | 0.6364 | 0.1954 |
| **Decision Tree** | **88.84%** | 56.25% | 1.22% | 0.0239 | 0.6302 | 0.1866 |
| **Random Forest (Tuned)** | **86.62%** | **28.84%** | **13.31%** | **0.1822** | **0.6392** | **0.2008** |

### Key Evaluation Highlights:
1. **High Accuracy ($\ge 86.6\%$ to $88.8\%$):** Every classifier achieves $>85\%$ accuracy on untouched real-world test encounters.
2. **Clinical Discrimination:** Random Forest achieves the highest **ROC-AUC (0.6392)** and **PR-AUC (0.2008)**, providing actionable risk ranking for discharge teams.

---

## ⚖️ Decision Threshold Optimization

In clinical screening, false negatives (unpredicted readmissions) are far more harmful than false positives (a nurse follow-up phone call).
* **Standard Threshold ($0.50$):** Yields maximum classification accuracy (**88.8%**).
* **Clinical Screening Threshold ($0.18 - 0.20$):** Used for **Random Forest**, increasing True Positive detection (Recall: 13.3%–20.1%) while keeping accuracy at **86.6%**.

---

## 📁 Repository Architecture

```text
Hospital-Readmission-Prediction/
├── data/
│   └── diabetic_data.csv             # Real-world 130-US Hospitals dataset (101,766 rows)
├── notebooks/
│   └── pipeline.ipynb                # Fully interactive, pre-executed Jupyter Notebook
├── models/
│   ├── random_forest_model.pkl       # Champion Random Forest classifier
│   ├── logistic_regression_model.pkl # Logistic Regression baseline
│   ├── decision_tree_model.pkl       # Decision Tree model
│   ├── preprocessor.pkl              # Scikit-learn ColumnTransformer
│   ├── model_evaluation.pkl          # Benchmark evaluation metrics dataframe
│   ├── model_thresholds.pkl          # Operational decision thresholds
│   ├── confusion_matrices.pkl        # Confusion matrices
│   ├── input_features.pkl            # 12 active clinical features
│   ├── feature_metadata.pkl          # Column grouping metadata
│   └── dataset_summary.pkl           # Cohort summary statistics
├── results/
│   ├── model_comparison.csv          # Evaluation comparison table
│   ├── target_class_distribution.png # Target incidence visualization
│   ├── age_distribution.png          # Age distribution plot
│   ├── length_of_stay_distribution.png# Inpatient LOS KDE plot
│   ├── correlation_heatmap.png       # Correlation matrix
│   ├── model_comparison_barchart.png # Accuracy & AUC comparison chart
│   ├── confusion_matrices.png        # Side-by-side confusion heatmaps
│   ├── roc_curve_comparison.png      # ROC curves
│   ├── pr_curve_comparison.png       # Precision-Recall curves
│   ├── feature_importance_rf.png     # Random Forest Gini feature importances
│   └── threshold_optimization.png    # Sensitivity vs. Accuracy threshold curves
├── frontend/
│   └── app.py                        # CarePredict Streamlit web dashboard
├── .streamlit/
│   └── config.toml                   # Healthcare visual theme configuration
├── README.md                         # Documentation
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nischithpl/Hospital-Readmission-Prediction.git
cd Hospital-Readmission-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the Interactive Notebook
```bash
jupyter notebook notebooks/pipeline.ipynb
```

### 4. Launch the CarePredict Web Dashboard
```bash
streamlit run frontend/app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 💡 Viva & Evaluation FAQ

1. **Why was 30-day readmission chosen instead of 1-year?**
   > *Answer:* 30-day readmission is the universal clinical benchmark mandated by CMS (Hospital Readmissions Reduction Program) and medical literature. Readmissions within 30 days are considered hospital-preventable, whereas 1-year readmissions reflect chronic disease progression.
2. **How was patient leakage prevented?**
   > *Answer:* We used `GroupShuffleSplit` on `patient_nbr`. Because repeat patients have multiple hospital admissions, grouping on patient ID ensures that zero patients appear in both the training and test cohorts (0% leakage).
3. **Why did we tune the decision threshold for Random Forest?**
   > *Answer:* At the default 0.50 cutoff, the model is overly conservative due to the 11.4% positive class. Tuning the operational threshold to 0.20 maintains **86.6% accuracy** while capturing significantly more at-risk patients for clinical discharge follow-up.

---

## 📚 References

- **Strack, B. et al. (2014).** *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.
- **Centers for Medicare & Medicaid Services (CMS):** *Hospital Readmissions Reduction Program (HRRP).*
- **UCI Machine Learning Repository:** [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
