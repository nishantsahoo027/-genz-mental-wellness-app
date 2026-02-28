# 🧠 Gen-Z Mental Wellness — ML Pipeline Dashboard

> A dual-target machine learning pipeline comparing **Regression** and **Classification** on Gen-Z mental wellness data, deployed as an interactive Streamlit web app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-username-genz-mental-wellness-app.streamlit.app)

---

## 📌 Project Overview

This project builds a complete end-to-end ML pipeline on a Gen-Z mental wellness dataset. It simultaneously solves **two different ML problems** using the same set of features:

| Target Variable | Type | Problem |
|----------------|------|---------|
| `Wellbeing_Index` | Continuous score (1–10) | Regression |
| `Burnout_Risk` | Low / Medium / High | Classification |

The pipeline follows 8 steps from the whiteboard plan:

```
Raw CSV Data
      ↓
Step 0 — Imports & Setup
      ↓
Step 1 — Load & Explore (EDA)
      ↓
Step 2 — Feature Engineering
         • Correlation Heatmap → Redundant Feature Detection
         • PCA — Principal Component Analysis
      ↓
Step 3 — Preprocessing (Encode, Split)
      ↓
      ┌──────────────────────────┐        ┌──────────────────────────┐
      │   🔴 CLASSIFICATION      │        │   🔵 REGRESSION           │
      │   Target: Burnout_Risk   │        │   Target: Wellbeing_Index │
      │                          │        │                           │
      │  Step 1: SMOTE Balancing │        │  Step 1: Train/Test Split │
      │  Step 2: Pie Charts      │        │  Step 2: StandardScaler   │
      │  Step 3: StandardScaler  │        │  Step 3: 6 Regressors +   │
      │  Step 4: 6 Classifiers + │        │          10-Fold CV       │
      │          10-Fold CV      │        │  Step 4: MAE / RMSE / R²  │
      │  Step 5: Acc/Prec/Rec/F1 │        │  Step 5: GridSearchCV     │
      │  Step 6: GridSearchCV    │        │          (on-demand btn)  │
      │          (on-demand btn) │        │  Step 6: Feature Importance│
      │  Step 7: Feature Imp     │        │  Step 7: Permutation XAI  │
      │  Step 8: Permutation XAI │        │                           │
      └──────────────────────────┘        └──────────────────────────┘
                          ↓
              🔮 Live Prediction (both targets)
```

---

## 📂 Project Structure

```
genz-mental-wellness-app/
├── app.py                                      # Streamlit web app (main file)
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
└── genz_mental_wellness_synthetic_dataset.csv  # Dataset (10,000 rows)
```

---

## 📊 Dataset

**File:** `genz_mental_wellness_synthetic_dataset.csv`
**Rows:** 10,000 synthetic samples · **Features:** 20 inputs + 2 targets

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Age of respondent (18–25) |
| `Gender` | Categorical | Male / Female |
| `Country` | Categorical | Country of residence |
| `Student_Working_Status` | Categorical | Student or Working |
| `Daily_Social_Media_Hours` | Numeric | Hours on social media per day |
| `Screen_Time_Hours` | Numeric | Total daily screen time |
| `Night_Scrolling_Frequency` | Numeric | Frequency of night scrolling (0–7) |
| `Online_Gaming_Hours` | Numeric | Daily hours spent gaming |
| `Content_Type_Preference` | Categorical | News / Gaming / Entertainment / Educational |
| `Exercise_Frequency_per_Week` | Numeric | Days per week exercising |
| `Daily_Sleep_Hours` | Numeric | Average sleep hours per night |
| `Caffeine_Intake_Cups` | Numeric | Daily caffeine cups |
| `Study_Work_Hours_per_Day` | Numeric | Hours studying or working per day |
| `Overthinking_Score` | Numeric | Self-reported overthinking (1–10) |
| `Anxiety_Score` | Numeric | Anxiety level (1–10) |
| `Mood_Stability_Score` | Numeric | Mood stability (1–10) |
| `Social_Comparison_Index` | Numeric | Social comparison tendency (1–10) |
| `Sleep_Quality_Score` | Numeric | Sleep quality rating (1–10) |
| `Motivation_Level` | Numeric | Motivation score (1–10) |
| `Emotional_Fatigue_Score` | Numeric | Emotional fatigue level (1–10) |

### Target Variables

| Variable | Type | Values | Task |
|----------|------|--------|------|
| `Wellbeing_Index` | Continuous | 1.0 – 10.0 | Regression |
| `Burnout_Risk` | Categorical | Low / Medium / High | Classification |

---

## 🖥️ App Tabs

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠 Gen-Z Mental Wellness — ML Pipeline Dashboard               │
├──────────────────┬──────────────────────────────────────────────┤
│  SIDEBAR         │                                              │
│                  │  📊 EDA & Feature Engineering                │
│  📂 Upload CSV   │     • Dataset overview metrics               │
│                  │     • Raw data preview                       │
│  Test Size       │     • Descriptive statistics                 │
│  [slider]        │     • Burnout_Risk pie chart                 │
│                  │     • Wellbeing_Index histogram              │
│  CV Folds        │     • Correlation heatmap                    │
│  [slider]        │     • PCA scree + cumulative variance        │
│                  │                                              │
│  Classifiers:    │  🔴 Classification (Burnout_Risk)            │
│  ✅ Log Reg      │     • SMOTE pie charts before/after          │
│  ✅ Dec Tree     │     • 6 models metrics table (highlighted)   │
│  ✅ Rand Forest  │     • CV details expandable                  │
│  ✅ Grad Boost   │     • Metrics bar chart comparison           │
│  ✅ SVM          │     • ROC-AUC table                          │
│  ✅ KNN          │     • Confusion matrix (best model)          │
│                  │     • 🔍 GridSearchCV button (on-demand)     │
│  Regressors:     │                                              │
│  ✅ Linear Reg   │  🔵 Regression (Wellbeing_Index)             │
│  ✅ Ridge        │     • 6 models metrics table (highlighted)   │
│  ✅ Lasso        │     • CV details expandable                  │
│  ✅ Dec Tree     │     • MAE / RMSE / R² bar charts             │
│  ✅ Rand Forest  │     • Actual vs Predicted scatter plot       │
│  ✅ Grad Boost   │     • 🔍 GridSearchCV button (on-demand)     │
│                  │                                              │
│  ▶ Run Pipeline  │  🧠 Explainable AI                           │
│                  │     • Feature importance (Classification)    │
│                  │     • Feature importance (Regression)        │
│                  │     • Permutation importance (Classification)│
│                  │     • Permutation importance (Regression)    │
│                  │     • Top 10 contributions tables            │
│                  │                                              │
│                  │  🔮 Predict                                  │
│                  │     • 20 input sliders + dropdowns           │
│                  │     • Burnout Risk prediction + icon         │
│                  │     • Wellbeing Index prediction             │
│                  │     • Class probability bar chart            │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 🔧 Pipeline — Step by Step

### Step 1 — Data Loading & EDA
- Load CSV into pandas DataFrame
- Check shape, dtypes, missing values
- View class distribution of `Burnout_Risk` and distribution of `Wellbeing_Index`

---

### Step 2 — Feature Engineering

**Correlation Heatmap:**
- Encode all categorical columns to numbers using `LabelEncoder`
- Compute Pearson correlation between every feature pair (values between -1 and +1)
- Visualize as a lower-triangle heatmap using seaborn
- Flag feature pairs with |r| > 0.85 as potentially redundant

**PCA — Principal Component Analysis:**
- Standardize all numeric features to mean=0, std=1
- Fit PCA and compute explained variance per component
- Plot Scree Plot (variance per component) and Cumulative Variance curve
- Find minimum components needed to explain ≥95% of total variance

---

### Step 3 — Preprocessing

| Step | What happens |
|------|-------------|
| **One-Hot Encoding** | `pd.get_dummies()` converts Gender, Country etc. to binary columns |
| **drop_first=True** | Drops one dummy column per feature to avoid multicollinearity |
| **LabelEncoder** | Converts Burnout_Risk (High/Low/Medium) → integers (0/1/2) |
| **Train/Test Split** | 80% train, 20% test · `stratify=y` for classification |
| **random_state=42** | Fixes randomness for reproducible results |

---

## 🔴 Part A — Classification (Burnout_Risk)

### Step 1 — SMOTE
The dataset has severe class imbalance. SMOTE fixes it:

```
Before SMOTE:  Low ≈ 0.6%   Medium ≈ 72%   High ≈ 7%
After SMOTE:   Low = 33.3%  Medium = 33.3%  High = 33.3%
```

**How SMOTE works:**
1. Pick a minority sample
2. Find its K nearest neighbors (also minority class)
3. Synthetically create a new point between them
4. Repeat until all classes are equal size

⚠️ Applied **only to training data** — never test data.

---

### Step 2 — Pie Charts (Before & After SMOTE)
Side-by-side pie charts showing class imbalance before SMOTE and perfect balance after.

---

### Step 3 — StandardScaler
```
z = (x − mean) / std
```
- `fit_transform` on training data — learns mean & std from training only
- `transform` on test data — applies same values (prevents data leakage)

---

### Step 4 — Six Classifiers with 10-Fold CV

| Model | Core Idea |
|-------|-----------|
| **Logistic Regression** | Sigmoid on linear combination of features → probability per class |
| **Decision Tree** | Recursive yes/no splits on feature thresholds |
| **Random Forest** | Hundreds of trees on random subsets → majority vote |
| **Gradient Boosting** | Sequential trees, each correcting prior errors |
| **SVM** | Maximum-margin hyperplane separating classes |
| **KNN** | Majority vote among K nearest training neighbors |

**10-Fold Cross-Validation (StratifiedKFold):**
```
[F1][F2][F3][F4][F5][F6][F7][F8][F9][F10]
Round 1:  Train F2–F10 → Test F1
Round 2:  Train F1,F3–F10 → Test F2
...
Round 10: Train F1–F9 → Test F10
Result:   mean ± std across 10 scores
```

---

### Step 5 — Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | Correct / Total | Overall % correct |
| **Precision** | TP / (TP+FP) | Of predicted positives, how many were right? |
| **Recall** | TP / (TP+FN) | Of actual positives, how many did we catch? |
| **F1-Score** | 2·(P·R)/(P+R) | Harmonic mean of Precision & Recall |
| **ROC-AUC** | Area under ROC | Separation quality across all thresholds (OvR) |

`average='weighted'` — per-class metric weighted by class size.

**Confusion Matrix:** 3×3 grid — rows = actual, cols = predicted, diagonal = correct.

---

### Step 6 — GridSearchCV (On-Demand Button)

Automatically tunes the best-performing classifier:

```
Example — Random Forest:
  n_estimators:     [100, 200]       → 2 options
  max_depth:        [None, 10, 20]   → 3 options
  min_samples_split:[2, 5]           → 2 options
  Total:  2 × 3 × 2 = 12 combinations × 10 folds = 120 fits
```

- Scoring: `f1_weighted`
- Shows: best params, CV F1, test F1 delta vs default, before/after chart

---

### Step 7 — Feature Importance (Random Forest)
Tracks how much each feature reduces Gini impurity across all trees.
Higher = more useful for predicting Burnout_Risk.

---

### Step 8 — Explainable AI (Permutation Importance)
Works for **any model** — not just trees:
1. Compute baseline F1 on test data
2. Shuffle one feature column randomly
3. Measure F1 drop → large drop = important feature
4. Repeat 20 times and average for stability

---

## 🔵 Part B — Regression (Wellbeing_Index)

### Key Differences from Classification
- No SMOTE — continuous target cannot be "balanced"
- No `stratify` in train/test split
- Uses `KFold` instead of `StratifiedKFold`

### Six Regressors

| Model | Core Idea |
|-------|-----------|
| **Linear Regression** | Minimize sum of squared errors — fits a hyperplane |
| **Ridge** | Linear + L2 penalty — shrinks large coefficients |
| **Lasso** | Linear + L1 penalty — zeros out unimportant coefficients |
| **Decision Tree** | Predicts mean value of samples in each leaf node |
| **Random Forest** | Averages predictions from many decision trees |
| **Gradient Boosting** | Sequential trees fitting residual errors |

### Regression Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** | mean(\|actual − pred\|) | Average absolute error — easy to interpret |
| **RMSE** | √mean((actual − pred)²) | Penalizes large errors more than MAE |
| **R²** | 1 − SS_res/SS_tot | % of variance explained (1.0 = perfect) |

### GridSearchCV (On-Demand Button)
Same pattern as classification — tunes best regressor:
- Scoring: `r2`
- Shows: best params, test R² delta, before/after chart for MAE/RMSE/R²
- Note: Linear Regression has no hyperparameters (handled gracefully)

---

## 🧠 Explainable AI Tab

| Chart | Model Used | What it measures |
|-------|-----------|-----------------|
| Feature Importance — Classification | Random Forest | Gini impurity reduction |
| Feature Importance — Regression | Random Forest | Gini impurity reduction |
| Permutation Importance — Classification | Best classifier | F1 drop on test data |
| Permutation Importance — Regression | Best regressor | R² drop on test data |

Adjustable "Top N features" slider (5–20) for all charts.

---

## 🔮 Predict Tab

Input sliders and dropdowns for all 20 features → click **Predict** → outputs:

| Output | Description |
|--------|-------------|
| **Burnout Risk** | 🔴 High / 🟡 Medium / 🟢 Low with color icon |
| **Wellbeing Index** | Predicted continuous score (e.g., 4.83 / 10) |
| **Probability Chart** | Horizontal bar chart showing confidence per class |

Uses the best-performing model from the pipeline run.

---

## ⚙️ Sidebar Controls

| Control | Description |
|---------|-------------|
| **Upload CSV** | Replace default dataset with your own |
| **Test Size** | Fraction for test split (0.10 – 0.40, default 0.20) |
| **CV Folds** | Number of cross-validation folds (3–15, default 10) |
| **Classifiers** | Toggle any of the 6 classifiers on/off |
| **Regressors** | Toggle any of the 6 regressors on/off |

---

## ⚙️ Installation & Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/genz-mental-wellness-app.git
cd genz-mental-wellness-app

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens automatically at: **http://localhost:8501**

---

## ☁️ Deployment — Streamlit Community Cloud

1. Push all 4 files to a **public** GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → click **New app**
4. Select your repo, set branch to `main`, main file to `app.py`
5. Click **Deploy** — ready in ~3 minutes

**Live URL format:**
```
https://YOUR_USERNAME-genz-mental-wellness-app.streamlit.app
```

**Auto-redeploy:** Push any change to GitHub → Streamlit redeploys automatically within 1–2 minutes.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.32.0 | Web app framework |
| `numpy` | ≥1.26.0 | Numerical computations |
| `pandas` | ≥2.0.0 | Data manipulation |
| `matplotlib` | ≥3.7.0 | Base plotting library |
| `seaborn` | ≥0.13.0 | Statistical visualizations |
| `scikit-learn` | ≥1.3.0 | ML models, metrics, preprocessing |
| `imbalanced-learn` | ≥0.11.0 | SMOTE for class balancing |

---

## 🔑 Key Concepts Quick Reference

| Concept | What it does |
|---------|-------------|
| **SMOTE** | Generates synthetic minority samples to fix class imbalance |
| **StandardScaler** | Rescales features to mean=0, std=1 |
| **Train/Test Split** | Simulates unseen real-world data for evaluation |
| **StratifiedKFold** | CV that maintains class ratio in each fold |
| **KFold** | Standard CV for regression (no classes to stratify) |
| **GridSearchCV** | Exhaustive search over hyperparameter combinations |
| **Accuracy** | % of all predictions that were correct |
| **F1-Score** | Best metric for imbalanced classification |
| **ROC-AUC** | Class separation quality across all thresholds |
| **Confusion Matrix** | Breakdown of correct and incorrect predictions per class |
| **MAE** | Average absolute error — easy to interpret |
| **RMSE** | Root mean squared error — penalizes large errors |
| **R²** | % of variance in target explained by the model |
| **Feature Importance** | Tree-based measure of each feature's predictive contribution |
| **Permutation Importance** | Model-agnostic XAI — measures F1/R² drop when feature is shuffled |
| **PCA** | Reduces dimensions while preserving maximum variance |
| **Correlation Heatmap** | Identifies redundant or multicollinear features |
| **Data Leakage** | When test data information influences training — prevented by fit/transform split |

---

## 👤 Author

Built as part of an Advanced Deep Learning project comparing classification and regression approaches on Gen-Z mental wellness data.
