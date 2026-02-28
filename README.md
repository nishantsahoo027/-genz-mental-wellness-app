# 🧠 Gen-Z Mental Wellness — ML Pipeline Dashboard

> A dual-target machine learning pipeline comparing **Regression** and **Classification** on Gen-Z mental wellness data, deployed as an interactive Streamlit web app.

---

## 📌 Project Overview

This project builds a complete end-to-end ML pipeline on a Gen-Z mental wellness dataset. It simultaneously solves **two different ML problems** using the same set of features:

| Target Variable | Type | Problem |
|----------------|------|---------|
| `Wellbeing_Index` | Continuous score (1–10) | Regression |
| `Burnout_Risk` | Low / Medium / High | Classification |

The pipeline follows these steps — directly from the whiteboard plan:

```
Data Loading & EDA
      ↓
Feature Engineering (Correlation, PCA, Redundant Feature Detection)
      ↓
      ┌─────────────────────┐         ┌──────────────────────┐
      │   CLASSIFICATION    │         │     REGRESSION        │
      │   Burnout_Risk      │         │   Wellbeing_Index     │
      │ Step 1: SMOTE       │         │ Step 1: Train/Test    │
      │ Step 2: Pie Charts  │         │        Split          │
      │ Step 3: Scaling     │         │ Step 2: Scaling       │
      │ Step 4: 6 Models +  │         │ Step 3: 6 Models +    │
      │         10-Fold CV  │         │         10-Fold CV    │
      │ Step 5: Metrics     │         │ Step 4: Metrics       │
      │ Step 6: GridSearch  │         │ Step 5: GridSearch    │
      │ Step 7: Feature Imp │         │ Step 6: Feature Imp   │
      │ Step 8: XAI         │         │ Step 7: XAI           │
      └─────────────────────┘         └──────────────────────┘
                        ↓
              Final Comparison + Live Prediction
```

---

## 📂 Project Structure

```
genz-mental-wellness-app/
├── app.py                                      # Streamlit web app
├── requirements.txt                            # Python dependencies
└── genz_mental_wellness_synthetic_dataset.csv  # Dataset
```

---

## 📊 Dataset

**File:** `genz_mental_wellness_synthetic_dataset.csv`
**Rows:** 10,000 synthetic samples
**Features:** 20 input features + 2 target variables

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Age of the person (18–25) |
| `Gender` | Categorical | Male / Female |
| `Country` | Categorical | Country of residence |
| `Student_Working_Status` | Categorical | Student or Working |
| `Daily_Social_Media_Hours` | Numeric | Hours spent on social media per day |
| `Screen_Time_Hours` | Numeric | Total daily screen time |
| `Night_Scrolling_Frequency` | Numeric | How often scrolling at night (0–7) |
| `Online_Gaming_Hours` | Numeric | Daily hours spent gaming |
| `Content_Type_Preference` | Categorical | News / Gaming / Entertainment / Educational |
| `Exercise_Frequency_per_Week` | Numeric | Days per week exercising |
| `Daily_Sleep_Hours` | Numeric | Average sleep hours |
| `Caffeine_Intake_Cups` | Numeric | Daily caffeine cups |
| `Study_Work_Hours_per_Day` | Numeric | Hours studying or working per day |
| `Overthinking_Score` | Numeric | Self-reported overthinking (1–10) |
| `Anxiety_Score` | Numeric | Anxiety level (1–10) |
| `Mood_Stability_Score` | Numeric | Mood stability (1–10) |
| `Social_Comparison_Index` | Numeric | Social comparison tendency (1–10) |
| `Sleep_Quality_Score` | Numeric | Sleep quality (1–10) |
| `Motivation_Level` | Numeric | Motivation (1–10) |
| `Emotional_Fatigue_Score` | Numeric | Emotional fatigue (1–10) |

### Target Variables

| Variable | Type | Values |
|----------|------|--------|
| `Wellbeing_Index` | Continuous | 1.0 – 10.0 |
| `Burnout_Risk` | Categorical | Low / Medium / High |

---

## 🔧 Pipeline — Step by Step Explanation

### Step 0 — Imports & Setup
All required libraries are imported once at the top:
- `scikit-learn` — all ML models, metrics, preprocessing
- `imbalanced-learn` — SMOTE for class balancing
- `matplotlib` / `seaborn` — visualizations
- `pandas` / `numpy` — data manipulation

---

### Step 1 — Load & Explore Data (EDA)
- Load CSV into a pandas DataFrame
- Check shape, data types, missing values
- View class distribution of `Burnout_Risk`
- View distribution of `Wellbeing_Index`

---

### Step 2 — Feature Engineering

#### 2a. Correlation Heatmap
- Encode all categorical columns to numbers
- Compute Pearson correlation between every feature pair
- Plot as a heatmap — values close to ±1 = highly correlated
- Detect redundant features (|r| > 0.85) that carry duplicate information

#### 2b. PCA — Principal Component Analysis
- Standardize all numeric features
- Fit PCA to find the directions of maximum variance
- Plot Scree plot (variance per component) and Cumulative variance curve
- Find how many components explain ≥95% of variance
- Helps understand dimensionality and feature redundancy

---

### Step 3 — Preprocessing

#### One-Hot Encoding
```
Gender: Male/Female  →  Gender_Male: 1 or 0
```
Converts categorical columns to binary columns. `drop_first=True` avoids the dummy variable trap.

#### Label Encoding
```
Burnout_Risk: High/Low/Medium  →  0 / 1 / 2
```
Converts classification target to integers for sklearn.

#### Train/Test Split
- 80% training, 20% testing
- `stratify=y` for classification — preserves class ratio in both splits
- `random_state=42` — reproducible results every run

---

## 🔴 Part A — Classification (Burnout_Risk)

### Step 1 — SMOTE (Synthetic Minority Over-sampling Technique)
The dataset has class imbalance — far more "Medium" samples than "Low" or "High". SMOTE fixes this:

```
Before SMOTE:  Low=64   Medium=7200  High=736
After SMOTE:   Low=7200 Medium=7200  High=7200
```

**How SMOTE works:**
1. Picks a minority class sample
2. Finds its K nearest neighbors (also minority class)
3. Creates a new synthetic point **between** them
4. Repeats until all classes are equal size

⚠️ SMOTE is applied **only to training data** — never to test data.

---

### Step 2 — Pie Charts (Before & After SMOTE)
Visual confirmation of class balancing — shows the distribution before and after SMOTE on the training set.

---

### Step 3 — StandardScaler
```
z = (x - mean) / std
```
Every feature is rescaled to mean=0, std=1.

- `fit_transform` on training data — **learns** mean and std from training
- `transform` only on test data — **applies** the same values (prevents data leakage)

**Why needed:** SVM and KNN are distance-based — a feature with range 0–10000 would dominate one with range 0–1.

---

### Step 4 — Six Classifiers with 10-Fold Cross-Validation

| Model | How It Works |
|-------|-------------|
| **Logistic Regression** | Sigmoid function on linear combination of features. Fast and interpretable |
| **Decision Tree** | Splits data by yes/no questions on feature values |
| **Random Forest** | 100s of trees on random data subsets — majority vote wins |
| **Gradient Boosting** | Sequential trees, each fixing errors of the previous |
| **SVM** | Finds maximum-margin hyperplane separating classes |
| **KNN** | Classifies by majority vote of K nearest neighbors |

**10-Fold Cross-Validation:**
```
Training data split into 10 equal folds:
[F1][F2][F3][F4][F5][F6][F7][F8][F9][F10]

Round 1:  Train on F2–F10, Test on F1  → Score
Round 2:  Train on F1,F3–F10, Test on F2 → Score
...
Round 10: Train on F1–F9, Test on F10 → Score

Final result: mean ± std of all 10 scores
```
Uses `StratifiedKFold` to maintain class ratio in each fold.

---

### Step 5 — Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | Correct / Total | % of all predictions that were right |
| **Precision** | TP / (TP + FP) | Of all predicted Highs, how many were actually High? |
| **Recall** | TP / (TP + FN) | Of all actual Highs, how many did we catch? |
| **F1-Score** | 2×(P×R)/(P+R) | Balance between precision and recall |
| **ROC-AUC** | Area under ROC curve | How well model separates classes at all thresholds |

`average='weighted'` — computes each metric per class then weights by class size.

**Confusion Matrix:** 3×3 grid showing predicted vs actual for all classes.

---

### Step 6 — GridSearchCV (Hyper-Parameter Tuning)
Tries every combination of hyperparameters:
```
Random Forest example:
n_estimators: [100, 200]      → 2 options
max_depth:    [None, 10, 20]  → 3 options
min_samples_split: [2, 5]     → 2 options

Total: 2 × 3 × 2 = 12 combinations
Each with 10-fold CV = 120 model fits
```
Picks the combination with the highest weighted F1-score.

---

### Step 7 — Feature Importance (Random Forest)
Random Forest tracks how much each feature reduces **Gini impurity** across all trees.
- Higher value = feature is more useful for prediction
- Scores sum to 1.0 across all features
- Top 15 features plotted as a horizontal bar chart

---

### Step 8 — Explainable AI (Permutation Importance)
Model-agnostic approach that works for **any model**:
1. Measure baseline F1 on test data
2. Randomly shuffle one feature (destroys its signal)
3. Measure how much F1 drops
4. Large drop = important feature
5. Repeat 20 times per feature and average for stability

```
Feature shuffled → F1 drops from 0.91 to 0.74 → importance = 0.17
Feature shuffled → F1 drops from 0.91 to 0.90 → importance = 0.01 (not important)
```

---

## 🔵 Part B — Regression (Wellbeing_Index)

### Key Difference
- No SMOTE — target is continuous, not categorical
- No `stratify` in train/test split
- Uses `KFold` instead of `StratifiedKFold`

### Six Regressors

| Model | How It Works |
|-------|-------------|
| **Linear Regression** | Fits line minimizing sum of squared errors |
| **Ridge** | Linear + L2 penalty, shrinks large coefficients |
| **Lasso** | Linear + L1 penalty, can zero out coefficients (feature selection) |
| **Decision Tree** | Predicts mean value of samples in each leaf node |
| **Random Forest** | Averages predictions from many decision trees |
| **Gradient Boosting** | Sequential trees fitting residual errors |

### Regression Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** | mean(\|actual - predicted\|) | Average absolute error — easy to interpret |
| **RMSE** | sqrt(mean((actual - predicted)²)) | Penalizes large errors more than MAE |
| **R²** | 1 - SS_res/SS_tot | % of variance explained (1.0 = perfect, 0 = no better than mean) |

### Actual vs Predicted Plot
Scatter plot where each dot = one test sample.
- X-axis = actual Wellbeing_Index value
- Y-axis = predicted value
- Red dashed diagonal = perfect prediction line
- Points close to diagonal = good model

---

## 🧠 Explainable AI Tab

Shows 4 charts side by side:

| Chart | Model | Metric |
|-------|-------|--------|
| Feature Importance — Classification | Random Forest | Gini impurity reduction |
| Feature Importance — Regression | Random Forest | Gini impurity reduction |
| Permutation Importance — Classification | Best classifier | F1 drop |
| Permutation Importance — Regression | Best regressor | R² drop |

---

## 🔮 Predict Tab

Input sliders for all 20 features → click **Predict** → get:
- **Burnout Risk** classification (Low / Medium / High) with color indicator
- **Wellbeing Index** predicted score
- **Probability bar chart** showing confidence for each class

---

## 🖥️ App Structure (Tabs)

```
┌─────────────────────────────────────────────────────────┐
│  🧠 Gen-Z Mental Wellness — ML Pipeline Dashboard       │
├──────────────┬──────────────────────────────────────────┤
│  SIDEBAR     │                                          │
│              │  📊 EDA & Feature Engineering            │
│ 📂 Upload    │     Dataset overview, correlation        │
│    CSV       │     heatmap, PCA analysis                │
│              │                                          │
│ Test Size    │  🔴 Classification                       │
│ CV Folds     │     SMOTE, 6 models, metrics,            │
│              │     confusion matrix, ROC-AUC            │
│ ✅ Model 1   │                                          │
│ ✅ Model 2   │  🔵 Regression                           │
│ ✅ Model 3   │     6 regressors, MAE/RMSE/R²,           │
│ ...          │     actual vs predicted                  │
│              │                                          │
│ ▶ Run        │  🧠 Explainable AI                       │
│  Pipeline    │     Feature importance,                  │
│              │     permutation importance               │
│              │                                          │
│              │  🔮 Predict                              │
│              │     Input sliders → live prediction      │
└──────────────┴──────────────────────────────────────────┘
```

---

## ⚙️ Installation & Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/genz-mental-wellness-app.git
cd genz-mental-wellness-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## ☁️ Deployment (Streamlit Cloud)

1. Push all 3 files to a **public** GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select your repo → set main file as `app.py`
5. Click **Deploy**

Live URL format:
```
https://YOUR_USERNAME-genz-mental-wellness-app.streamlit.app
```

---

## 📦 Dependencies

```txt
streamlit        — web app framework
numpy            — numerical computations
pandas           — data manipulation
matplotlib       — base plotting
seaborn          — statistical visualizations
scikit-learn     — ML models, metrics, preprocessing
imbalanced-learn — SMOTE for class balancing
```

---

## 🔑 Key Concepts Summary

| Concept | Purpose |
|---------|---------|
| SMOTE | Synthetic oversampling to fix class imbalance |
| StandardScaler | Normalize features to mean=0, std=1 |
| Train/Test Split | Simulate real-world unseen data evaluation |
| 10-Fold CV | Reliable performance estimate using all training data |
| GridSearchCV | Find the best hyperparameters automatically |
| F1-Score | Best metric for imbalanced classification |
| R² Score | Best single metric for regression quality |
| Feature Importance | Which features the model uses most (tree-based) |
| Permutation Importance | Model-agnostic feature contribution on test data |
| PCA | Reduce dimensions while preserving maximum variance |
| Correlation Heatmap | Identify redundant / multicollinear features |
| ROC-AUC | How well model separates classes across all thresholds |
| Confusion Matrix | Detailed breakdown of correct and incorrect predictions |

---