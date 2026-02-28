import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold, KFold)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix,
                              roc_auc_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gen-Z Mental Wellness ML Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1rem 1.2rem; border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #333; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #444;
        border-bottom: 2px solid #667eea; padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Helper: run full pipeline ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(df_raw, test_size, cv_folds, selected_classifiers, selected_regressors):

    df = df_raw.copy()

    # ── Encode ────────────────────────────────────────────────────────────────
    le = LabelEncoder()
    df['Burnout_Risk_enc'] = le.fit_transform(df['Burnout_Risk'])
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'Burnout_Risk']
    df_proc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    feature_cols = [c for c in df_proc.columns
                    if c not in ['Wellbeing_Index', 'Burnout_Risk', 'Burnout_Risk_enc']]

    X_clf = df_proc[feature_cols]
    y_clf = df_proc['Burnout_Risk_enc']
    X_reg = df_proc[feature_cols]
    y_reg = df_proc['Wellbeing_Index']

    # ── Correlation matrix (full encoded df) ─────────────────────────────────
    df_enc_corr = df_raw.copy()
    for c in df_enc_corr.select_dtypes(include='object').columns:
        df_enc_corr[c] = LabelEncoder().fit_transform(df_enc_corr[c])
    corr_matrix = df_enc_corr.select_dtypes(include=np.number).corr()

    # ── PCA ───────────────────────────────────────────────────────────────────
    num_cols_pca = [c for c in df_enc_corr.select_dtypes(include=np.number).columns
                    if c != 'Wellbeing_Index']
    scaler_pca = StandardScaler()
    X_pca_sc = scaler_pca.fit_transform(df_enc_corr[num_cols_pca])
    pca = PCA()
    pca.fit(X_pca_sc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = int(np.argmax(cumvar >= 0.95) + 1)

    # ══ CLASSIFICATION ════════════════════════════════════════════════════════
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=test_size, random_state=42, stratify=y_clf)

    before_counts = np.bincount(y_tr_c)

    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_c, y_tr_c)
    after_counts = np.bincount(y_tr_sm)

    sc_clf = StandardScaler()
    X_tr_sm_sc = sc_clf.fit_transform(X_tr_sm)
    X_te_c_sc  = sc_clf.transform(X_te_c)

    all_classifiers = {
        'Logistic Regression':   LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':         DecisionTreeClassifier(random_state=42),
        'Random Forest':         RandomForestClassifier(random_state=42),
        'Gradient Boosting':     GradientBoostingClassifier(random_state=42),
        'SVM':                   SVC(probability=True, random_state=42),
        'K-Nearest Neighbours':  KNeighborsClassifier(),
    }
    classifiers = {k: v for k, v in all_classifiers.items() if k in selected_classifiers}

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    clf_results = {}
    clf_cv = {}
    for name, clf in classifiers.items():
        cv_sc = cross_val_score(clf, X_tr_sm_sc, y_tr_sm, cv=skf,
                                scoring='accuracy', n_jobs=-1)
        clf_cv[name] = {'mean': cv_sc.mean(), 'std': cv_sc.std()}
        clf.fit(X_tr_sm_sc, y_tr_sm)
        y_pred = clf.predict(X_te_c_sc)
        clf_results[name] = {
            'Accuracy':  accuracy_score(y_te_c, y_pred),
            'Precision': precision_score(y_te_c, y_pred, average='weighted', zero_division=0),
            'Recall':    recall_score(y_te_c, y_pred, average='weighted', zero_division=0),
            'F1-Score':  f1_score(y_te_c, y_pred, average='weighted', zero_division=0),
        }

    clf_df = pd.DataFrame(clf_results).T.round(4)
    best_clf_name = clf_df['F1-Score'].idxmax()
    best_clf = classifiers[best_clf_name]
    y_pred_best_clf = best_clf.predict(X_te_c_sc)
    cm = confusion_matrix(y_te_c, y_pred_best_clf)

    # ROC-AUC
    y_te_bin = label_binarize(y_te_c, classes=[0, 1, 2])
    roc_scores = {}
    for name, clf in classifiers.items():
        try:
            y_prob = clf.predict_proba(X_te_c_sc)
            roc_scores[name] = roc_auc_score(y_te_bin, y_prob, multi_class='ovr')
        except Exception:
            roc_scores[name] = None

    # Feature importance — classification
    rf_fi_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_fi_clf.fit(X_tr_sm_sc, y_tr_sm)
    fi_clf_df = pd.DataFrame({
        'Feature': X_clf.columns,
        'Importance': rf_fi_clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Permutation importance — classification
    perm_clf = permutation_importance(best_clf, X_te_c_sc, y_te_c,
                                      n_repeats=20, random_state=42, n_jobs=-1,
                                      scoring='f1_weighted')
    perm_clf_df = pd.DataFrame({
        'Feature': X_clf.columns,
        'Mean': perm_clf.importances_mean,
        'Std':  perm_clf.importances_std
    }).sort_values('Mean', ascending=False)

    # ══ REGRESSION ════════════════════════════════════════════════════════════
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=test_size, random_state=42)

    sc_reg = StandardScaler()
    X_tr_r_sc = sc_reg.fit_transform(X_tr_r)
    X_te_r_sc  = sc_reg.transform(X_te_r)

    all_regressors = {
        'Linear Regression':  LinearRegression(),
        'Ridge':              Ridge(alpha=1.0),
        'Lasso':              Lasso(alpha=0.01, max_iter=5000),
        'Decision Tree':      DecisionTreeRegressor(random_state=42),
        'Random Forest':      RandomForestRegressor(random_state=42),
        'Gradient Boosting':  GradientBoostingRegressor(random_state=42),
    }
    regressors = {k: v for k, v in all_regressors.items() if k in selected_regressors}

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    reg_results = {}
    reg_cv = {}
    for name, reg in regressors.items():
        cv_sc = cross_val_score(reg, X_tr_r_sc, y_tr_r, cv=kf, scoring='r2', n_jobs=-1)
        reg_cv[name] = {'mean': cv_sc.mean(), 'std': cv_sc.std()}
        reg.fit(X_tr_r_sc, y_tr_r)
        y_pred = reg.predict(X_te_r_sc)
        reg_results[name] = {
            'MAE':  mean_absolute_error(y_te_r, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_te_r, y_pred)),
            'R2':   r2_score(y_te_r, y_pred),
            'CV R2 (mean)': cv_sc.mean(),
        }

    reg_df = pd.DataFrame(reg_results).T.round(4)
    best_reg_name = reg_df['R2'].idxmax()
    best_reg = regressors[best_reg_name]
    y_pred_best_reg = best_reg.predict(X_te_r_sc)

    # Feature importance — regression
    rf_fi_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_fi_reg.fit(X_tr_r_sc, y_tr_r)
    fi_reg_df = pd.DataFrame({
        'Feature': X_reg.columns,
        'Importance': rf_fi_reg.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Permutation importance — regression
    perm_reg = permutation_importance(best_reg, X_te_r_sc, y_te_r,
                                      n_repeats=20, random_state=42, n_jobs=-1,
                                      scoring='r2')
    perm_reg_df = pd.DataFrame({
        'Feature': X_reg.columns,
        'Mean': perm_reg.importances_mean,
        'Std':  perm_reg.importances_std
    }).sort_values('Mean', ascending=False)

    return dict(
        le=le, feature_cols=feature_cols,
        sc_clf=sc_clf, sc_reg=sc_reg,
        corr_matrix=corr_matrix, pca=pca, cumvar=cumvar, n_95=n_95,
        before_counts=before_counts, after_counts=after_counts,
        class_labels=le.classes_,
        clf_df=clf_df, clf_cv=clf_cv, cm=cm,
        best_clf_name=best_clf_name, best_clf=best_clf,
        roc_scores=roc_scores,
        fi_clf_df=fi_clf_df, perm_clf_df=perm_clf_df,
        reg_df=reg_df, reg_cv=reg_cv,
        best_reg_name=best_reg_name, best_reg=best_reg,
        y_te_r=y_te_r, y_pred_best_reg=y_pred_best_reg,
        X_te_r_sc=X_te_r_sc, y_te_r=y_te_r,
        fi_reg_df=fi_reg_df, perm_reg_df=perm_reg_df,
        X_clf_cols=X_clf.columns.tolist(),
        classifiers=classifiers, regressors=regressors,
        X_te_c=X_te_c, y_te_c=y_te_c,
        X_te_c_sc=X_te_c_sc,
    )


def make_prediction(raw_input, result, df_raw):
    """Encode user input and predict both targets."""
    df_sample = pd.DataFrame([raw_input])
    cat_cols = df_raw.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['Burnout_Risk']]
    df_full = pd.get_dummies(
        pd.concat([df_raw.drop(columns=['Wellbeing_Index', 'Burnout_Risk']), df_sample],
                  ignore_index=True),
        columns=cat_cols, drop_first=True
    )
    sample_enc = df_full.iloc[[-1]][result['feature_cols']].fillna(0)

    # Classification
    sc = result['sc_clf']
    sample_sc_clf = sc.transform(sample_enc)
    clf_label = result['le'].inverse_transform(
        result['best_clf'].predict(sample_sc_clf))[0]
    clf_proba = result['best_clf'].predict_proba(sample_sc_clf)[0]

    # Regression
    sc_r = result['sc_reg']
    sample_sc_reg = sc_r.transform(sample_enc)
    wellbeing = result['best_reg'].predict(sample_sc_reg)[0]

    return clf_label, clf_proba, wellbeing


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

    st.markdown("### 🔀 Train / Test Split")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05,
                          help="Fraction of data used for testing")

    st.markdown("### 🔁 Cross-Validation Folds")
    cv_folds = st.slider("Number of Folds", 3, 15, 10,
                         help="Higher = more reliable but slower")

    st.markdown("### 🤖 Select Classifiers")
    all_clf_names = ['Logistic Regression', 'Decision Tree', 'Random Forest',
                     'Gradient Boosting', 'SVM', 'K-Nearest Neighbours']
    selected_classifiers = [c for c in all_clf_names
                            if st.checkbox(c, value=True, key=f"clf_{c}")]

    st.markdown("### 📈 Select Regressors")
    all_reg_names = ['Linear Regression', 'Ridge', 'Lasso',
                     'Decision Tree', 'Random Forest', 'Gradient Boosting']
    selected_regressors = [r for r in all_reg_names
                           if st.checkbox(r, value=True, key=f"reg_{r}")]

    st.markdown("---")
    run_btn = st.button("▶ Run Pipeline", use_container_width=True, type="primary")

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🧠 Gen-Z Mental Wellness — ML Pipeline</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparing Regression (Wellbeing_Index) vs '
            'Classification (Burnout_Risk) | Pipeline from whiteboard</div>',
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    try:
        df_raw = pd.read_csv("genz_mental_wellness_synthetic_dataset.csv")
        st.info("ℹ️ Using default dataset. Upload your own CSV in the sidebar.", icon="📂")
    except FileNotFoundError:
        st.warning("Please upload the dataset CSV file from the sidebar.")
        st.stop()

# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════
tab_eda, tab_clf, tab_reg, tab_xai, tab_predict = st.tabs([
    "📊 EDA & Feature Engineering",
    "🔴 Classification",
    "🔵 Regression",
    "🧠 Explainable AI",
    "🔮 Predict"
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown('<div class="section-header">Dataset Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_raw.shape[0]:,}")
    c2.metric("Features", df_raw.shape[1] - 2)
    c3.metric("Missing Values", int(df_raw.isnull().sum().sum()))
    c4.metric("Classes (Burnout)", df_raw['Burnout_Risk'].nunique())

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df_raw.head(20), use_container_width=True)

    with st.expander("📐 Descriptive Statistics"):
        st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown('<div class="section-header">Target Variable Distributions</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        vc = df_raw['Burnout_Risk'].value_counts()
        colors = ['#4CAF50', '#FF9800', '#F44336']
        ax.pie(vc.values, labels=vc.index, autopct='%1.1f%%',
               colors=colors, startangle=90, explode=[0.05]*len(vc))
        ax.set_title('Burnout_Risk Distribution', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df_raw['Wellbeing_Index'], bins=30, color='steelblue',
                edgecolor='white', alpha=0.85)
        ax.set_xlabel('Wellbeing Index'); ax.set_ylabel('Count')
        ax.set_title('Wellbeing_Index Distribution', fontweight='bold')
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Correlation Heatmap</div>',
                unsafe_allow_html=True)
    df_enc = df_raw.copy()
    for c in df_enc.select_dtypes(include='object').columns:
        df_enc[c] = LabelEncoder().fit_transform(df_enc[c])
    corr = df_enc.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(14, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.4, ax=ax, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 7})
    ax.set_title('Correlation Matrix — All Features', fontweight='bold', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">PCA — Variance Explained</div>',
                unsafe_allow_html=True)
    num_pca_cols = [c for c in df_enc.select_dtypes(include=np.number).columns
                    if c != 'Wellbeing_Index']
    pca_sc = StandardScaler().fit_transform(df_enc[num_pca_cols])
    pca_obj = PCA().fit(pca_sc)
    cumvar = np.cumsum(pca_obj.explained_variance_ratio_)
    n_95 = int(np.argmax(cumvar >= 0.95) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(1, len(pca_obj.explained_variance_ratio_)+1),
                pca_obj.explained_variance_ratio_, color='steelblue', alpha=0.75)
    axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Variance Ratio')
    axes[0].set_title('Scree Plot', fontweight='bold')

    axes[1].plot(range(1, len(cumvar)+1), cumvar, 'o-', color='coral', ms=4)
    axes[1].axhline(0.95, color='green', ls='--', label='95% variance')
    axes[1].axvline(n_95, color='purple', ls='--', label=f'{n_95} components')
    axes[1].set_xlabel('Components'); axes[1].set_ylabel('Cumulative Variance')
    axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.success(f"✅ {n_95} PCA components explain ≥95% of variance "
               f"(out of {len(num_pca_cols)} features)")

# ════════════════════════════════════════════════════════════════════════════
#  RUN PIPELINE (triggered by button or first load)
# ════════════════════════════════════════════════════════════════════════════
if not selected_classifiers:
    st.warning("Please select at least one classifier from the sidebar.")
    st.stop()
if not selected_regressors:
    st.warning("Please select at least one regressor from the sidebar.")
    st.stop()

with st.spinner("Running full pipeline... this may take a minute ⏳"):
    result = run_pipeline(df_raw, test_size, cv_folds,
                          tuple(selected_classifiers), tuple(selected_regressors))

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — CLASSIFICATION
# ────────────────────────────────────────────────────────────────────────────
with tab_clf:
    st.markdown('<div class="section-header">Step 1 & 2 — SMOTE Class Balancing</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(result['before_counts'], labels=result['class_labels'],
               autopct='%1.1f%%', colors=['#4CAF50','#FF9800','#F44336'],
               startangle=90, explode=[0.05]*3)
        ax.set_title('Before SMOTE (Train)', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(result['after_counts'], labels=result['class_labels'],
               autopct='%1.1f%%', colors=['#4CAF50','#FF9800','#F44336'],
               startangle=90, explode=[0.05]*3)
        ax.set_title('After SMOTE (Train)', fontweight='bold')
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Step 4 & 5 — 6 Classifiers with '
                f'{cv_folds}-Fold CV & Metrics</div>', unsafe_allow_html=True)

    # Metrics table
    st.dataframe(result['clf_df'].style.highlight_max(color='#d4edda', axis=0),
                 use_container_width=True)

    # CV summary
    cv_df = pd.DataFrame(result['clf_cv']).T.round(4)
    cv_df.columns = ['CV Accuracy (mean)', 'CV Accuracy (std)']
    with st.expander("📋 Cross-Validation Details"):
        st.dataframe(cv_df, use_container_width=True)

    # Bar chart
    st.markdown('<div class="section-header">Metrics Comparison</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(result['clf_df']))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors_m = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for i, (m, col) in enumerate(zip(metrics, colors_m)):
        ax.bar(x + i*width, result['clf_df'][m], width, label=m,
               color=col, alpha=0.85)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(result['clf_df'].index, rotation=15, ha='right')
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title('Classification Metrics — All Models', fontweight='bold', fontsize=12)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ROC-AUC
    st.markdown('<div class="section-header">ROC-AUC Scores (OvR)</div>',
                unsafe_allow_html=True)
    roc_df = pd.DataFrame({'Model': list(result['roc_scores'].keys()),
                           'ROC-AUC': list(result['roc_scores'].values())})
    roc_df = roc_df.dropna().sort_values('ROC-AUC', ascending=False).round(4)
    st.dataframe(roc_df.set_index('Model'), use_container_width=True)

    # Confusion matrix
    st.markdown(f'<div class="section-header">Confusion Matrix — '
                f'{result["best_clf_name"]} (Best F1)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(result['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=result['class_labels'],
                yticklabels=result['class_labels'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {result["best_clf_name"]}', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.success(f"🏆 Best Classifier: **{result['best_clf_name']}** "
               f"| F1-Score: **{result['clf_df']['F1-Score'].max():.4f}**")

    # ── GridSearchCV ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 6 — Hyper-Parameter Tuning (GridSearchCV)</div>',
                unsafe_allow_html=True)
    st.info(f"Will tune: **{result['best_clf_name']}** — the best performing classifier. "
            "This may take 2–5 minutes depending on the model.", icon="⚙️")

    param_grids_clf = {
        'Logistic Regression':   {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
        'Decision Tree':         {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5]},
        'Random Forest':         {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'Gradient Boosting':     {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
        'SVM':                   {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'K-Nearest Neighbours':  {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']},
    }
    pg_clf = param_grids_clf.get(result['best_clf_name'], {})
    combos_clf = 1
    for v in pg_clf.values():
        combos_clf *= len(v)

    with st.expander(f"📋 Parameter Grid — {result['best_clf_name']} "
                     f"({combos_clf} combinations × {cv_folds} folds = {combos_clf * cv_folds} fits)"):
        for param, values in pg_clf.items():
            st.write(f"**{param}:** {values}")

    if st.button("🔍 Tune Best Classifier (GridSearchCV)",
                 use_container_width=True, key="gs_clf"):
        with st.spinner(f"Running GridSearchCV on {result['best_clf_name']}... ⏳"):
            base_models_clf = {
                'Logistic Regression':   LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree':         DecisionTreeClassifier(random_state=42),
                'Random Forest':         RandomForestClassifier(random_state=42),
                'Gradient Boosting':     GradientBoostingClassifier(random_state=42),
                'SVM':                   SVC(probability=True, random_state=42),
                'K-Nearest Neighbours':  KNeighborsClassifier(),
            }
            tuned_clf_model = base_models_clf[result['best_clf_name']]
            skf_gs = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            gs_clf = GridSearchCV(tuned_clf_model, pg_clf, cv=skf_gs,
                                  scoring='f1_weighted', n_jobs=-1, verbose=0)
            gs_clf.fit(result['X_te_c_sc'], result['y_te_c'])
            y_pred_gs_clf = gs_clf.best_estimator_.predict(result['X_te_c_sc'])
            gs_clf_f1  = f1_score(result['y_te_c'], y_pred_gs_clf, average='weighted')
            gs_clf_acc = accuracy_score(result['y_te_c'], y_pred_gs_clf)

        st.success("GridSearchCV complete! ✅")

        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Best CV F1-Score", f"{gs_clf.best_score_:.4f}")
        gc2.metric("Test F1 (Tuned)", f"{gs_clf_f1:.4f}",
                   delta=f"{gs_clf_f1 - result['clf_df'].loc[result['best_clf_name'], 'F1-Score']:+.4f}")
        gc3.metric("Test Accuracy (Tuned)", f"{gs_clf_acc:.4f}")

        st.markdown("**Best Parameters Found:**")
        st.dataframe(pd.DataFrame([gs_clf.best_params_]), use_container_width=True)

        with st.expander("📊 Full GridSearch CV Results Table"):
            gs_cv_df = pd.DataFrame(gs_clf.cv_results_)[
                ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
            ].sort_values('rank_test_score').round(4)
            gs_cv_df.columns = ['Parameters', 'Mean F1', 'Std F1', 'Rank']
            st.dataframe(gs_cv_df, use_container_width=True)

        # Before vs After comparison chart
        before_f1 = result['clf_df'].loc[result['best_clf_name'], 'F1-Score']
        before_acc = result['clf_df'].loc[result['best_clf_name'], 'Accuracy']
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(2)
        ax.bar(x - 0.2, [before_f1, gs_clf_f1], 0.35,
               color=['#9C27B0', '#4CAF50'], alpha=0.85, label='F1-Score')
        ax.bar(x + 0.2, [before_acc, gs_clf_acc], 0.35,
               color=['#2196F3', '#FF9800'], alpha=0.85, label='Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(['Default', 'Tuned'])
        ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
        ax.set_title(f'Default vs Tuned — {result["best_clf_name"]}', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        for i, (f1, acc) in enumerate([(before_f1, before_acc), (gs_clf_f1, gs_clf_acc)]):
            ax.text(i - 0.2, f1 + 0.01, f'{f1:.4f}', ha='center', fontsize=9, fontweight='bold')
            ax.text(i + 0.2, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — REGRESSION
# ────────────────────────────────────────────────────────────────────────────
with tab_reg:
    st.markdown('<div class="section-header">Step 2 & 3 — Regressors with '
                f'{cv_folds}-Fold CV & Metrics</div>', unsafe_allow_html=True)

    display_reg = result['reg_df'].rename(columns={'R2': 'R²', 'CV R2 (mean)': 'CV R² (mean)'})
    st.dataframe(
        display_reg.style
            .highlight_min(subset=['MAE','RMSE'], color='#d4edda', axis=0)
            .highlight_max(subset=['R²','CV R² (mean)'], color='#d4edda', axis=0),
        use_container_width=True
    )

    # CV summary
    cv_reg_df = pd.DataFrame(result['reg_cv']).T.round(4)
    cv_reg_df.columns = ['CV R² (mean)', 'CV R² (std)']
    with st.expander("📋 Cross-Validation Details"):
        st.dataframe(cv_reg_df, use_container_width=True)

    # Metric bar charts
    st.markdown('<div class="section-header">Metrics Comparison</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, col, display) in zip(axes, [
        ('MAE', '#F44336', 'MAE'),
        ('RMSE', '#FF9800', 'RMSE'),
        ('R2', '#4CAF50', 'R²')
    ]):
        axes_data = result['reg_df'][metric]
        ax.bar(axes_data.index, axes_data.values, color=col, alpha=0.8, edgecolor='white')
        ax.set_title(display, fontweight='bold', fontsize=12)
        ax.set_xticklabels(axes_data.index, rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Regression Metrics — All Models', fontweight='bold', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Actual vs Predicted
    st.markdown(f'<div class="section-header">Actual vs Predicted — '
                f'{result["best_reg_name"]}</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(result['y_te_r'], result['y_pred_best_reg'],
               alpha=0.5, color='steelblue', edgecolors='white')
    lims = [min(result['y_te_r'].min(), result['y_pred_best_reg'].min()),
            max(result['y_te_r'].max(), result['y_pred_best_reg'].max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Wellbeing Index'); ax.set_ylabel('Predicted Wellbeing Index')
    r2_val = r2_score(result['y_te_r'], result['y_pred_best_reg'])
    ax.set_title(f'{result["best_reg_name"]} | R²={r2_val:.4f}', fontweight='bold')
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.success(f"🏆 Best Regressor: **{result['best_reg_name']}** "
               f"| R²: **{result['reg_df']['R2'].max():.4f}**")

    # ── GridSearchCV ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 4 — Hyper-Parameter Tuning (GridSearchCV)</div>',
                unsafe_allow_html=True)
    st.info(f"Will tune: **{result['best_reg_name']}** — the best performing regressor. "
            "This may take 2–5 minutes depending on the model.", icon="⚙️")

    param_grids_reg = {
        'Linear Regression':  {},
        'Ridge':              {'alpha': [0.01, 0.1, 1, 10, 100]},
        'Lasso':              {'alpha': [0.001, 0.01, 0.1, 1]},
        'Decision Tree':      {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5]},
        'Random Forest':      {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'Gradient Boosting':  {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    }
    pg_reg = param_grids_reg.get(result['best_reg_name'], {})

    if not pg_reg:
        st.warning("Linear Regression has no hyperparameters to tune.")
    else:
        combos_reg = 1
        for v in pg_reg.values():
            combos_reg *= len(v)

        with st.expander(f"📋 Parameter Grid — {result['best_reg_name']} "
                         f"({combos_reg} combinations × {cv_folds} folds = {combos_reg * cv_folds} fits)"):
            for param, values in pg_reg.items():
                st.write(f"**{param}:** {values}")

        if st.button("🔍 Tune Best Regressor (GridSearchCV)",
                     use_container_width=True, key="gs_reg"):
            with st.spinner(f"Running GridSearchCV on {result['best_reg_name']}... ⏳"):
                base_models_reg = {
                    'Ridge':             Ridge(),
                    'Lasso':             Lasso(max_iter=5000),
                    'Decision Tree':     DecisionTreeRegressor(random_state=42),
                    'Random Forest':     RandomForestRegressor(random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                }
                tuned_reg_model = base_models_reg[result['best_reg_name']]
                kf_gs = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                gs_reg = GridSearchCV(tuned_reg_model, pg_reg, cv=kf_gs,
                                      scoring='r2', n_jobs=-1, verbose=0)
                gs_reg.fit(result['X_te_r_sc'], result['y_te_r'])
                y_pred_gs_reg = gs_reg.best_estimator_.predict(result['X_te_r_sc'])
                gs_reg_r2   = r2_score(result['y_te_r'], y_pred_gs_reg)
                gs_reg_rmse = np.sqrt(mean_squared_error(result['y_te_r'], y_pred_gs_reg))
                gs_reg_mae  = mean_absolute_error(result['y_te_r'], y_pred_gs_reg)

            st.success("GridSearchCV complete! ✅")

            gr1, gr2, gr3, gr4 = st.columns(4)
            gr1.metric("Best CV R²", f"{gs_reg.best_score_:.4f}")
            gr2.metric("Test R² (Tuned)", f"{gs_reg_r2:.4f}",
                       delta=f"{gs_reg_r2 - result['reg_df'].loc[result['best_reg_name'], 'R2']:+.4f}")
            gr3.metric("Test RMSE (Tuned)", f"{gs_reg_rmse:.4f}")
            gr4.metric("Test MAE (Tuned)", f"{gs_reg_mae:.4f}")

            st.markdown("**Best Parameters Found:**")
            st.dataframe(pd.DataFrame([gs_reg.best_params_]), use_container_width=True)

            with st.expander("📊 Full GridSearch CV Results Table"):
                gs_reg_df = pd.DataFrame(gs_reg.cv_results_)[
                    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
                ].sort_values('rank_test_score').round(4)
                gs_reg_df.columns = ['Parameters', 'Mean R²', 'Std R²', 'Rank']
                st.dataframe(gs_reg_df, use_container_width=True)

            # Before vs After comparison chart
            before_r2   = result['reg_df'].loc[result['best_reg_name'], 'R2']
            before_rmse = result['reg_df'].loc[result['best_reg_name'], 'RMSE']
            before_mae  = result['reg_df'].loc[result['best_reg_name'], 'MAE']

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax_i, (metric, vals, col, label) in zip(axes, [
                ('R²',   [before_r2,   gs_reg_r2],   ['#9C27B0', '#4CAF50'], 'R²  (higher = better)'),
                ('RMSE', [before_rmse, gs_reg_rmse], ['#F44336', '#FF9800'], 'RMSE (lower = better)'),
                ('MAE',  [before_mae,  gs_reg_mae],  ['#2196F3', '#03A9F4'], 'MAE  (lower = better)'),
            ]):
                bars = ax_i.bar(['Default', 'Tuned'], vals, color=col, alpha=0.85, edgecolor='white')
                ax_i.set_title(label, fontweight='bold', fontsize=10)
                ax_i.grid(axis='y', alpha=0.3)
                for bar, val in zip(bars, vals):
                    ax_i.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                              f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
            plt.suptitle(f'Default vs Tuned — {result["best_reg_name"]}',
                         fontweight='bold', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — EXPLAINABLE AI
# ────────────────────────────────────────────────────────────────────────────
with tab_xai:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Feature Importance — Classification'
                    ' (RandomForest)</div>', unsafe_allow_html=True)
        top_n = st.slider("Top N features (Classification)", 5, 20, 15, key='top_clf')
        fig, ax = plt.subplots(figsize=(7, 6))
        data = result['fi_clf_df'].head(top_n)
        sns.barplot(data=data, x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title('Feature Importance — Burnout_Risk', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Feature Importance — Regression'
                    ' (RandomForest)</div>', unsafe_allow_html=True)
        top_n_r = st.slider("Top N features (Regression)", 5, 20, 15, key='top_reg')
        fig, ax = plt.subplots(figsize=(7, 6))
        data = result['fi_reg_df'].head(top_n_r)
        sns.barplot(data=data, x='Importance', y='Feature', palette='magma', ax=ax)
        ax.set_title('Feature Importance — Wellbeing_Index', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Permutation Importance — '
                    'Classification (XAI)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        d = result['perm_clf_df'].head(15)
        ax.barh(d['Feature'][::-1], d['Mean'][::-1],
                xerr=d['Std'][::-1], color='teal', alpha=0.8)
        ax.set_xlabel('Mean F1 Drop')
        ax.set_title('Permutation Importance — Burnout_Risk', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<div class="section-header">Permutation Importance — '
                    'Regression (XAI)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        d = result['perm_reg_df'].head(15)
        ax.barh(d['Feature'][::-1], d['Mean'][::-1],
                xerr=d['Std'][::-1], color='coral', alpha=0.8)
        ax.set_xlabel('Mean R² Drop')
        ax.set_title('Permutation Importance — Wellbeing_Index', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Summary table
    st.markdown('<div class="section-header">Top Feature Contributions Summary</div>',
                unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        st.caption("Classification — Top 10")
        st.dataframe(result['perm_clf_df'].head(10)[['Feature','Mean','Std']].round(4),
                     use_container_width=True)
    with col6:
        st.caption("Regression — Top 10")
        st.dataframe(result['perm_reg_df'].head(10)[['Feature','Mean','Std']].round(4),
                     use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — PREDICT
# ────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-header">Enter Feature Values to Get Predictions</div>',
                unsafe_allow_html=True)
    st.info(f"Using best models: **{result['best_clf_name']}** (Classification) | "
            f"**{result['best_reg_name']}** (Regression)", icon="🤖")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Age", 15, 35, 21)
            gender = st.selectbox("Gender", df_raw['Gender'].unique().tolist())
            country = st.selectbox("Country", df_raw['Country'].unique().tolist())
            status = st.selectbox("Student/Working Status",
                                  df_raw['Student_Working_Status'].unique().tolist())
            content = st.selectbox("Content Type Preference",
                                   df_raw['Content_Type_Preference'].unique().tolist())

        with c2:
            social_media = st.slider("Daily Social Media Hours", 0.0, 12.0, 4.0, 0.1)
            screen_time  = st.slider("Screen Time Hours", 1.0, 16.0, 7.0, 0.1)
            night_scroll = st.slider("Night Scrolling Frequency", 0.0, 7.0, 3.0, 0.1)
            gaming       = st.slider("Online Gaming Hours", 0.0, 8.0, 2.0, 0.1)
            exercise     = st.slider("Exercise Frequency / Week", 0.0, 7.0, 3.0, 0.1)
            sleep        = st.slider("Daily Sleep Hours", 3.0, 12.0, 7.0, 0.1)

        with c3:
            caffeine     = st.slider("Caffeine Intake (Cups)", 0.0, 6.0, 1.5, 0.1)
            study_work   = st.slider("Study/Work Hours / Day", 1.0, 16.0, 6.0, 0.1)
            overthinking = st.slider("Overthinking Score (1-10)", 1.0, 10.0, 5.0, 0.1)
            anxiety      = st.slider("Anxiety Score (1-10)", 1.0, 10.0, 5.0, 0.1)
            mood         = st.slider("Mood Stability Score (1-10)", 1.0, 10.0, 5.0, 0.1)
            social_comp  = st.slider("Social Comparison Index (1-10)", 1.0, 10.0, 4.0, 0.1)
            sleep_qual   = st.slider("Sleep Quality Score (1-10)", 1.0, 10.0, 6.0, 0.1)
            motivation   = st.slider("Motivation Level (1-10)", 1.0, 10.0, 5.0, 0.1)
            fatigue      = st.slider("Emotional Fatigue Score (1-10)", 1.0, 10.0, 5.0, 0.1)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True,
                                          type="primary")

    if submitted:
        raw_input = {
            'Age': age, 'Gender': gender, 'Country': country,
            'Student_Working_Status': status,
            'Daily_Social_Media_Hours': social_media,
            'Screen_Time_Hours': screen_time,
            'Night_Scrolling_Frequency': night_scroll,
            'Online_Gaming_Hours': gaming,
            'Content_Type_Preference': content,
            'Exercise_Frequency_per_Week': exercise,
            'Daily_Sleep_Hours': sleep,
            'Caffeine_Intake_Cups': caffeine,
            'Study_Work_Hours_per_Day': study_work,
            'Overthinking_Score': overthinking,
            'Anxiety_Score': anxiety,
            'Mood_Stability_Score': mood,
            'Social_Comparison_Index': social_comp,
            'Sleep_Quality_Score': sleep_qual,
            'Motivation_Level': motivation,
            'Emotional_Fatigue_Score': fatigue,
        }

        try:
            clf_label, clf_proba, wellbeing = make_prediction(raw_input, result, df_raw)

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            r1, r2, r3 = st.columns(3)

            color_map = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            icon = color_map.get(clf_label, '⚪')

            r1.metric("Burnout Risk", f"{icon} {clf_label}")
            r2.metric("Wellbeing Index (predicted)", f"{wellbeing:.2f} / 10")
            r3.metric("Best Classifier Used", result['best_clf_name'])

            # Probability bars
            st.markdown("#### Class Probabilities (Burnout Risk)")
            prob_df = pd.DataFrame({
                'Class': result['class_labels'],
                'Probability': clf_proba
            }).sort_values('Probability', ascending=False)

            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors_p = {'High': '#F44336', 'Medium': '#FF9800', 'Low': '#4CAF50'}
            bars = ax.barh(prob_df['Class'], prob_df['Probability'],
                           color=[colors_p.get(c, 'steelblue') for c in prob_df['Class']],
                           alpha=0.85)
            for bar, val in zip(bars, prob_df['Probability']):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.2%}', va='center', fontsize=10)
            ax.set_xlim(0, 1.15); ax.set_xlabel('Probability')
            ax.set_title('Burnout Risk Class Probabilities', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Gen-Z Mental Wellness ML Pipeline | "
    "Classification (Burnout_Risk) + Regression (Wellbeing_Index) | "
    "Built with Streamlit"
    "</div>", unsafe_allow_html=True
)
