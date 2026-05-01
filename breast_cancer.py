import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, recall_score

# ==========================================
# 0. SETUP & DATA LOADING
# ==========================================
st.set_page_config(page_title="Breast Cancer CRISP-DM Portal", layout="wide")
sns.set_theme(style="whitegrid", palette="viridis")

@st.cache_data
def get_dataset():
    """Phase 2: Data Acquisition & Initial Encoding"""
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    y_encoded = y.iloc[:, 0].map({'M': 1, 'B': 0})
    df = pd.concat([X, y_encoded], axis=1)
    df.rename(columns={df.columns[-1]: 'Diagnosis'}, inplace=True)
    return df, X.columns.tolist()

df_raw, feature_list = get_dataset()

# ==========================================
# SIDEBAR: PERSISTENT FILTERS (PHASE 6)
# ==========================================
st.sidebar.title("🛠️ CRISP-DM Control Panel")
st.sidebar.markdown("### Phase 4: Model Settings")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2)
n_trees = st.sidebar.number_input("Random Forest Estimators", 10, 500, 100)

st.sidebar.markdown("---")
st.sidebar.markdown("### User Input for Prediction")
user_inputs = {}
for feat in feature_list[:5]: # Using top 5 features for prediction demo
    user_inputs[feat] = st.sidebar.number_input(f"Input {feat}", value=float(df_raw[feat].mean()))

# ==========================================
# PHASE 1: BUSINESS UNDERSTANDING
# ==========================================
st.title("🔬 Breast Cancer Diagnostic Framework")
with st.expander("📌 CRISP-DM Phase 1: Business Understanding", expanded=True):
    st.write("**Objective:** Develop a robust diagnostic tool to classify tumors as Malignant (1) or Benign (0).")
    st.write("**Clinical Priority:** Minimize False Negatives (missed diagnoses) and ensure model explainability.")
    st.info("Statistical Significance Level established at $\\alpha = 0.01$.")

# ==========================================
# PHASE 2: DATA UNDERSTANDING
# ==========================================
phase2_tab1, phase2_tab2, phase2_tab3 = st.tabs(["Data Integrity", "Univariate Analysis", "Bivariate Significance"])

with phase2_tab1:
    st.header("Data Quality Audit")
    fig_miss, ax_miss = plt.subplots(figsize=(10, 2))
    sns.heatmap(df_raw.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax_miss)
    st.pyplot(fig_miss)
    st.caption("Missingness Map: Yellow indicates NaN values.")

with phase2_tab2:
    st.header("Distribution & Outliers")
    feat_to_plot = st.selectbox("Select Feature to Examine", feature_list)
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(df_raw[feat_to_plot], kde=True, ax=ax_hist)
        st.pyplot(fig_hist)
    with col_b:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df_raw, x='Diagnosis', y=feat_to_plot, palette='coolwarm', ax=ax_box)
        st.pyplot(fig_box)

with phase2_tab3:
    st.header("Statistical Tally")
    alpha = 0.01
    sig_results = []
    significant_features = []

    for col in feature_list:
        # Normality and Test Selection[cite: 1, 2]
        _, p_norm = stats.shapiro(df_raw[col])
        is_normal = p_norm > alpha
        mal = df_raw[df_raw['Diagnosis'] == 1][col]
        ben = df_raw[df_raw['Diagnosis'] == 0][col]
        
        _, p_val = stats.ttest_ind(mal, ben) if is_normal else stats.mannwhitneyu(mal, ben)
        
        if p_val < alpha:
            significant_features.append(col)
            
        sig_results.append({"Feature": col, "P-Value": f"{p_val:.2e}", "Significant": p_val < alpha})

    st.dataframe(pd.DataFrame(sig_results))

# ==========================================
# PHASE 3: DATA PREPARATION
# ==========================================
st.divider()
st.header("⚙️ CRISP-DM Phase 3: Data Preparation")
col_p1, col_p2 = st.columns(2)

with col_p1:
    st.write("**Feature Selection:** Only features with $p < 0.01$ are retained[cite: 1].")
    st.write(f"Retained {len(significant_features)} significant features.")

with col_p2:
    st.write("**Feature Scaling:** Applying Standard Scaler for model consistency.")
    X = df_raw[significant_features]
    y = df_raw['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# ==========================================
# PHASE 4 & 5: MODELING & EVALUATION
# ==========================================
st.divider()
st.header("🤖 Phase 4 & 5: Modeling & Evaluation")

# Train Model
model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

col_m1, col_m2 = st.columns([1, 2])

with col_m1:
    st.subheader("Metrics")
    st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    st.metric("Sensitivity (Recall)", f"{recall_score(y_test, y_pred):.2%}")
    st.help("Sensitivity is prioritized to minimize missed Malignant cases[cite: 1].")

with col_m2:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

# ==========================================
# PHASE 6: DEPLOYMENT (Inference)
# ==========================================
st.divider()
st.header("🚀 Phase 6: Deployment (Clinical Inference)")
st.write("Enter values in the sidebar to simulate a live clinical diagnosis.")

# Prepare user data for prediction
input_df = pd.DataFrame([user_inputs])
# Fill missing significant features with means for the demo
for col in significant_features:
    if col not in input_df.columns:
        input_df[col] = df_raw[col].mean()

input_scaled = scaler.transform(input_df[significant_features])
prediction = model.predict(input_scaled)
prob = model.predict_proba(input_scaled)

if prediction[0] == 1:
    st.error(f"Prediction: MALIGNANT (Confidence: {prob[0][1]:.2%})")
else:
    st.success(f"Prediction: BENIGN (Confidence: {prob[0][0]:.2%})")
