import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from streamlit_shap import st_shap
from ucimlrepo import fetch_ucirepo
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# ==============================================================================
# CONFIGURATION & DATA LOADING (PHASE 2)
# ==============================================================================
st.set_page_config(page_title="Breast Cancer Decision Support", layout="wide")
sns.set_theme(style="whitegrid", palette="viridis")

@st.cache_data
def load_data():
    """Phase 2: Data Acquisition and Initial Encoding[cite: 1, 2]."""
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    # Clinical Encoding: Malignant = 1, Benign = 0[cite: 1]
    y_encoded = y.iloc[:, 0].map({'M': 1, 'B': 0})
    df = pd.concat([X, y_encoded], axis=1)
    df.rename(columns={df.columns[-1]: 'Diagnosis'}, inplace=True)
    return df, X.columns.tolist()

df_raw, all_features = load_data()

# ==========================================
# PHASE 1: BUSINESS UNDERSTANDING[cite: 1]
# ==========================================
st.title("🔬 Breast Cancer Diagnostic Analytics Portal")
with st.container():
    st.header("Phase 1: Business Understanding")
    st.markdown("""
    **Objective:** Develop an explainable diagnostic tool to classify tumors as Malignant or Benign based on FNA images[cite: 1].
    
    **Clinical Mandates:**
    *   **Statistical Rigor:** Alpha level of $\alpha = 0.01$ to ensure feature robustness[cite: 1, 2].
    *   **Sensitivity:** Primary focus on minimizing False Negatives (missed diagnoses)[cite: 1].
    *   **Explainability:** Doctors must understand which nuclear features drive a high-risk prediction[cite: 1].
    """)

# ==========================================
# PHASE 2: DATA UNDERSTANDING (EDA)[cite: 1, 2]
# ==========================================
st.divider()
st.header("📊 Phase 2: Data Understanding")
tab_integrity, tab_dist, tab_stats = st.tabs(["Data Integrity", "Univariate Analysis", "Statistical Significance"])

with tab_integrity:
    st.subheader("Data Quality Audit")
    fig_miss, ax_miss = plt.subplots(figsize=(10, 2))
    sns.heatmap(df_raw.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax_miss)
    st.pyplot(fig_miss)
    st.caption("Missingness Map: Yellow lines indicate missing data points[cite: 1, 2].")

with tab_dist:
    st.subheader("Distribution & Outliers")
    selected_feat = st.selectbox("Select a feature to examine:", all_features)
    col_hist, col_box = st.columns(2)
    with col_hist:
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(df_raw[selected_feat], kde=True, ax=ax_hist, color="teal")
        st.pyplot(fig_hist)
    with col_box:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df_raw, x='Diagnosis', y=selected_feat, palette='coolwarm', ax=ax_box)
        st.pyplot(fig_box)

# Global variables for Phase 3 and 4
significant_features = []

with tab_stats:
    st.subheader("Normality & Significance (Alpha=0.01)[cite: 1, 2]")
    alpha = 0.01
    sig_summary = []
    for col in all_features:
        _, p_norm = stats.shapiro(df_raw[col])
        is_normal = p_norm > alpha
        mal, ben = df_raw[df_raw['Diagnosis'] == 1][col], df_raw[df_raw['Diagnosis'] == 0][col]
        _, p_val = stats.ttest_ind(mal, ben) if is_normal else stats.mannwhitneyu(mal, ben)
        
        if p_val < alpha:
            significant_features.append(col)
        sig_summary.append({"Feature": col, "P-Value": f"{p_val:.2e}", "Significant": p_val < alpha})
    st.table(pd.DataFrame(sig_summary))

# ==========================================
# PHASE 3 & 4: PREPARATION & MODELING
# ==========================================
@st.cache_resource
def train_clinical_model(data, selected_cols):
    """Phases 3 & 4: Automated Preparation and Modeling."""
    # Handle Multicollinearity
    X_subset = data[selected_cols]
    corr_matrix = X_subset.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    X_filtered = X_subset.drop(columns=to_drop)
    
    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, data['Diagnosis'], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_test_scaled, y_test, X_filtered.columns.tolist(), to_drop

model, scaler, X_test_scaled, y_test, final_features, dropped_cols = train_clinical_model(df_raw, significant_features)

st.divider()
st.header("⚙️ Phase 3 & 4: Preparation & Modeling")
st.write(f"**Feature Engineering:** Removed {len(dropped_cols)} redundant features with >90% correlation.")
st.write(f"**Final Feature Count:** {len(final_features)}")

# ==========================================
# PHASE 5: EVALUATION[cite: 1]
# ==========================================
st.header("🤖 Phase 5: Clinical Evaluation")
st.sidebar.title("🩺 Diagnostic Controls")
threshold = st.sidebar.slider("Malignancy Probability Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.caption("Lowering this threshold increases sensitivity (Recall)[cite: 1].")

y_probs = model.predict_proba(X_test_scaled)[:, 1]
y_pred_custom = (y_probs >= threshold).astype(int)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred_custom):.2%}")
    st.metric("Clinical Sensitivity (Recall)", f"{recall_score(y_test, y_pred_custom):.2%}")
with col_m2:
    cm = confusion_matrix(y_test, y_pred_custom)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

# ==========================================
# PHASE 6: DEPLOYMENT & EXPLAINABILITY[cite: 1, 2]
# ==========================================
st.divider()
st.header("🚀 Phase 6: Clinical Deployment")
st.sidebar.markdown("---")
st.sidebar.markdown("### Patient Input for Prediction")
user_inputs = {}
for feat in final_features[:5]: # Input for top 5 features
    user_inputs[feat] = st.sidebar.number_input(f"Enter {feat}", value=float(df_raw[feat].mean()))

# Prediction logic
full_input = {feat: df_raw[feat].mean() for feat in final_features}
full_input.update(user_inputs)
input_df = pd.DataFrame([full_input])
input_scaled = scaler.transform(input_df)
prob_malignant = model.predict_proba(input_scaled)[0][1]

st.subheader("Live Diagnostic Result")
if prob_malignant >= threshold:
    st.error(f"High Risk of Malignancy: {prob_malignant:.2%}")
else:
    st.success(f"Low Risk (Benign): {(1-prob_malignant):.2%}")

# Explainability with IndexError fix
st.subheader("🧬 Clinical Explanation (SHAP)[cite: 1]")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)

# Fix for IndexError: Handle list vs array output from SHAP
if isinstance(shap_values, list):
    # Standard format: select index 1 for Malignant class
    class_shap_values = shap_values[1]
    base_value = explainer.expected_value[1]
else:
    # Single array format for binary classification
    class_shap_values = shap_values
    base_value = explainer.expected_value

st_shap(shap.force_plot(base_value, class_shap_values, input_df), height=200)
st.caption("Red: Increases risk | Blue: Decreases risk[cite: 1].")
