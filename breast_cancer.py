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
# CONFIGURATION & DATA ACQUISITION (PHASE 2)
# ==============================================================================
st.set_page_config(page_title="Breast Cancer Diagnostic Portal", layout="wide")
sns.set_theme(style="whitegrid", palette="viridis")

@st.cache_data
def get_clinical_data():
    # Data Understanding: Acquisition
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    # Encoding: Malignant = 1, Benign = 0
    y_encoded = y.iloc[:, 0].map({'M': 1, 'B': 0})
    df = pd.concat([X, y_encoded], axis=1)
    df.rename(columns={df.columns[-1]: 'Diagnosis'}, inplace=True)
    return df, X.columns.tolist()

df_raw, all_features = get_clinical_data()

# ==============================================================================
# PHASE 1: BUSINESS UNDERSTANDING
# ==============================================================================
st.title("🔬 Clinical Breast Cancer Diagnostic System")
with st.expander("📌 CRISP-DM Phase 1: Business Understanding", expanded=True):
    st.markdown("""
    **Clinical Objective:** Develop an automated system to classify tumors with high statistical confidence.
    *   **Rigor:** Alpha level $\alpha=0.01$ ensures only robust features are used[cite: 1].
    *   **Clinical Priority:** High sensitivity (Recall) is mandatory to minimize missed diagnoses[cite: 1].
    *   **Requirement:** Explainable AI (SHAP) to justify clinical decisions to medical staff[cite: 1].
    """)

# ==============================================================================
# PHASE 2: DATA UNDERSTANDING (EDA)
# ==============================================================================
st.header("📊 Phase 2: Data Understanding")
p2_tab1, p2_tab2, p2_tab3 = st.tabs(["Data Quality", "Distribution", "Significance"])

with p2_tab1:
    st.subheader("Data Integrity Audit")
    fig_miss, ax_miss = plt.subplots(figsize=(10, 2))
    sns.heatmap(df_raw.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax_miss)
    st.pyplot(fig_miss)
    st.caption("Visualizing missingness patterns across clinical features[cite: 1, 2].")

with p2_tab2:
    st.subheader("Univariate Feature Spreads")
    f_select = st.selectbox("Select Feature for Analysis", all_features)
    c1, c2 = st.columns(2)
    with c1:
        fig_h, ax_h = plt.subplots()
        sns.histplot(df_raw[f_select], kde=True, ax=ax_h, color="teal")
        st.pyplot(fig_h)
    with c2:
        fig_b, ax_b = plt.subplots()
        sns.boxplot(data=df_raw, x='Diagnosis', y=f_select, palette='coolwarm', ax=ax_b)
        st.pyplot(fig_b)

# Variable to hold features for Phase 3
sig_features = []

with p2_tab3:
    st.subheader("Statistical Tally (Alpha=0.01)")
    alpha = 0.01
    tally = []
    for col in all_features:
        _, p_norm = stats.shapiro(df_raw[col])
        is_normal = p_norm > alpha
        m, b = df_raw[df_raw['Diagnosis']==1][col], df_raw[df_raw['Diagnosis']==0][col]
        # Automatic test selection based on distribution[cite: 1, 2]
        _, p_val = stats.ttest_ind(m, b) if is_normal else stats.mannwhitneyu(m, b)
        if p_val < alpha:
            sig_features.append(col)
        tally.append({"Feature": col, "Distribution": "Normal" if is_normal else "Skewed", "P-Value": f"{p_val:.2e}", "Significant": p_val < alpha})
    st.dataframe(pd.DataFrame(tally), use_container_width=True)

# ==============================================================================
# PHASE 3 & 4: DATA PREPARATION & MODELING
# ==============================================================================
@st.cache_resource
def build_clinical_model(df, features):
    # Phase 3: Data Preparation
    X_sub = df[features]
    # Remove redundant features (>0.9 correlation)
    corr = X_sub.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.90)]
    X_clean = X_sub.drop(columns=to_drop)
    
    # Scaling and Splitting
    X_train, X_test, y_train, y_test = train_test_split(X_clean, df['Diagnosis'], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Phase 4: Modeling
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_test_scaled, y_test, X_clean.columns.tolist()

model, scaler, X_test_scaled, y_test, final_features = build_clinical_model(df_raw, sig_features)

st.divider()
st.header("⚙️ Phase 3 & 4: Preparation & Modeling")
st.write(f"**Final Feature Set:** {len(final_features)} variables selected after removing redundancy.")

# ==============================================================================
# PHASE 5: EVALUATION
# ==============================================================================
st.header("🤖 Phase 5: Evaluation")
st.sidebar.title("🩺 Control Panel")
risk_threshold = st.sidebar.slider("Malignancy Risk Threshold", 0.0, 1.0, 0.5, 0.05)

probs = model.predict_proba(X_test_scaled)[:, 1]
preds = (probs >= risk_threshold).astype(int)

m1, m2 = st.columns(2)
with m1:
    st.metric("Detection Sensitivity (Recall)", f"{recall_score(y_test, preds):.2%}")
    st.info("Sensitivity tracks the ability to catch all malignant cases[cite: 1].")
with m2:
    cm = confusion_matrix(y_test, preds)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted Risk')
    ax_cm.set_ylabel('Actual Pathology')
    st.pyplot(fig_cm)

# ==============================================================================
# PHASE 6: DEPLOYMENT (INFERENCE)
# ==============================================================================
st.divider()
st.header("🚀 Phase 6: Clinical Deployment")
st.sidebar.markdown("### Patient Vitals Input")

inputs = {}
for f in final_features[:5]: # Input for top 5 features
    inputs[f] = st.sidebar.number_input(f"Value for {f}", value=float(df_raw[f].mean()))

# Prediction process
full_input = {f: df_raw[f].mean() for f in final_features}
full_input.update(inputs)
input_df = pd.DataFrame([full_input])
input_scaled = scaler.transform(input_df)
patient_prob = model.predict_proba(input_scaled)[0][1]

st.subheader("Diagnostic Assessment")
if patient_prob >= risk_threshold:
    st.error(f"High Pathological Risk Identified: {patient_prob:.2%}")
else:
    st.success(f"Low Pathological Risk (Benign): {1-patient_prob:.2%}")

# ==============================================================================
# EXPLAINABILITY (BUG FIX APPLIED)
# ==============================================================================
st.subheader("🧬 Explainable Clinical Justification (SHAP)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)

# 1. Safely extract 1D SHAP values for the Malignant class
if isinstance(shap_values, list):
    s_values = shap_values[1][0] 
    base_val = explainer.expected_value[1]
elif len(np.shape(shap_values)) == 3:
    # Handles SHAP > 0.41 3D array formats
    s_values = shap_values[0, :, 1]
    base_val = explainer.expected_value[1]
else:
    s_values = shap_values[0]
    base_val = explainer.expected_value

# Ensure base_value is a native Python float
if hasattr(base_val, "__len__"):
    base_val = float(base_val[0])
else:
    base_val = float(base_val)

# 2. Extract raw 1D numpy array to prevent Pandas Series IndexErrors
feature_vals = np.round(input_df.values[0], 4)

# 3. Generate Plot using native types
plot_fig = shap.force_plot(
    base_value=base_val,
    shap_values=s_values,
    features=feature_vals,
    feature_names=final_features
)

st_shap(plot_fig, height=200)
st.caption("Red: Increases Malignancy Risk | Blue: Decreases Risk[cite: 1].")
