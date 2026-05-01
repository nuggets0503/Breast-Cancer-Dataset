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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score

# ==============================================================================
# CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(page_title="Breast Cancer Clinical Decision Support", layout="wide")
sns.set_theme(style="whitegrid", palette="viridis")

# ==============================================================================
# DATA ACQUISITION (PHASE 2 START)
# ==============================================================================
@st.cache_data
def load_and_preprocess_data():
    # Fetch dataset from UCI repository
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    
    # Clinical Encoding: Malignant = 1, Benign = 0
    y_encoded = y.iloc[:, 0].map({'M': 1, 'B': 0})
    
    df = pd.concat([X, y_encoded], axis=1)
    df.rename(columns={df.columns[-1]: 'Diagnosis'}, inplace=True)
    return df, X.columns.tolist()

df_raw, all_features = load_and_preprocess_data()

# ==============================================================================
# SIDEBAR CONTROLS (PHASE 6 INFRASTRUCTURE)
# ==============================================================================
st.sidebar.title("🩺 Diagnostic Controls")

st.sidebar.markdown("### Clinical Thresholds")
threshold = st.sidebar.slider("Malignancy Probability Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.caption("Lowering this threshold increases sensitivity to catch more cases.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Patient Vitals (Input for Prediction)")
user_inputs = {}
# Using a selection of high-impact features for the sidebar input demo
demo_features = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1']
for feat in demo_features:
    user_inputs[feat] = st.sidebar.number_input(
        f"Enter {feat}", 
        value=float(df_raw[feat].mean()),
        format="%.4f"
    )

# ==============================================================================
# PHASE 1: BUSINESS UNDERSTANDING
# ==============================================================================
st.title("🔬 Breast Cancer Diagnostic Analytics Portal")
with st.container():
    st.header("Phase 1: Business Understanding")
    st.markdown("""
    **Objective:** Develop an explainable diagnostic tool to classify tumors as Malignant or Benign based on FNA images.
    
    **Clinical Mandates:**
    *   **Statistical Rigor:** Alpha level of $\alpha = 0.01$ to ensure feature robustness.
    *   **Sensitivity:** Primary focus on minimizing False Negatives (missed diagnoses).
    *   **Explainability:** Doctors must understand which nuclear features drive a high-risk prediction.
    """)

# ==============================================================================
# PHASE 2: DATA UNDERSTANDING
# ==============================================================================
st.divider()
st.header("📊 Phase 2: Data Understanding (EDA)")

tab_quality, tab_dist, tab_stats = st.tabs(["Data Quality", "Distribution & Outliers", "Statistical Tally"])

with tab_quality:
    st.subheader("Data Integrity Audit")
    fig_miss, ax_miss = plt.subplots(figsize=(10, 2))
    sns.heatmap(df_raw.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax_miss)
    st.pyplot(fig_miss)
    st.caption("Missingness Map: Yellow lines would indicate missing data points.")

with tab_dist:
    st.subheader("Feature Visualizations")
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

with tab_stats:
    st.subheader("Normality & Significance Tally (Alpha=0.01)")
    alpha = 0.01
    sig_summary = []
    significant_features = []

    for col in all_features:
        # Normality Testing
        _, p_norm = stats.shapiro(df_raw[col])
        is_normal = p_norm > alpha
        mal = df_raw[df_raw['Diagnosis'] == 1][col]
        ben = df_raw[df_raw['Diagnosis'] == 0][col]
        
        # Test Selection
        _, p_val = stats.ttest_ind(mal, ben) if is_normal else stats.mannwhitneyu(mal, ben)
        
        if p_val < alpha:
            significant_features.append(col)
            
        sig_summary.append({
            "Feature": col, 
            "Distribution": "Normal" if is_normal else "Skewed", 
            "P-Value": f"{p_val:.2e}", 
            "Significant": p_val < alpha
        })
    st.table(pd.DataFrame(sig_summary))

# ==============================================================================
# PHASE 3: DATA PREPARATION
# ==============================================================================
st.divider()
st.header("⚙️ Phase 3: Data Preparation")

# 1. Feature Selection (Filtering by p-value)
X = df_raw[significant_features]
y = df_raw['Diagnosis']

# 2. Addressing Multicollinearity (Professional Improvement)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
X_filtered = X.drop(columns=to_drop)

st.write(f"Initially identified **{len(significant_features)}** significant features.")
st.write(f"Removed **{len(to_drop)}** redundant/highly correlated features: {', '.join(to_drop)}")
final_features = X_filtered.columns.tolist()

# 3. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# PHASE 4 & 5: MODELING & EVALUATION
# ==============================================================================
st.divider()
st.header("🤖 Phase 4 & 5: Modeling & Clinical Evaluation")

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Generate Probabilities
y_probs = model.predict_proba(X_test_scaled)[:, 1]
y_pred_custom = (y_probs >= threshold).astype(int)

col_metrics, col_cm = st.columns(2)

with col_metrics:
    st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred_custom):.2%}")
    st.metric("Clinical Sensitivity (Recall)", f"{recall_score(y_test, y_pred_custom):.2%}")
    st.info("Recall is our primary metric: it measures our ability to detect all malignant cases.")

with col_cm:
    cm = confusion_matrix(y_test, y_pred_custom)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

# ==============================================================================
# PHASE 6: DEPLOYMENT (INFERENCE & EXPLAINABILITY)
# ==============================================================================
st.divider()
st.header("🚀 Phase 6: Clinical Deployment")

# Prepare user input for prediction
# We map user inputs to a full dataframe and fill missing values with dataset means
full_input_dict = {feat: df_raw[feat].mean() for feat in final_features}
for k, v in user_inputs.items():
    if k in full_input_dict:
        full_input_dict[k] = v

input_df = pd.DataFrame([full_input_dict])
input_scaled = scaler.transform(input_df)
prob_malignant = model.predict_proba(input_scaled)[0][1]

st.subheader("Live Diagnostic Result")
if prob_malignant >= threshold:
    st.error(f"High Risk of Malignancy: {prob_malignant:.2%}")
else:
    st.success(f"Low Risk of Malignancy (Benign): {(1-prob_malignant):.2%}")

# Explainability Section
st.subheader("🧬 Clinical Explanation (SHAP)")
st.write("This visualization explains which factors most contributed to the diagnosis above.")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)
# SHAP for Random Forest outputs a list of arrays [Benign, Malignant]. We use index 1.
st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_df), height=200)
st.caption("Red bars indicate features that increased the risk. Blue bars indicate features that decreased the risk.")
