import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from scipy import stats

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Breast Cancer Diagnostic Analytics",
    page_icon="🔬",
    layout="wide"
)

# Set global plotting theme
sns.set_theme(style="whitegrid", palette="viridis")

# ==========================================
# 2. DATA ACQUISITION (CACHED)
# ==========================================
@st.cache_data
def load_data():
    """Fetches and encodes the UCI Breast Cancer Wisconsin dataset."""
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets
    
    # Encode labels: Malignant = 1, Benign = 0
    y_encoded = y.iloc[:, 0].map({'M': 1, 'B': 0})
    
    # Combine for analysis
    df = pd.concat([X, y_encoded], axis=1)
    df.rename(columns={df.columns[-1]: 'Diagnosis'}, inplace=True)
    return df, X.columns.tolist()

df_raw, all_features = load_data()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🔍 Analytics Control Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("Feature Selection")
selected_features = st.sidebar.multiselect(
    "Select features for visualization:",
    options=all_features,
    default=all_features[:6]
)

st.sidebar.subheader("Filter by Class")
show_malignant = st.sidebar.checkbox("Show Malignant Cases", value=True)
show_benign = st.sidebar.checkbox("Show Benign Cases", value=True)

# Apply filters to dataframe
mask = pd.Series([False] * len(df_raw))
if show_malignant:
    mask |= (df_raw['Diagnosis'] == 1)
if show_benign:
    mask |= (df_raw['Diagnosis'] == 0)

df = df_raw[mask]

# ==========================================
# PHASE 1: BUSINESS UNDERSTANDING
# ==========================================
st.title("🔬 Breast Cancer Diagnostic Analytics")
st.markdown("---")

st.markdown("""
### CRISP-DM Phase 1: Business Understanding
The primary objective is to develop a diagnostic tool to classify breast tumors based on physical characteristics derived from digitized images.

**Clinical Standards:**
*   **Explainability:** Ensuring models are transparent for clinical review.
*   **Statistical Rigor:** Applying a strict significance level of $\alpha = 0.01$ to minimize false negatives[cite: 1].
""")

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(df))
col2.metric("Malignant (Class 1)", len(df[df['Diagnosis'] == 1]))
col3.metric("Benign (Class 0)", len(df[df['Diagnosis'] == 0]))

# ==========================================
# PHASE 2: DATA UNDERSTANDING (EDA)
# ==========================================
st.header("📊 CRISP-DM Phase 2: Data Understanding")

tabs = st.tabs(["Data Quality", "Distribution & Outlier Analysis", "Statistical Tally"])

with tabs[0]:
    st.subheader("Missingness Map")
    fig_miss, ax_miss = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax_miss)
    st.pyplot(fig_miss)
    st.caption("Yellow indicates NaN values. The map verifies data integrity[cite: 1].")

with tabs[1]:
    if not selected_features:
        st.warning("Please select features in the sidebar to view distributions.")
    else:
        st.subheader("Outlier Detection & Skewness")
        
        # Boxplots
        fig_box, ax_box = plt.subplots(figsize=(15, 8))
        df_melt = df.melt(id_vars='Diagnosis', value_vars=selected_features)
        sns.boxplot(data=df_melt, x='variable', y='value', hue='Diagnosis', palette='coolwarm', ax=ax_box)
        plt.xticks(rotation=45)
        st.pyplot(fig_box)
        
        # Histograms
        st.subheader("Feature Frequency Distributions")
        fig_hist, axes = plt.subplots(nrows=(len(selected_features)+2)//3, ncols=3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(selected_features):
            sns.histplot(df[col], kde=True, color="teal", ax=axes[i])
            axes[i].set_title(f"Distribution: {col}")
        plt.tight_layout()
        st.pyplot(fig_hist)

with tabs[2]:
    st.subheader("Automated Statistical Significance ($\alpha=0.01$)")
    alpha = 0.01
    results = []

    for col in all_features:
        # Normality Check
        _, p_norm = stats.shapiro(df_raw[col])
        is_normal = p_norm > alpha

        malignant = df_raw[df_raw['Diagnosis'] == 1][col]
        benign = df_raw[df_raw['Diagnosis'] == 0][col]

        # Test selection based on normality[cite: 1]
        if is_normal:
            _, p_val = stats.ttest_ind(malignant, benign)
            test_type = "T-test"
        else:
            _, p_val = stats.mannwhitneyu(malignant, benign)
            test_type = "Mann-Whitney U"

        results.append({
            "Feature": col,
            "Distribution": "Normal" if is_normal else "Skewed",
            "Test Used": test_type,
            "P-Value": f"{p_val:.2e}",
            "Significant": p_val < alpha
        })

    st.table(pd.DataFrame(results))

# ==========================================
# MULTIVARIATE ANALYSIS
# ==========================================
st.header("🔗 Multivariate Analysis")
st.markdown("Correlogram of statistically significant features.")
sig_cols = [r['Feature'] for r in results if r['Significant']]
if sig_cols:
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[sig_cols].corr(), cmap='coolwarm', center=0, ax=ax_corr)
    st.pyplot(fig_corr)
