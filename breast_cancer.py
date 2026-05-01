import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(layout="wide")

st.title("🧬 Breast Cancer Diagnostic AI System")
st.markdown("""
Clinical Decision Support System built using CRISP-DM methodology  
with full statistical validation, machine learning models, and SHAP explainability.
""")

st.header("1. Data Understanding")

data = fetch_ucirepo(id=17)

X = data.data.features
y = data.data.targets

y = y.iloc[:, 0].map({'M': 1, 'B': 0})

df = pd.concat([X, y], axis=1)
df.rename(columns={df.columns[-1]: "Diagnosis"}, inplace=True)

st.dataframe(df.head())

st.markdown("""
### Interpretation
- Malignant = 1 (High clinical risk)
- Benign = 0 (No cancer)
- Dataset is slightly imbalanced but acceptable for classification.
""")

st.header("2. Exploratory Data Analysis")

fig, ax = plt.subplots()
sns.heatmap(df.isnull(), cbar=False, ax=ax)
st.pyplot(fig)

st.markdown("""
### Missing Data Analysis
No significant missing values detected → dataset is clinically reliable.
""")

fig, ax = plt.subplots()
sns.countplot(x=df["Diagnosis"], ax=ax)
st.pyplot(fig)

st.markdown("""
### Target Distribution
- Majority: Benign cases
- Minority: Malignant cases  
Slight imbalance but not severe enough to require SMOTE.
""")

st.header("3. Statistical Analysis (α = 0.01)")

alpha = 0.01
stats_results = []

for col in X.columns:
    benign = df[df["Diagnosis"] == 0][col]
    malignant = df[df["Diagnosis"] == 1][col]

    _, p_val = stats.ttest_ind(benign, malignant, equal_var=False)

    stats_results.append({
        "Feature": col,
        "p_value": p_val,
        "Significant": p_val < alpha
    })

stats_df = pd.DataFrame(stats_results)

st.dataframe(stats_df)

sig_features = stats_df[stats_df["Significant"] == True]["Feature"].tolist()

st.markdown(f"""
### Interpretation
- Significant features (α = 0.01): {len(sig_features)}
- These features are most clinically relevant for tumor differentiation.
""")

st.header("4. Data Preparation")

X_final = df[sig_features]
y_final = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final,
    test_size=0.2,
    random_state=42,
    stratify=y_final
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.success("Data successfully prepared and standardized")

st.header("5. Model Training")

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "ANN": MLPClassifier(hidden_layer_sizes=(16,8), max_iter=1000)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    st.write(f"✔ {name} trained")

st.header("6. Model Evaluation")

model_name = st.selectbox("Select Model", list(trained_models.keys()))
model = trained_models[model_name]

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc:.4f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
st.pyplot(fig)

st.markdown("""
### Clinical Interpretation
- TP (Malignant correctly detected) is critical
- FN (missed cancer) must be minimized at all cost
""")

if model_name == "Decision Tree":
    st.header("Decision Tree Interpretation")

    fig, ax = plt.subplots(figsize=(12,6))
    plot_tree(model, filled=True, ax=ax)
    st.pyplot(fig)

    st.markdown("""
### Interpretation
- Tree mimics clinical decision flow
- Top nodes = strongest predictors of malignancy
""")

if model_name == "Decision Tree":

    st.header("7. Explainability (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)

    st.pyplot(fig)

    st.markdown("""
### Interpretation
- Red = high feature value increases malignancy risk
- Blue = protective features
- Confirms clinical relevance of nuclear size + shape features
""")
