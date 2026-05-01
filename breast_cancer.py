import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.title("🧬 Breast Cancer Diagnostic AI System")
st.write("Clinical Decision Support using Machine Learning + SHAP Explainability")

st.header("1. Data Loading")

data = fetch_ucirepo(id=17)

X = data.data.features
y = data.data.targets

y = y.iloc[:, 0].map({'M': 1, 'B': 0})

df = pd.concat([X, y], axis=1)
df.rename(columns={df.columns[-1]: "Diagnosis"}, inplace=True)

st.success("Dataset loaded successfully")
st.dataframe(df.head())

st.header("2. Exploratory Data Analysis")

fig, ax = plt.subplots()
sns.heatmap(df.isnull(), cbar=False, ax=ax)
st.pyplot(fig)

st.subheader("Target Distribution")
st.bar_chart(df["Diagnosis"].value_counts())

st.header("3. Model Preparation")

sig_features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    df[sig_features], df["Diagnosis"],
    test_size=0.2,
    random_state=42,
    stratify=df["Diagnosis"]
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.success("Data prepared")

st.header("4. Model Training")

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "ANN": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    st.write(f"{name} trained ✅")

st.header("5. Model Evaluation")

selected_model = st.selectbox("Choose Model", list(trained_models.keys()))

model = trained_models[selected_model]
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
st.pyplot(fig)

if selected_model == "Decision Tree":
    st.subheader("Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, filled=True, ax=ax)
    st.pyplot(fig)

st.header("6. Explainability (SHAP)")

if selected_model == "Decision Tree":

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.write("SHAP Summary Plot")

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)
