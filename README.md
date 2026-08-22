# 🔬 Clinical Breast Cancer Diagnostic System

_An explainable machine-learning tool that classifies breast tumors as malignant or benign, which prioritize diagnostic sensitivity so that no cancer case goes missed.*_

## 📌 Executive Summary
Missed malignancies carry a far higher cost than false alarms. This application builds a diagnostic classifier on the Wisconsin Breast Cancer dataset that is tuned for **high sensitivity (recall)** to catch as many true cancer cases as possible and pairs every prediction with a **SHAP explanation** so clinicians can see *why* a tumor was flagged, rather than trusting a black box.

The project follows the full **CRISP-DM** framework, from business understanding through deployment.

---

## 🎯 Business Problem
In clinical screening, a false negative (telling a patient they are healthy when they are not) is the costliest possible error. The goal is an automated decision-support tool that:
* Maximizes **detection sensitivity** to minimize missed diagnoses.
* Uses only **statistically significant** tumor features (α = 0.01) to avoid noise-driven predictions.
* Provides **explainable, per-patient justifications** that medical staff can interpret and trust.

---

## 🧪 Approach (CRISP-DM)
| Phase | What happens |
| :--- | :--- |
| **1. Business Understanding** | Defines sensitivity as the primary success metric. |
| **2. Data Understanding** | Data-integrity audit, distribution analysis, and per-feature significance testing (Shapiro–Wilk → t-test / Mann–Whitney at α = 0.01). |
| **3–4. Preparation & Modeling** | Removes redundant features (>0.90 correlation), scales inputs, trains a Random Forest classifier. |
| **5. Evaluation** | Confusion matrix and an adjustable **risk-threshold slider** to tune the sensitivity/precision trade-off. |
| **6. Deployment** | Live inference on patient inputs, with a SHAP force plot explaining each prediction. |

---

## 📊 Key Results
* Feature set reduced to only statistically significant, non-redundant predictors.
* An interactive threshold control lets clinicians dial sensitivity up or down for their risk tolerance.
* Every prediction ships with a SHAP explanation to identify which tumor characteristics drove the result.

## 💡 Recommendation
Deploy this as a **screening triage aid, not a replacement for pathology**: set the malignancy threshold conservatively to favor sensitivity, route every flagged case to human review, and use the SHAP output to document the clinical rationale for each referral.

---

## 🛠️ Tech Stack
* **Python** — Streamlit, scikit-learn, SHAP, SciPy, pandas, seaborn, matplotlib
* **Data:** UCI Wisconsin Breast Cancer dataset (via `ucimlrepo`)

## 💻 Run Locally
```bash
git clone https://github.com/nuggets0503/breast-cancer-dataset.git
cd breast-cancer-dataset
pip install -r requirements.txt
python breast_cancer.py
```

## ⚖️ License
GNU General Public License v3.0 — see the LICENSE file.

## ⚠️ Disclaimer
This is an educational project and is **not** a certified medical device. It must not be used for actual clinical diagnosis.
