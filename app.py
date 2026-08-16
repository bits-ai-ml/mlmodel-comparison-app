import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}

FEATURE_COLUMNS = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("🩺 Breast Cancer Classification — Model Comparison App")
st.markdown(
    "**BITS MTech ML Assignment 2** &nbsp;|&nbsp; Akhil Jain (2025ac05543)\n\n"
    "Dataset: UCI Breast Cancer Wisconsin (Diagnostic) — binary classification "
    "(malignant vs benign), 30 features, 569 instances."
)

st.sidebar.header("1. Upload test data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (must contain the 30 feature columns + a 'target' column)",
    type=["csv"]
)

st.sidebar.header("2. Choose a model")
model_choice = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing {len(missing_cols)} required feature column(s), "
                  f"e.g. {missing_cols[:5]}. Please upload the provided test_data.csv.")
        st.stop()

    has_target = "target" in df.columns

    X = df[FEATURE_COLUMNS]
    scaler = load_scaler()
    X_scaled = scaler.transform(X)

    model = load_model(MODEL_FILES[model_choice])
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    st.subheader(f"Results — {model_choice}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Predictions (first 20 rows)**")
        preview = df.copy()
        preview["prediction"] = np.where(y_pred == 1, "benign", "malignant")
        preview["malignant_prob"] = np.round(1 - y_proba, 4)
        st.dataframe(preview.head(20), use_container_width=True)

    with col2:
        if has_target:
            y_true = df["target"]
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            st.markdown("**Evaluation Metrics**")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                "Value": [acc, auc, prec, rec, f1, mcc]
            })
            metrics_df["Value"] = metrics_df["Value"].round(4)
            st.table(metrics_df.set_index("Metric"))
        else:
            st.info("Upload a CSV with a 'target' column to see evaluation metrics, "
                    "confusion matrix, and classification report.")

    if has_target:
        st.subheader("Confusion Matrix & Classification Report")
        c1, c2 = st.columns([1, 1])

        with c1:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"]).plot(
                ax=ax, colorbar=False
            )
            st.pyplot(fig)

        with c2:
            report = classification_report(
                y_true, y_pred, target_names=["malignant", "benign"], output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
else:
    st.info("👈 Upload the provided `data/test_data.csv` from the sidebar to see model "
            "predictions and metrics. Then pick a model from the dropdown to compare.")

st.markdown("---")
st.caption("All 5 models (Logistic Regression, Decision Tree, kNN, Naive Bayes, "
           "Random Forest) were trained on an 80/20 stratified split of the same dataset. "
           "See README.md / model_training.ipynb for the full comparison table.")
