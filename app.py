import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

st.set_page_config(page_title="ML Assignment 2 - Adult Income", layout="wide")

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = "model"
METRICS_FILE = os.path.join(MODEL_DIR, "metrics_comparison.csv")
DEFAULT_TEST_PATH = "Data/demo_test.csv"  # <- Your demo file path

MODEL_FILES = {
    "Logistic Regression": os.path.join(MODEL_DIR, "Logistic_Regression.joblib"),
    "Decision Tree": os.path.join(MODEL_DIR, "Decision_Tree.joblib"),
    "KNN": os.path.join(MODEL_DIR, "KNN.joblib"),
    "Naive Bayes": os.path.join(MODEL_DIR, "Naive_Bayes.joblib"),
    "Random Forest": os.path.join(MODEL_DIR, "Random_Forest.joblib"),
    "XGBoost": os.path.join(MODEL_DIR, "XGBoost.joblib"),
}

TARGET_COL = "income"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_metrics_table(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def normalize_strings(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("?", np.nan)
    return df


def binarize_income(y):
    y_str = y.astype(str).str.strip().str.replace(".", "", regex=False)
    mapping = {"<=50K": 0, ">50K": 1}
    return y_str.map(mapping)


def safe_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_proba)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": safe_auc(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

# -----------------------------
# UI
# -----------------------------
st.title("ML Assignment 2 — Adult Income Classification (6 Models)")

st.markdown("""
This app lets you:
- Upload a CSV test dataset
- Choose one of 6 classification models
- View evaluation metrics
- View confusion matrix / classification report
""")

# Sidebar
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload CSV (test data only)", type=["csv"])

# -----------------------------
# Show saved training metrics
# -----------------------------
st.subheader("Saved Evaluation Metrics (from training run)")
metrics_df = load_metrics_table(METRICS_FILE)

if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning("metrics_comparison.csv not found inside model/ folder.")

st.divider()

# -----------------------------
# Load Data (Auto Demo or Upload)
# -----------------------------
if uploaded_file is None:
    st.info("No file uploaded. Running demo dataset.")
    if not os.path.exists(DEFAULT_TEST_PATH):
        st.error(f"Demo file not found at {DEFAULT_TEST_PATH}")
        st.stop()
    df = pd.read_csv(DEFAULT_TEST_PATH)
else:
    df = pd.read_csv(uploaded_file)

df = normalize_strings(df)

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# Separate target if available
# -----------------------------
has_target = TARGET_COL in df.columns

if has_target:
    y = binarize_income(df[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])
else:
    y = None
    X = df.copy()

# -----------------------------
# Load Model
# -----------------------------
model_path = MODEL_FILES[model_name]

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X)[:, 1]
else:
    y_proba = y_pred.astype(float)

st.subheader("Prediction Distribution")
st.write(pd.Series(y_pred).value_counts())

st.divider()

# -----------------------------
# Evaluation (only if target exists)
# -----------------------------
st.subheader("Confusion Matrix / Classification Report")

if y is None:
    st.info("Target column not found. Upload dataset with 'income' column to see evaluation.")
    st.stop()

metrics = compute_metrics(y, y_pred, y_proba)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Evaluation Metrics")
    st.json(metrics)

with col2:
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.dataframe(pd.DataFrame(cm,
                              index=["True_0", "True_1"],
                              columns=["Pred_0", "Pred_1"]))

st.markdown("### Classification Report")
st.text(classification_report(y, y_pred, zero_division=0))
