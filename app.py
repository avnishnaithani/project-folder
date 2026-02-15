# app.py
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
# Config
# -----------------------------
MODEL_DIR = "model"
METRICS_FILE = os.path.join(MODEL_DIR, "metrics_comparison.csv")

MODEL_FILES = {
    "Logistic Regression": os.path.join(MODEL_DIR, "Logistic_Regression.joblib"),
    "Decision Tree": os.path.join(MODEL_DIR, "Decision_Tree.joblib"),
    "KNN": os.path.join(MODEL_DIR, "KNN.joblib"),
    "Naive Bayes": os.path.join(MODEL_DIR, "Naive_Bayes.joblib"),
    "Random Forest": os.path.join(MODEL_DIR, "Random_Forest.joblib"),
    "XGBoost": os.path.join(MODEL_DIR, "XGBoost.joblib"),
}

# Adult Income common target name (your dataset uses this)
TARGET_COL = "income"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_metrics_table(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures uploaded CSV matches training cleaning:
    - trims spaces
    - converts '?' to NaN for object columns
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].replace("?", np.nan)
    return out


def binarize_income(y: pd.Series) -> pd.Series:
    """
    Convert target labels to 0/1.
    Accepts <=50K, <=50K., >50K, >50K.
    """
    y_str = y.astype(str).str.strip().str.replace(".", "", regex=False)
    mapping = {"<=50K": 0, ">50K": 1, "0": 0, "1": 1, "False": 0, "True": 1}
    y_bin = y_str.map(mapping)

    if y_bin.isna().any():
        uniques = sorted(y_str.dropna().unique().tolist())
        if len(uniques) == 2:
            auto_map = {uniques[0]: 0, uniques[1]: 1}
            y_bin = y_str.map(auto_map)
        else:
            return None
    return y_bin.astype(int)


def safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": safe_auc(y_true, y_proba),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


# -----------------------------
# UI
# -----------------------------
st.title("ML Assignment 2 — Adult Income Classification (6 Models)")

st.markdown(
    """
This app lets you:
- Upload a **CSV test dataset**
- Choose one of **6 classification models**
- View **evaluation metrics**
- View **confusion matrix / classification report** (if target column is provided)
"""
)

# Sidebar controls
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload CSV (test data only)", type=["csv"])

# Display saved evaluation metrics table (from training) – requirement (c)
st.subheader("Saved Evaluation Metrics (from training run)")
metrics_df = load_metrics_table(METRICS_FILE)
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning(
        f"metrics_comparison.csv not found at `{METRICS_FILE}`. "
        "If you saved it during training, commit it inside the model/ folder."
    )

st.divider()

# Prediction section
st.subheader("Upload Data → Predict")

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
df = normalize_strings(df)

st.write("Preview of uploaded data:")
st.dataframe(df.head(10), use_container_width=True)

# Separate target if present
has_target = TARGET_COL in df.columns
if has_target:
    y_raw = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    y = binarize_income(y_raw)
    if y is None:
        st.error(
            f"Target column `{TARGET_COL}` exists but labels are not recognized. "
            "Use <=50K / >50K (or their dotted variants)."
        )
        st.stop()
else:
    X = df.copy()
    y = None

# Load model and predict
model_path = MODEL_FILES[model_name]
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Ensure it exists in your GitHub repo under model/.")
    st.stop()

pipe = load_model(model_path)

try:
    y_pred = pipe.predict(X)
except Exception as e:
    st.error(
        "Prediction failed. Common causes:\n"
        "- Uploaded CSV columns do not match training features\n"
        "- Extra/missing columns\n\n"
        f"Error: {e}"
    )
    st.stop()

# Probability for AUC and confidence (if available)
y_proba = None
if hasattr(pipe, "predict_proba"):
    try:
        y_proba = pipe.predict_proba(X)[:, 1]
    except Exception:
        y_proba = None

# Show predictions summary
pred_counts = pd.Series(y_pred).value_counts().sort_index()
st.write("Prediction class counts (0 = <=50K, 1 = >50K):")
st.write(pred_counts)

# Optional: show predictions table
out_df = X.copy()
out_df["predicted_income_class"] = y_pred
st.write("Predictions (first 20 rows):")
st.dataframe(out_df.head(20), use_container_width=True)

st.divider()

# Requirement (d): Confusion matrix / classification report
st.subheader("Confusion Matrix / Classification Report")

if y is None:
    st.info(
        f"Your uploaded CSV does not include the target column `{TARGET_COL}`.\n\n"
        "To see confusion matrix and classification report, upload a CSV that also includes the true labels "
        f"in a column named `{TARGET_COL}`."
    )
    st.stop()

# Compute metrics live on uploaded test data
if y_proba is None:
    # If proba isn't available, approximate AUC won't be meaningful; set y_proba to y_pred
    y_proba = y_pred.astype(float)

live_metrics = compute_metrics(y.values, y_pred, y_proba)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Live Evaluation Metrics (on uploaded data)")
    st.json(live_metrics)

with col2:
    st.markdown("### Confusion Matrix (rows=true, cols=pred)")
    cm = confusion_matrix(y.values, y_pred)
    cm_df = pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    st.dataframe(cm_df, use_container_width=True)

st.markdown("### Classification Report")
st.text(classification_report(y.values, y_pred, zero_division=0))
