import streamlit as st
import pandas as pd
import joblib
import gc
import requests
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, matthews_corrcoef
)

st.set_page_config(page_title="Asteroid Hazard Prediction", layout="wide")
st.title("🚀 Asteroid Hazard Prediction - ML Models")
st.markdown("Predict whether an asteroid is **Potentially Hazardous (PHA)** using multiple ML models.")

scaler = joblib.load("model/scaler.pkl")
train_columns = joblib.load("model/train_columns.pkl")

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

dataset_url = "https://raw.githubusercontent.com/Kiraiiiii/asteroid-hazard-prediction/main/datas/test_data.csv"

@st.cache_data
def load_test_csv(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

st.sidebar.header("📂 Upload Dataset & Select Model")

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(model_files.keys())
)

model = joblib.load(model_files[model_name])

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File (Test Data)",
    type=["csv"]
)

st.sidebar.markdown("### 📥 Download Sample Test Dataset")

try:
    csv_data = load_test_csv(dataset_url)
    st.sidebar.download_button(
        label="Download test_data.csv",
        data=csv_data,
        file_name="test_data.csv",
        mime="text/csv"
    )
except:
    st.sidebar.warning("⚠️ Unable to fetch dataset from GitHub.")

use_sample = st.sidebar.checkbox("Use Sample Dataset (test_data.csv)", value=False)
data = None

if use_sample:
    try:
        data = pd.read_csv(dataset_url)
        st.success("✅ Loaded sample dataset from GitHub (test_data.csv)")
    except:
        st.error("❌ Failed to load sample dataset. Please upload manually.")

elif uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded dataset loaded successfully.")

else:
    st.info("⬅️ Upload a CSV file OR select sample dataset from sidebar to begin.")
    st.stop()

st.subheader("📌 Dataset Preview")
st.dataframe(data.head(10))

if "pha" not in data.columns:
    st.error("❌ Dataset must contain 'pha' column as target variable.")
    st.stop()

X = data.drop("pha", axis=1)
y = data["pha"]

if y.dtype == "object":
    y = y.map({"N": 0, "Y": 1})

X = X.reindex(columns=train_columns, fill_value=0)

X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_scaled)[:, 1]
else:
    y_proba = y_pred

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
mcc = matthews_corrcoef(y, y_pred)

try:
    auc = roc_auc_score(y, y_proba)
except:
    auc = 0.0

cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred, zero_division=0)

output_df = data.copy()
output_df["Predicted_PHA"] = y_pred

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview",
    "📊 Metrics",
    "📌 Confusion Matrix",
    "📄 Predictions",
    "📂 Dataset Info"
])

with tab1:
    st.subheader("📌 Project Overview")
    st.write(f"### ✅ Selected Model: **{model_name}**")

    st.markdown("""
This project predicts whether an asteroid is **Potentially Hazardous (PHA)** using machine learning models.

**PHA target meaning:**
- `0` → Not Hazardous  
- `1` → Hazardous  
    """)

    st.write("### 📌 Target Distribution")
    st.write(y.value_counts())

with tab2:
    st.subheader("📊 Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("AUC Score", f"{auc:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    st.markdown("---")
    st.subheader("📄 Classification Report")
    st.text(report)

with tab3:
    st.subheader("📌 Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")

    st.pyplot(fig)

with tab4:
    st.subheader("✅ Prediction Output")
    st.dataframe(output_df.head(30))

    st.markdown("### 📥 Download Predictions")
    st.download_button(
        label="Download Predictions CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions_output.csv",
        mime="text/csv"
    )

with tab5:
    st.subheader("📂 Dataset Information")
    st.write("### Shape of Dataset")
    st.write(data.shape)

    st.write("### Column Names")
    st.write(list(data.columns))

    st.write("### Missing Values")
    st.write(data.isnull().sum())

    st.write("### Feature Summary (Numerical)")
    st.dataframe(data.describe())

del X_scaled
gc.collect()
