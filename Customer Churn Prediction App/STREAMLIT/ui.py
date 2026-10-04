import glob
import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
MODEL_DIRS = [os.path.join(BASE, "models"), BASE, os.path.join(ROOT, "MODEL FILE"), os.path.join(ROOT, "models")]
DATA_DIRS = [os.path.join(BASE, "data"), os.path.join(ROOT, "DATASET"), os.path.join(ROOT, "data")]

NOTEBOOK_METRICS = {
    "Logistic Regression": {"Accuracy": 0.8723, "Precision": 0.9992, "Recall": 0.8714, "F1-score": 0.9310,
                            "ROC-AUC": 0.9628, "cm": [[234, 14], [2612, 17706]]},
    "Decision Tree": {"Accuracy": 0.9990, "Precision": 0.9992, "Recall": 0.9998, "F1-score": 0.9995,
                      "ROC-AUC": 0.9676, "cm": [[232, 16], [4, 20314]]},
    "Random Forest": {"Accuracy": 0.9965, "Precision": 0.9965, "Recall": 1.0000, "F1-score": 0.9982,
                      "ROC-AUC": 0.9997, "cm": [[176, 72], [1, 20317]]},
    "Random Forest (Tuned)": {"Accuracy": 0.9979, "Precision": 0.9983, "Recall": 0.9996, "F1-score": 0.9989,
                              "ROC-AUC": 0.9997, "cm": None},
}
METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]


def available_models():
    found = {}
    for d in MODEL_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            base = os.path.basename(p)[:-4]
            if base == "modelll":
                name = "Uploaded Model (modelll.pkl)"
            else:
                name = base.replace("_", " ").replace("Random Forest Tuned", "Random Forest (Tuned)")
            found.setdefault(name, p)
    return found


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


def load_metrics():
    p = next((os.path.join(d, "metrics.json") for d in MODEL_DIRS
              if os.path.exists(os.path.join(d, "metrics.json"))), None)
    if p:
        with open(p) as f:
            d = json.load(f)
        m = {k: {**{c: v[c] for c in METRIC_COLS}, "cm": v.get("confusion_matrix")} for k, v in d["metrics"].items()}
        return m, d.get("importance", {})
    return NOTEBOOK_METRICS, {}


def get_importance(pipe, name, saved):
    if name in saved:
        return pd.Series(saved[name]).sort_values(ascending=False)
    try:
        m = pipe.named_steps["model"]
        names = [n.split("__", 1)[-1] for n in pipe.named_steps["transformers"].get_feature_names_out()]
        if hasattr(m, "feature_importances_"):
            return pd.Series(m.feature_importances_, index=names).sort_values(ascending=False)
        if hasattr(m, "coef_"):
            return pd.Series(np.abs(m.coef_[0]), index=names).sort_values(ascending=False)
    except Exception:
        pass
    return None


def risk_level(p):
    if p < 0.4:
        return "Low", "🟢"
    if p < 0.7:
        return "Medium", "🟡"
    return "High", "🔴"


def risk_factors(r):
    f = []
    if r["Contract Length"] == "Monthly":
        f.append("Monthly contract: monthly subscribers showed the highest churn in the EDA.")
    if r["Support Calls"] >= 6:
        f.append("High number of support calls: strongest positive correlation with churn.")
    if r["Payment Delay"] >= 15:
        f.append("Long payment delay: positive correlation with churn.")
    if r["Total Spend"] < 400:
        f.append("Low total spend: higher-spending customers are retained more often.")
    if r["Tenure"] < 12:
        f.append("Short tenure: newer customers have a higher risk of leaving.")
    return f or ["No major risk factors detected."]


def bar_importance(s, top=12):
    s = s.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(s.index, s.values, color="#4C78A8")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


st.sidebar.title("📉 Churn Predictor")
page = st.sidebar.radio("Navigation", ["🔮 Prediction", "📊 Data Visualization", "📈 Model Performance"])

models = available_models()
if not models:
    st.error("No model found. Place modelll.pkl in the MODEL FILE folder (or next to app.py).")
    st.stop()
default = "Random Forest (Tuned)" if "Random Forest (Tuned)" in models else list(models)[0]
model_name = st.sidebar.selectbox("Select model", list(models), index=list(models).index(default))
try:
    pipe = load_model(models[model_name])
except Exception as e:
    st.error(f"Could not load the model: {e}\n\nInstall scikit-learn==1.6.1 (the pkl was created with this version).")
    st.stop()

if page.startswith("🔮"):
    st.title("Customer Churn Prediction")
    st.caption("Enter the customer's details to estimate the risk of churn.")

    with st.form("form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.slider("Tenure (months)", 0, 72, 24)
            usage = st.slider("Usage Frequency", 0, 40, 15)
        with c2:
            calls = st.slider("Support Calls", 0, 15, 3)
            delay = st.slider("Payment Delay (days)", 0, 40, 5)
            last = st.slider("Last Interaction (days ago)", 0, 40, 10)
        with c3:
            sub = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            spend = st.number_input("Total Spend", 0.0, 5000.0, 500.0, step=10.0)
        go = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)

    if go:
        X = pd.DataFrame([{"Age": age, "Gender": gender, "Tenure": tenure, "Usage Frequency": usage,
                           "Support Calls": calls, "Payment Delay": delay, "Subscription Type": sub,
                           "Contract Length": contract, "Total Spend": spend, "Last Interaction": last}])
        pred = int(pipe.predict(X)[0])
        prob = float(pipe.predict_proba(X)[0][list(pipe.classes_).index(1)])
        level, icon = risk_level(prob)

        left, right = st.columns(2)
        with left:
            if pred == 1:
                st.error("⚠️ This customer is likely to churn")
            else:
                st.success("✅ This customer is likely to stay")
            m1, m2 = st.columns(2)
            m1.metric("Churn Probability", f"{prob * 100:.1f}%")
            m2.metric("Risk Level", f"{icon} {level}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Model: {model_name}")
        with right:
            st.subheader("About this prediction")
            for line in risk_factors(X.iloc[0]):
                st.write("•", line)

        imp = get_importance(pipe, model_name, load_metrics()[1])
        if imp is not None:
            with st.expander("Most important features for this model"):
                st.pyplot(bar_importance(imp))
        with st.expander("Input data"):
            st.dataframe(X, use_container_width=True)

elif page.startswith("📊"):
    st.title("Data Visualization")
    csvs = [f for d in DATA_DIRS for f in glob.glob(os.path.join(d, "*.csv"))]
    df = load_data(csvs[0]) if csvs else None
    if df is None:
        up = st.file_uploader("Upload the dataset CSV", type="csv")
        df = pd.read_csv(up) if up else None
    if df is None:
        st.info("A dataset is required for the charts (put it in the DATASET folder or upload it).")
        st.stop()

    df = df.dropna(subset=["Churn"])
    st.write(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(10), use_container_width=True)

    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Churn", data=df, hue="Churn", legend=False, palette="Set2", ax=ax)
    ax.set_title("Churn Count (0 = Retained, 1 = Churned)")
    a.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Contract Length", hue="Churn", data=df, palette="Set2", ax=ax)
    ax.set_title("Contract Length vs Churn")
    b.pyplot(fig)

    num = [c for c in df.select_dtypes("number").columns if c not in ("CustomerID", "Churn")]
    col = st.selectbox("Select a feature", num)
    c, d = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Churn", y=col, data=df, hue="Churn", legend=False, palette="Set2", ax=ax)
    ax.set_title(f"{col} by Churn")
    c.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue="Churn", kde=True, palette="Set2", ax=ax)
    ax.set_title(f"Distribution of {col}")
    d.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.select_dtypes("number").drop(columns=["CustomerID"], errors="ignore").corr(),
                annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

else:
    st.title("Model Performance")
    metrics, saved_imp = load_metrics()
    tbl = pd.DataFrame({k: {c: v[c] for c in METRIC_COLS} for k, v in metrics.items()}).T
    st.dataframe(tbl.style.format("{:.4f}").highlight_max(axis=0, color="#bbf7d0"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    tbl.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    st.warning("The churned class (1) dominates the dataset, so accuracy alone can be misleading. "
               "Also check the confusion matrix for the retained class.")
    pick = st.selectbox("Confusion matrix", [k for k, v in metrics.items() if v.get("cm")])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(np.array(metrics[pick]["cm"]), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(pick)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    imp = get_importance(pipe, model_name, saved_imp)
    if imp is not None:
        st.pyplot(bar_importance(imp))
    else:
        st.write("Feature importance is not available for this model (e.g. KNN).")
