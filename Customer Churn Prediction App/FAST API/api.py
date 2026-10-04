import glob
import json
import os
from functools import lru_cache
from typing import Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADED_MODEL_NAME = "Uploaded Model (modelll.pkl)"
DEFAULT_MODEL = "Random Forest (Tuned)"


NOTEBOOK_METRICS = {
    "Logistic Regression": {"Accuracy": 0.8723, "Precision": 0.9992, "Recall": 0.8714,
                            "F1-score": 0.9310, "ROC-AUC": 0.9628},
    "Decision Tree": {"Accuracy": 0.9990, "Precision": 0.9992, "Recall": 0.9998,
                      "F1-score": 0.9995, "ROC-AUC": 0.9676},
    "Random Forest": {"Accuracy": 0.9965, "Precision": 0.9965, "Recall": 1.0000,
                      "F1-score": 0.9982, "ROC-AUC": 0.9997},
    "Random Forest (Tuned)": {"Accuracy": 0.9979, "Precision": 0.9983, "Recall": 0.9996,
                              "F1-score": 0.9989, "ROC-AUC": 0.9997},
}
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn, with probability and feature importance.",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[35])
    gender: Literal["Male", "Female"] = Field(..., examples=["Male"])
    tenure: int = Field(..., ge=0, le=120, description="Months as a customer", examples=[24])
    usage_frequency: int = Field(..., ge=0, le=100, examples=[15])
    support_calls: int = Field(..., ge=0, le=50, examples=[3])
    payment_delay: int = Field(..., ge=0, le=100, description="Days", examples=[5])
    subscription_type: Literal["Basic", "Standard", "Premium"] = Field(..., examples=["Basic"])
    contract_length: Literal["Monthly", "Quarterly", "Annual"] = Field(..., examples=["Monthly"])
    total_spend: float = Field(..., ge=0, examples=[500.0])
    last_interaction: int = Field(..., ge=0, le=100, description="Days since last interaction", examples=[10])


class PredictionResponse(BaseModel):
    model: str
    prediction: int = Field(..., description="1 = churn, 0 = retained")
    label: str
    churn_probability: float
    retain_probability: float
    risk_level: str
    risk_factors: List[str]


class BatchRequest(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, max_length=1000)


def available_models() -> Dict[str, str]:
    found: Dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(BASE, "models", "*.pkl"))):
        name = os.path.basename(p)[:-4].replace("_", " ").replace("Random Forest Tuned", "Random Forest (Tuned)")
        found[name] = p
    uploaded = os.path.join(BASE, "modelll.pkl")
    if os.path.exists(uploaded):
        found[UPLOADED_MODEL_NAME] = uploaded
    return found


@lru_cache(maxsize=8)
def _load(path: str):
    return joblib.load(path)


def get_model(name: Optional[str]):
    models = available_models()
    if not models:
        raise HTTPException(status_code=503, detail="No model files found. Place modelll.pkl next to api.py.")
    if name is None:
        name = DEFAULT_MODEL if DEFAULT_MODEL in models else next(iter(models))
    if name not in models:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'. Available: {list(models)}")
    try:
        return name, _load(models[name])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model '{name}': {e}. Install scikit-learn==1.6.1.",
        )


def to_frame(items: List[CustomerInput]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Age": c.age, "Gender": c.gender, "Tenure": c.tenure, "Usage Frequency": c.usage_frequency,
        "Support Calls": c.support_calls, "Payment Delay": c.payment_delay,
        "Subscription Type": c.subscription_type, "Contract Length": c.contract_length,
        "Total Spend": c.total_spend, "Last Interaction": c.last_interaction,
    } for c in items])


def risk_level(p: float) -> str:
    return "Low" if p < 0.4 else ("Medium" if p < 0.7 else "High")


def risk_factors(c: CustomerInput) -> List[str]:
    f = []
    if c.contract_length == "Monthly":
        f.append("Monthly contract: monthly subscribers showed the highest churn in the EDA.")
    if c.support_calls >= 6:
        f.append("High number of support calls: strongest positive correlation with churn.")
    if c.payment_delay >= 15:
        f.append("Long payment delay: positive correlation with churn.")
    if c.total_spend < 400:
        f.append("Low total spend: higher-spending customers are retained more often.")
    if c.tenure < 12:
        f.append("Short tenure: newer customers have a higher risk of leaving.")
    return f or ["No major risk factors detected."]


def run_prediction(name: str, pipe, items: List[CustomerInput]) -> List[PredictionResponse]:
    X = to_frame(items)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)
    churn_idx = list(pipe.classes_).index(1)
    results = []
    for item, pred, p in zip(items, preds, probs):
        churn_p = float(p[churn_idx])
        results.append(PredictionResponse(
            model=name,
            prediction=int(pred),
            label="Churn" if int(pred) == 1 else "Retained",
            churn_probability=round(churn_p, 4),
            retain_probability=round(1 - churn_p, 4),
            risk_level=risk_level(churn_p),
            risk_factors=risk_factors(item),
        ))
    return results


def importance_series(pipe) -> Optional[pd.Series]:
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


@app.get("/", tags=["General"])
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}


@app.get("/health", tags=["General"])
def health():
    return {"status": "ok", "models_available": len(available_models())}


@app.get("/models", tags=["Models"])
def list_models():
    models = list(available_models())
    return {"models": models, "default": DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput, model: Optional[str] = Query(None, description="Model name from /models")):
    name, pipe = get_model(model)
    return run_prediction(name, pipe, [customer])[0]


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
def predict_batch(req: BatchRequest, model: Optional[str] = Query(None, description="Model name from /models")):
    name, pipe = get_model(model)
    return run_prediction(name, pipe, req.customers)


@app.get("/feature-importance", tags=["Models"])
def feature_importance(model: Optional[str] = Query(None), top: int = Query(10, ge=1, le=50)):
    name, pipe = get_model(model)
    saved_path = os.path.join(BASE, "models", "metrics.json")
    if os.path.exists(saved_path):
        with open(saved_path) as f:
            saved = json.load(f).get("importance", {}).get(name)
        if saved:
            s = pd.Series(saved).sort_values(ascending=False)
            return {"model": name, "features": {k: round(float(v), 6) for k, v in s.head(top).items()}}
    s = importance_series(pipe)
    if s is None:
        raise HTTPException(status_code=400, detail=f"Feature importance is not available for '{name}'.")
    return {"model": name, "features": {k: round(float(v), 6) for k, v in s.head(top).items()}}


@app.get("/metrics", tags=["Models"])
def metrics(model: Optional[str] = Query(None, description="Leave empty to get all models")):
    path = os.path.join(BASE, "models", "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)["metrics"]
        data = {k: {m: v[m] for m in METRIC_KEYS} for k, v in raw.items()}
    else:
        data = NOTEBOOK_METRICS
    if model is None:
        return data
    if model not in data:
        raise HTTPException(status_code=404, detail=f"No metrics for '{model}'. Available: {list(data)}")
    return {model: data[model]}