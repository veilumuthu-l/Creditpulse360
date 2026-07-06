"""
Trains the CreditPulse360 scoring model on the synthetic dataset.

Model: XGBoost binary classifier predicting P(default within 12 months).
Output: a 300-900 "Financial Health Score" where higher = lower risk,
        computed as score = 900 - probability_of_default * 600.

Also saves a SHAP TreeExplainer so the app can generate real, model-derived
reason codes for every score (not hardcoded text).
"""

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "gst_filing_regularity", "gst_turnover_growth", "gst_itc_mismatch_rate",
    "upi_avg_daily_inflow", "upi_inflow_volatility", "upi_customer_concentration",
    "aa_avg_balance", "aa_bounce_rate", "aa_existing_emi_ratio",
    "bureau_thin_signal", "vintage_months",
]

FEATURE_LABELS = {
    "gst_filing_regularity": "GST filing regularity",
    "gst_turnover_growth": "GST turnover growth (YoY)",
    "gst_itc_mismatch_rate": "GST input-tax mismatch rate",
    "upi_avg_daily_inflow": "Avg. daily UPI/bank inflow",
    "upi_inflow_volatility": "Cash-flow volatility",
    "upi_customer_concentration": "Customer concentration (top-3 payers)",
    "aa_avg_balance": "Average bank balance",
    "aa_bounce_rate": "Cheque/mandate bounce rate",
    "aa_existing_emi_ratio": "Existing EMI-to-inflow ratio",
    "bureau_thin_signal": "Bureau footprint signal",
    "vintage_months": "Business vintage (months)",
}


def main():
    df = pd.read_csv("data/msme_synthetic_dataset.csv")
    X = df[FEATURES]
    y = df["default_12m"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Train AUC: {train_auc:.3f}  |  Test AUC: {test_auc:.3f}")

    explainer = shap.TreeExplainer(model)

    joblib.dump(
        {
            "model": model,
            "explainer": explainer,
            "features": FEATURES,
            "feature_labels": FEATURE_LABELS,
            "test_auc": test_auc,
        },
        "model/creditpulse_model.pkl",
    )
    print("Saved model bundle -> model/creditpulse_model.pkl")


if __name__ == "__main__":
    main()
