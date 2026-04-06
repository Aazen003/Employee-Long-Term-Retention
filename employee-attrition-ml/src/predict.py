"""
Load the saved Optuna XGBoost pipeline and tuned threshold; score new employee rows.

Run from project root:
  python -m src.predict

Requires models/final_xgb_pipeline.joblib and models/final_threshold.joblib
(produced by the notebook save cells).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = PROJECT_ROOT / "models" / "final_xgb_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "final_threshold.joblib"

_pipeline = None
_threshold_meta: dict[str, Any] | None = None


def _load_artifacts() -> tuple[Any, float, dict[str, Any]]:
    global _pipeline, _threshold_meta
    if _pipeline is None:
        if not PIPELINE_PATH.is_file():
            raise FileNotFoundError(
                f"Missing {PIPELINE_PATH}. Train and run the notebook save cells first."
            )
        if not THRESHOLD_PATH.is_file():
            raise FileNotFoundError(
                f"Missing {THRESHOLD_PATH}. Train and run the notebook save cells first."
            )
        _pipeline = joblib.load(PIPELINE_PATH)
        _threshold_meta = joblib.load(THRESHOLD_PATH)
    assert _pipeline is not None and _threshold_meta is not None
    threshold = float(_threshold_meta["threshold"])
    return _pipeline, threshold, dict(_threshold_meta)


def predict_employee(record: dict[str, Any]) -> dict[str, Any]:
    """
    One row as a dict keyed like training columns (no target, no dropped ID columns).
    Returns probability of positive class, integer prediction (1 = attrition), and threshold used.
    """
    pipeline, threshold, meta = _load_artifacts()
    X = pd.DataFrame([record])
    proba = float(pipeline.predict_proba(X)[:, 1][0])
    pred = int(proba >= threshold)
    return {
        "attrition_probability": proba,
        "predicted_class": pred,
        "threshold": threshold,
        "model_name": meta.get("model_name", "unknown"),
    }


def _demo_sample() -> dict[str, Any]:
    return {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1100,
        "Department": "Sales",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 2,
        "Gender": "Male",
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 2,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 12000,
        "NumCompaniesWorked": 3,
        "OverTime": "Yes",
        "PercentSalaryHike": 12,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 4,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2
    }


def main() -> int:
    out = predict_employee(_demo_sample())
    print("Attrition probability:", out["attrition_probability"])
    print("Predicted class (1=attrition):", out["predicted_class"])
    print("Threshold used:", out["threshold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
