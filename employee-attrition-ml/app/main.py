# """
# FastAPI service for attrition scoring.

# Run from project root (employee-attrition-ml):
#   set PYTHONPATH=.
#   uvicorn app.main:app --reload

# Or:
#   python -m uvicorn app.main:app --reload --app-dir .
# """

# from __future__ import annotations

# import sys
# from pathlib import Path
# from typing import Any

# from fastapi import FastAPI
# from pydantic import BaseModel, Field

# # Project root = parent of app/
# ROOT = Path(__file__).resolve().parent.parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from src.predict import predict_employee  # noqa: E402

# app = FastAPI(title="Employee attrition API", version="0.1.0")


# class EmployeePayload(BaseModel):
#     """Same fields as training matrix X (after dropping ID / constant columns)."""

#     Age: int
#     BusinessTravel: str
#     DailyRate: int
#     Department: str
#     DistanceFromHome: int
#     Education: int
#     EducationField: str
#     EnvironmentSatisfaction: int
#     Gender: str
#     HourlyRate: int
#     JobInvolvement: int
#     JobLevel: int
#     JobRole: str
#     JobSatisfaction: int
#     MaritalStatus: str
#     MonthlyIncome: int
#     MonthlyRate: int
#     NumCompaniesWorked: int
#     OverTime: str
#     PercentSalaryHike: int
#     PerformanceRating: int
#     RelationshipSatisfaction: int
#     StockOptionLevel: int
#     TotalWorkingYears: int
#     TrainingTimesLastYear: int
#     WorkLifeBalance: int
#     YearsAtCompany: int
#     YearsInCurrentRole: int
#     YearsSinceLastPromotion: int
#     YearsWithCurrManager: int


# class PredictResponse(BaseModel):
#     attrition_probability: float = Field(..., description="P(attrition | features)")
#     predicted_class: int = Field(..., description="1 if proba >= threshold else 0")
#     threshold: float
#     model_name: str


# @app.get("/health")
# def health() -> dict[str, str]:
#     return {"status": "ok"}


# @app.post("/predict", response_model=PredictResponse)
# def predict(payload: EmployeePayload) -> dict[str, Any]:
#     record = payload.model_dump()
#     return predict_employee(record)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_employee


app = FastAPI(
    title="Employee Attrition Prediction API",
    version="1.0.0",
    description="Predict employee attrition risk using the trained XGBoost pipeline.",
)


class EmployeeInput(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(employee: EmployeeInput) -> dict:
    result = predict_employee(employee.model_dump())
    return result