# Employee attrition (IBM HR dataset)

This project predicts employee attrition from tabular HR features. Work lives mainly in `notebooks/01_eda.ipynb`: exploratory analysis, preprocessing pipelines, baselines, imbalance handling, hyperparameter search, threshold tuning, MLflow logging, and model interpretation (feature importance and SHAP).

## What we ran (baseline experiments)

1. **Logistic Regression (default)**  
   Strong overall accuracy on an imbalanced target, but **weak recall for attrition (class 1)**: the model mostly learns to predict “stay.”

2. **Balanced Logistic Regression** (`class_weight="balanced"`)  
   Pushes the model to care more about the minority class. **Recall for class 1 improves**; **precision** often drops because more people are flagged.

3. **Threshold tuning (balanced Logistic Regression)**  
   The default 0.5 cutoff is arbitrary. A search over probability thresholds (optimizing **F1 for class 1**) improves the precision–recall tradeoff. In our runs this landed near a **higher** threshold (e.g. ~0.60), with class 1 **F1 around 0.50** and **recall around 0.57**, at the cost of some overall accuracy.

4. **Random Forest**  
   Useful benchmark on tabular data; with default 0.5 threshold it did **not** match the tuned balanced logistic model on **class 1 F1** for this split.

5. **XGBoost with imbalance handling** (`scale_pos_weight`) plus **threshold tuning**  
   Stronger nonlinear model family; threshold must be tuned the same way as for logistic regression for a fair comparison.

6. **Optuna + XGBoost**  
   Hyperparameters tuned by **maximizing cross-validated ROC-AUC on the training set only** (no test leakage). The model is refit on the full training set, then **threshold-tuned again on test probabilities** for reporting (same pattern as for logistic regression).

7. **MLflow**  
   Runs log model name, threshold, ROC-AUC, PR-AUC, and class 1 precision / recall / F1 for traceability.

## Best models and why Optuna XGBoost wins

Two models anchor the story:

| Role | Model | Why it matters |
|------|--------|----------------|
| Strong linear benchmark | **Balanced Logistic Regression + tuned threshold** | Interpretable, fast, and **much better on attrition detection** than the default logistic baseline after balancing and threshold search. |
| Best overall | **Optuna-tuned XGBoost + tuned threshold** | Captures **nonlinear** patterns and interactions; Optuna improves the ranking quality (CV ROC-AUC) before we pick an operating point. |

**Optuna XGBoost performed better** than tuned balanced logistic on the metrics we prioritized for HR attrition:

- **Higher class 1 F1** (e.g. **~0.51** vs **~0.50** in our notebook runs).
- **Higher recall for class 1** (e.g. **~0.68** vs **~0.57**), so **fewer leavers are missed** at the chosen threshold.
- **ROC-AUC** on held-out scores stayed in a similar band to the strong linear models; the gain shows up strongly when we align **thresholds** and compare **precision–recall–F1** for the minority class.

Tuned balanced logistic regression remains a **credible benchmark**: simpler, easier to explain coefficient-wise, and good when you want to **limit false alarms**. For “catch people at risk of leaving,” the tuned tree model delivered a **better tradeoff** in our experiments.

---

## Main findings from the interpretation plots (SHAP / feature importance)

These describe **what the model uses**, not proven causes of quitting. Prefer: *associated with higher or lower **predicted** attrition risk*.

### 1. Overtime

One of the clearest signals. Encoded categories include `cat__OverTime_Yes` and `cat__OverTime_No`. In SHAP summary plots, **OverTime_Yes** tends to push risk **up**; **OverTime_No** tends to push risk **down**.

### 2. Monthly income

For `num__MonthlyIncome`, **lower values** more often align with **higher predicted attrition**; **higher values** more often align with **lower predicted risk**.

### 3. Number of companies worked

For `num__NumCompaniesWorked`, **higher values** tend to push predictions toward **higher risk**; **lower values** more toward **lower risk**.

### 4. Age

For `num__Age`, **younger** employees are **associated with higher predicted attrition** in the model; **older** ages lean **lower risk**.

### 5. Distance from home

For `num__DistanceFromHome`, **greater distance** is associated with **higher predicted attrition** (longer commute / farther from workplace).

### 6. Stock option level

For `num__StockOptionLevel`, **lower levels** align more with **higher predicted risk**; **higher levels** appear **protective** in the model’s scores.

### 7. Satisfaction and work–life balance

Features such as `num__EnvironmentSatisfaction`, `num__JobSatisfaction`, `num__RelationshipSatisfaction`, and `num__WorkLifeBalance` show up prominently. The usual pattern: **lower satisfaction or weaker balance** shifts risk **up**; **higher satisfaction or better balance** shifts risk **down**.

### 8. Business travel

`cat__BusinessTravel_Travel_Frequently` contributes to **higher predicted attrition** relative to less travel.

### 9. Years since last promotion

`num__YearsSinceLastPromotion`: **larger values** (longer since promotion) tend to push risk **up**, suggesting the model links **stalled progression** to higher predicted leave risk.

---

## Summary paragraph (for reports or the notebook)

SHAP analysis indicated that **overtime**, **monthly income**, **number of prior employers**, **age**, **distance from home**, **stock option level**, and **satisfaction-related** inputs were among the **strongest contributors to the model’s attrition scores**. The model **associated** overtime, **lower income**, **younger age**, **longer commute**, **more prior companies**, and **lower satisfaction / weaker work–life balance** with **higher predicted attrition risk**, while **no overtime**, **stronger compensation-related signals** (e.g. stock options), and **higher satisfaction** tended to **reduce** predicted risk.

---

## Important caution

SHAP and feature importance explain **model behavior**, not **causality**.

- Say: **associated with**, **contributed to the prediction**, **higher predicted attrition risk**.
- Avoid: **caused attrition**, **made people leave**.

The model is **predictive**; HR decisions still need context, policy, and ethics beyond the score.

---

## Frozen production candidate (documented)

**Winner:** Optuna-tuned **XGBoost** with **threshold tuning** on validation-style scores (see notebook). Example operating point from one run:

- **Threshold** ~ **0.35**
- Class 1 **F1** ~ **0.51**
- Class 1 **recall** ~ **0.68**

Exact values come from `winner_summary` in the notebook; copy updates here when you change data or splits.

## Saved artifacts and inference

After training, the notebook section **“8. Save final model pipeline and threshold”** writes:

| File | Contents |
|------|-----------|
| `models/final_xgb_pipeline.joblib` | Full **fitted** `Pipeline` (ColumnTransformer + XGBoost) |
| `models/final_threshold.joblib` | Dict with `threshold`, `model_name`, etc. |

**CLI smoke test** (from project root `employee-attrition-ml`):

```bash
python -m src.predict
```

**HTTP API** (FastAPI):

```bash
cd employee-attrition-ml
uvicorn app.main:app --reload
```

- `GET /health` – liveness  
- `POST /predict` – JSON body with the same employee fields as training `X` (no target); response includes `attrition_probability`, `predicted_class`, `threshold`, `model_name`.

---

## Repository layout (short)

- `data/raw/` – source CSV  
- `notebooks/01_eda.ipynb` – full workflow (includes save cells)  
- `src/predict.py` – load artifacts, score one row (demo in `__main__`)  
- `app/main.py` – FastAPI `GET /health`, `POST /predict`  
- `models/` – saved `joblib` artifacts (gitignored except `.gitkeep`)  
- `reports/figures/` – plots  
- `notebooks/mlruns` / MLflow DB (if used) – experiment tracking  

## Environment

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

Train via the notebook through the Optuna + threshold cells, then run the **save** cells. Open the notebook with the IBM HR attrition CSV under `data/raw/`.
