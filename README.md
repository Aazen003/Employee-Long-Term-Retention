# Employee Attrition Prediction

This project builds an end-to-end machine learning system for predicting employee attrition using tabular HR data from the IBM HR Analytics dataset. The goal is to move from raw employee records to a usable prediction service by covering the full workflow: data loading, exploratory analysis, preprocessing, model training, model comparison, imbalance handling, hyperparameter tuning, interpretation, artifact saving, and API-based inference.

The project is designed to demonstrate practical data science workflow rather than only notebook-level experimentation. In addition to baseline models, it includes threshold tuning, MLflow experiment tracking, SHAP-based interpretation, and a FastAPI endpoint for serving predictions.

## Project motivation

Employee attrition is a clear and business-relevant prediction problem. Many organizations care about retaining employees, reducing turnover costs, and identifying employees who may be at higher risk of leaving.

The motivation for this project was to build a complete machine learning pipeline for a realistic tabular classification problem and turn it into an end-to-end system. Instead of stopping at model training, the project carries the best model forward into a prediction module and API so that new employee records can be scored consistently using the saved preprocessing pipeline and trained model.

## Problem statement

Given a set of employee HR features, predict whether an employee is likely to leave the company.

- Problem type: Binary classification
- Target variable: `Attrition`
- Positive class: `1` = attrition
- Dataset: IBM HR Analytics Employee Attrition dataset
- Main focus: minority-class detection and practical model deployment

Because the dataset is imbalanced, overall accuracy is not treated as the primary metric. The project focuses more on minority-class detection quality, especially class 1 precision, recall, F1, ROC-AUC, and PR-AUC.

## End-to-end workflow

The project follows this pipeline:

1. Load the raw HR dataset
2. Perform exploratory data analysis and identify non-informative columns
3. Build preprocessing pipelines for numeric and categorical features
4. Train and compare baseline models
5. Handle class imbalance
6. Tune decision thresholds
7. Tune XGBoost hyperparameters with Optuna
8. Interpret the final model using feature importance and SHAP
9. Save the trained pipeline and threshold
10. Serve predictions through a FastAPI application

This is what makes the project end to end: the same fitted pipeline used in training is saved and reused during inference.

## Repository structure
````
```text
employee-attrition-ml/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── predict.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data_preprocessing.py
│   └── feature_engineering.py
├── README.md
├── requirements.txt
└── .gitignore
````

## Dataset and preprocessing

The dataset contains employee-level HR features, including demographic, compensation, satisfaction, travel, and work-history variables.

Obvious non-informative columns were removed early in the workflow:

* `EmployeeNumber`
* `EmployeeCount`
* `Over18`
* `StandardHours`

The remaining features were split into numeric and categorical groups and processed with a scikit-learn `ColumnTransformer`.

### Numeric preprocessing

* median imputation
* standard scaling

### Categorical preprocessing

* most frequent imputation
* one-hot encoding with `handle_unknown="ignore"`

The final trained model uses the fitted preprocessing pipeline together with the classifier, so inference uses exactly the same transformations as training.

## Baseline experiments

The project begins with baseline testing before moving to more advanced models. This is important because it shows what each modeling decision contributes.

### 1. Logistic Regression

A basic linear baseline. It achieved strong overall accuracy on the imbalanced dataset, but recall for attrition was weak. The model mostly learned to predict the majority class.

### 2. Balanced Logistic Regression

A Logistic Regression model with `class_weight="balanced"`. This improved recall for attrition by giving more importance to the minority class, although precision dropped because more employees were flagged as likely leavers.

### 3. Threshold tuning for Balanced Logistic Regression

The default threshold of 0.5 was treated as arbitrary. Threshold search was performed to optimize F1 for class 1. This improved the precision-recall tradeoff and produced a much stronger operating point than the default threshold.

### 4. Random Forest

A useful nonlinear tabular baseline. In this project, Random Forest did not outperform the tuned balanced logistic model on minority-class F1 for the train-test split used.

### 5. XGBoost with imbalance handling

XGBoost was trained with imbalance-aware settings, including `scale_pos_weight`, and then threshold-tuned in the same way as Logistic Regression for a fair comparison.

### 6. Optuna-tuned XGBoost

Optuna was used to tune XGBoost hyperparameters by maximizing cross-validated ROC-AUC on the training set only. The final fitted model was then evaluated on the held-out test set, and threshold tuning was applied again to select the final operating point.

### 7. MLflow tracking

MLflow was used to log runs, parameters, thresholds, and evaluation metrics for reproducibility and comparison across experiments.

## Best model used for prediction

The final production candidate is:

**Optuna-tuned XGBoost with threshold tuning**

This model was selected because it gave the best overall tradeoff for the attrition class among the tested models.

### Why this model won

Compared with the tuned balanced Logistic Regression baseline, the Optuna-tuned XGBoost model:

* achieved slightly higher class 1 F1
* improved class 1 recall
* reduced the number of missed attrition cases
* captured nonlinear feature interactions better than the linear baseline

A representative operating point from the notebook runs was:

* Threshold: around `0.35`
* Class 1 F1: around `0.51`
* Class 1 Recall: around `0.68`

These values may change slightly if the split or training run changes, but Optuna-tuned XGBoost is the current best model used in inference.

## Model comparison summary

Two models anchor the final story:

| Role             | Model                                             | Why it matters                                                                                       |
| ---------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Strong benchmark | Balanced Logistic Regression with tuned threshold | Simple, fast, interpretable, and much better on attrition detection than default Logistic Regression |
| Best final model | Optuna-tuned XGBoost with tuned threshold         | Best minority-class tradeoff in this project and strongest final candidate for prediction            |

The project shows that accuracy alone is not sufficient for imbalanced classification. More meaningful improvements came from class balancing, threshold tuning, and hyperparameter optimization.

## Evaluation focus

Because attrition is the minority class, the project does not optimize for accuracy alone.

Primary evaluation signals:

* ROC-AUC
* PR-AUC
* Precision for class 1
* Recall for class 1
* F1 for class 1

Threshold tuning was a major part of the workflow because the default 0.5 classification threshold was not necessarily the best operating point for attrition detection.

## Model interpretation

After selecting the final model, interpretation was performed using:

* XGBoost feature importance
* SHAP summary plots

These analyses were used to understand which features most strongly influenced predicted attrition risk.

### Main findings from SHAP analysis

The strongest contributors to higher predicted attrition risk included:

* overtime
* lower monthly income
* younger age
* more companies worked
* greater distance from home
* lower stock option level
* lower job and environment satisfaction
* weaker work-life balance
* frequent business travel
* longer time since last promotion

These findings describe model behavior, not proven causal relationships.

### Important caution

This project is predictive, not causal.

Correct phrasing:

* associated with higher predicted attrition risk
* contributed to the model's prediction

Incorrect phrasing:

* caused attrition
* made employees leave

HR decisions should not rely on model scores alone. Organizational policy, fairness, ethics, and human review still matter.

## Saved artifacts

After training and final model selection, the notebook saves the following files:

| File                               | Description                                      |
| ---------------------------------- | ------------------------------------------------ |
| `models/final_xgb_pipeline.joblib` | Full fitted preprocessing and modeling pipeline  |
| `models/final_threshold.joblib`    | Threshold metadata used for final classification |

These files are used by the local prediction module and FastAPI service.

## Inference

### Local prediction

A local prediction script is available in `src/predict.py`.

Run from project root:

```bash
python -m src.predict
```

This loads the saved model pipeline and tuned threshold, scores a sample employee record, and returns:

* attrition probability
* predicted class
* threshold used
* model name

### API inference

FastAPI is used to expose the model as a local prediction service.

Run from project root:

```bash
python -m uvicorn app.main:app --reload
```

Available endpoints:

* `GET /health`
  Returns a simple status check

* `POST /predict`
  Accepts a JSON payload with the same employee fields used in training and returns:

  * `attrition_probability`
  * `predicted_class`
  * `threshold`
  * `model_name`

Example response:

```json
{
  "attrition_probability": 0.6855533123016357,
  "predicted_class": 1,
  "threshold": 0.35,
  "model_name": "optuna_xgboost"
}
```

## How to run the project

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd employee-attrition-ml
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the IBM HR dataset CSV inside:

```text
data/raw/
```

### 5. Run the notebook

Open and run:

```text
notebooks/01_eda.ipynb
```

This notebook covers:

* data inspection
* preprocessing
* baseline experiments
* threshold tuning
* XGBoost and Optuna tuning
* MLflow logging
* SHAP interpretation
* saving final artifacts

### 6. Run local prediction

```bash
python -m src.predict
```

### 7. Run the API

```bash
python -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Technologies used

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* Optuna
* MLflow
* SHAP
* FastAPI
* joblib
* matplotlib

## Future improvements

Possible next steps:

* add a validation split dedicated to threshold selection
* add Docker support
* deploy the API
* build a Streamlit frontend
* add fairness analysis for sensitive HR-related features
* add stronger configuration and experiment reproducibility controls

## Final takeaway

This project demonstrates a full applied machine learning workflow for employee attrition prediction. It starts with raw HR data, builds and compares multiple models, handles class imbalance properly, uses threshold tuning and Optuna for optimization, interprets the final model with SHAP, and serves predictions through FastAPI.

The final prediction system uses an Optuna-tuned XGBoost model because it provided the strongest attrition-detection tradeoff among the tested approaches.

```
```
