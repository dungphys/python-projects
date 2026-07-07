# ML Binary Classification App

A Streamlit application for end-to-end binary classification on **any** tabular
CSV file. Column names, types, and the target column are all discovered at
runtime — nothing is hardcoded.

## Features

- **Data Overview** — shape, dtypes, missing values, duplicates, descriptive stats
- **EDA** — target balance, missing-value chart, numeric/categorical distributions
  (split by target), correlation heatmap, boxplots
- **Modelling** — trains any combination of:
  - Decision Tree *(fit directly, no tuning — per spec)*
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (scikit-learn)
  - XGBoost
  - LightGBM
  - CatBoost

  All models except Decision Tree are tuned with `RandomizedSearchCV` under
  stratified k-fold cross-validation (folds, search iterations, and scoring
  metric are all configurable in the UI).
- **Evaluation** — leaderboard (accuracy, precision, recall, F1, ROC AUC,
  average precision), ROC curve comparison, confusion matrix, precision-recall
  curve
- **Feature Importance** — tree-based `feature_importances_` or logistic
  regression coefficients, shown as an interactive bar chart
- **Prediction** — upload a new CSV to batch-predict (with a downloadable
  results file), or fill in a manual single-record form

## Target detection

The target column just needs exactly two unique values. Common encodings are
handled automatically: `0/1`, `Yes/No`, `True/False`, `Churn/No Churn`, etc.
If your target uses a different pair of labels, it will still work — the
app falls back to an alphabetical 0/1 assignment and shows you the mapping
it used.

## Run locally without Docker

```bash
python -m venv venv
source venv/bin/activate       
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints (typically **http://localhost:8501**).

## Run with Docker
 
No local Python setup needed - just Docker.
 
**Build and run directly:**

```bash
docker build -t ml-binary-classifier .
docker run -p 8501:8501 ml-binary-classifier
```
 
**Or with Docker Compose:**

```bash
docker compose up --build
```
 
Then open **http://localhost:8501**.
 
Notes:
- The image installs `libgomp1` (required by LightGBM at runtime; XGBoost bundles its own copy) and runs as a non-root user.
- A `HEALTHCHECK` hits Streamlit's `/_stcore/health` endpoint, so `docker ps` / Compose will report the container as `healthy` once it's actually serving.
- To stop: `Ctrl+C`, then `docker compose down` if you used Compose.
- CSV files are uploaded through the browser at runtime, so no volume mount is required for normal use.


## Try it immediately

A synthetic sample dataset, `sample_customer_churn.csv`, is included
(customer churn: age, income, tenure, contract type, etc. → `churn` Yes/No).
Upload it in the sidebar to explore the app right away.

## Project structure

```
ml_binary_classifier_app/
├── app.py                    # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── data/
|   └── sample_customer_churn.csv # example dataset 
└── utils/
    ├── data_utils.py         # CSV loading, target/ID detection & encoding, overview stats
    ├── eda_utils.py          # Plotly EDA chart builders
    ├── preprocessing.py      # ColumnTransformer (impute/scale/one-hot), built from
    │                         #   whatever columns are found at runtime
    ├── model_utils.py        # Model registry, hyperparameter search spaces,
    │                         #   training (RandomizedSearchCV + CV), feature importance
    └── evaluation.py         # Metrics, confusion matrix, ROC/PR curves, leaderboard
```

## Notes

- If `xgboost`, `lightgbm`, or `catboost` aren't installed, the app detects
  this and simply skips those models with a note in the UI rather than
  crashing.
- Any column with as many unique values as there are rows (e.g. an ID column)
  is flagged in the sidebar and excluded from the default feature selection —
  you can still add it back in manually if you want.
- All preprocessing (imputation, scaling, one-hot encoding) is fit only on
  the training split and applied to the test split / new prediction data via
  the same `Pipeline`, avoiding data leakage.
