# Time-Series Analysis & Forecasting Studio

A Streamlit app for exploring and forecasting time series from any CSV, TSV, or Excel file.

## Features

- **Flexible ingestion**: upload CSV, TSV, XLS, or XLSX files with any column layout. The app guesses the date column and lets you override it, infers the sampling frequency (daily/weekly/monthly/etc.), and handles missing timestamps or values (forward/backward fill, linear interpolation, mean).
- **Exploration (Tab 2)**: interactive line charts across multiple columns, summary statistics, ADF and KPSS stationarity tests, seasonal decomposition (additive/multiplicative), and ACF/PACF plots.
- **Forecasting (Tab 3)**: run any combination of three models on a train/holdout split, then optionally extend the forecast into the future:
  - **SARIMAX** — configurable (p,d,q)(P,D,Q,m) orders, with an ADF-based suggestion for `d`.
  - **Prophet** — configurable seasonality mode and changepoint flexibility.
  - **XGBoost** — lag + rolling-window + calendar features, recursive multi-step forecasting.
- **Comparison & export (Tab 4)**: side-by-side MAE / RMSE / MAPE / R² on the held-out period, XGBoost feature importances, and a CSV download of all models' forecasts.
- **Multivariate anomaly detection (Tab 5)**: pick any subset of numeric columns and flag points where the *combination* of values looks unusual — not just single-column spikes. Three complementary detectors:
  - **Isolation Forest** — tree-based, handles non-linear/non-Gaussian structure.
  - **PCA reconstruction error** — flags points that break the dominant correlation structure between the chosen columns.
  - **Robust covariance (Elliptic Envelope / Mahalanobis distance)** — flags points far from the joint center relative to the covariance structure (assumes a roughly elliptical/Gaussian joint distribution).

  Results are combined with an ensemble vote (flag a point if at least N of the methods agree), visualized as per-method score plots and an overlay of anomalies on the raw series, and exportable as CSV.

## Running locally

```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Running with Docker

```bash
docker build -t ts-forecast-studio .
docker run -p 8501:8501 ts-forecast-studio
```

Then open http://localhost:8501.

## Running with Docker Compose
 
```bash
docker compose up --build
```
 
Then open http://localhost:8501. To stop it: `docker compose down`.

## Workflow

1. **Upload & Setup**: upload your file, pick the date column and target series, confirm/override the frequency, choose how to handle missing values, and click "Apply configuration".
2. **Explore**: inspect trends, stationarity, seasonality, and autocorrelation.
3. **Forecast**: pick a holdout size, optionally add extra future periods, select models, tune their settings, and run.
4. **Compare & Export**: check accuracy metrics on the holdout period and download the forecasts as CSV.
5. **Anomaly Detection**: select the columns to analyze jointly, set an expected anomaly rate and which methods to run, then review the flagged points and export them.

## Notes

- SARIMAX and Prophet return confidence intervals; XGBoost currently returns point forecasts only.
- For SARIMAX, set the seasonal period `m` to `0` to fit a non-seasonal ARIMA model (faster, and fine for series without strong periodicity).
- The XGBoost lag features default to `1,2,3,7,14` (good for daily data with weekly patterns) — adjust accordingly (e.g. `1,2,3,12,24` for monthly data with yearly patterns).
- Very short series (fewer than ~2x the seasonal period) will not support seasonal decomposition or seasonal SARIMAX terms.
- Anomaly detection needs at least 2 columns to be genuinely multivariate (1 column falls back to univariate outlier detection); the Robust Covariance method also needs more rows than columns, and can fail if columns are near-perfectly correlated.
- The "expected anomaly rate" (contamination) is a rough prior for all three methods, not a guarantee — actual flagged counts will vary by method.




