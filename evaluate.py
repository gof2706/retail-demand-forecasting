import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engineering import FEATURE_COLS, CATEGORICAL_COLS, build_features

df = pd.read_parquet("data/sales_calendar.parquet")

sample_ids = np.load("models/sample_ids.npy", allow_pickle=True)
df = df[df["id"].isin(sample_ids)]

df = build_features(df)

split_date = df["date"].max() - pd.Timedelta(days=28)
test = df[df["date"] > split_date]

X_test = test[FEATURE_COLS]
y_test = test["sales"]

X_test_xgb = X_test.copy()
for col in CATEGORICAL_COLS:
    X_test_xgb[col] = X_test_xgb[col].astype("category")

models = {
    "LightGBM": (joblib.load("models/model_lgb.pkl"), X_test),
    "CatBoost": (joblib.load("models/model_cat.pkl"), X_test),
    "XGBoost": (joblib.load("models/model_xgb.pkl"), X_test_xgb),
}

for name, (model, X) in models.items():
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

baseline_mae = mean_absolute_error(test["sales"], test["sales_lag_7"])
baseline_rmse = mean_squared_error(test["sales"], test["sales_lag_7"]) ** 0.5
print(f"Baseline (lag-7): MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}")
