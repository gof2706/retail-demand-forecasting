import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_parquet("data/sales_calendar.parquet")
df = df.sort_values(["id", "date"])

lag_days = [1, 7, 14, 28]
for lag in lag_days:
    df[f"sales_lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

df["sales_roll_mean_7"] = (
    df.groupby("id")["sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

conditions = [
    df["state_id"] == "CA",
    df["state_id"] == "TX",
    df["state_id"] == "WI",
]
choices = [df["snap_CA"], df["snap_TX"], df["snap_WI"]]
df["snap"] = np.select(conditions, choices)

df = df.dropna(subset=["sales_lag_28"])
df["has_event"] = df["event_name_1"].notna().astype("int8")

split_date = df["date"].max() - pd.Timedelta(days=28)
test = df[df["date"] > split_date]

feature_cols = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7",
    "snap", "has_event",
    "sell_price", "wday", "month", "year",
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
]

X_test = test[feature_cols]
y_test = test["sales"]

model_lgb = joblib.load("models/model_lgb.pkl")
y_pred = model_lgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("LightGBM MAE:", mae)
print("LightGBM RMSE:", rmse)

baseline_mae = mean_absolute_error(test["sales"], test["sales_lag_7"])
baseline_rmse = mean_squared_error(test["sales"], test["sales_lag_7"]) ** 0.5
print("Baseline (lag-7) MAE on test:", baseline_mae)
print("Baseline (lag-7) RMSE on test:", baseline_rmse)
