import pandas as pd
import numpy as np

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
train = df[df["date"] <= split_date]
test = df[df["date"] > split_date]


print(train.shape, test.shape)


feature_cols = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7",
    "snap", "has_event",
    "sell_price", "wday", "month", "year",
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
]

X_train = train[feature_cols]
y_train = train["sales"]
X_test = test[feature_cols]
y_test = test["sales"]

import lightgbm as lgb

categorical_features = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

model_lgb = lgb.LGBMRegressor(random_state=42)
model_lgb.fit(X_train, y_train, categorical_feature=categorical_features)

import joblib

joblib.dump(model_lgb, "models/model_lgb.pkl")



