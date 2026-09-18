import pandas as pd
import numpy as np


def build_features(df):
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

    return df


FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7",
    "snap", "has_event",
    "sell_price", "wday", "month", "year",
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
]

CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
