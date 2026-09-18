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

df["has_event"] = df["event_name_1"].notna().astype("int8")
print(df["has_event"].value_counts())
