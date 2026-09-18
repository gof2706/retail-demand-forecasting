import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_parquet("data/sales_calendar.parquet")
df = df.sort_values(["id", "date"])

df["sales_lag_7"] = df.groupby("id")["sales"].shift(7)

# print(df[["id", "date", "sales", "sales_lag_7"]].head(30))
# hobbies = df[df["id"] == "HOBBIES_1_001_CA_1_validation"]
# print(hobbies[["date", "sales", "sales_lag_7"]].head(10))

valid = df.dropna(subset=["sales_lag_7"])

mae = mean_absolute_error(valid["sales"], valid["sales_lag_7"])
print("MAE baseline (lag-7):", mae) #1.2453109468968215
rmse = root_mean_squared_error(valid["sales"], valid["sales_lag_7"])
print("RMSE baseline (lag-7):", rmse) #3.261744702710567
