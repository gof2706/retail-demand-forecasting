import pandas as pd

sales_train = pd.read_csv(
    "data/sales_train_validation.csv", 
    dtype={
        "id": "category",
        "item_id": "category",
        "dept_id": "category",
        "cat_id": "category",
        "store_id": "category",
        "state_id": "category",
        }
    )

sales_train = sales_train.melt(
            id_vars=
                [
                    "id", 
                    "item_id", 
                    "dept_id", 
                    "cat_id", 
                    "store_id", 
                    "state_id"
                 ], 
            var_name="d", 
            value_name="sales", 
            value_vars=[f"d_{i}" for i in range(1, 1914)]
        )

calendar = pd.read_csv("data/calendar.csv",
                       dtype={
                                "wday": "int8",
                                "month": "int8",
                                "year": "int16",
                                "snap_CA": "int8",
                                "snap_TX": "int8",
                                "snap_WI": "int8",
                                "weekday": "category",
                                "event_name_1": "category",
                                "event_type_1": "category",
                                "event_name_2": "category",
                                "event_type_2": "category",
                                "wm_yr_wk": "int16",
                               },
                       parse_dates=["date"],
                       )

merged = sales_train.merge(calendar, how="left", on="d")
merged = merged.astype({"sales": "int16", "d": "category"})

sell_prices = pd.read_csv(
    "data/sell_prices.csv",
    dtype={
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int16",
        "sell_price": "float32",
    },
)

merged = merged.merge(sell_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
merged = merged.dropna(subset=["sell_price"])


merged.to_parquet("data/sales_calendar.parquet")
print(merged.shape)
# print(merged["sell_price"].isna().sum())
# print(merged.head())

# print(merged.dtypes)
print(merged.memory_usage(deep=True).sum() / 1024**2, "MB")