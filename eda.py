import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/sales_calendar.parquet")
daily_sales = df.groupby("date")["sales"].sum()
daily_sales.plot(figsize=(15, 5), title="Daily Sales")

# zero_days = daily_sales[daily_sales == 0]
# print(zero_days)
print(daily_sales.sort_values().head(10))

weekday_sales = df.groupby("wday")["sales"].mean()
print(weekday_sales)
# weekday_sales.plot(kind="bar", title="Average Sales by Weekday")

snap_effect = df.groupby("snap_CA")["sales"].mean()
print(snap_effect)
# snap_effect.plot(kind="bar", title="Average Sales by SNAP in CA")

category_sales = df.groupby("cat_id")["sales"].sum()
print(category_sales)
# category_sales.plot(kind="bar", title="Total Sales by Category")
# plt.show()