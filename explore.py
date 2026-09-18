import pandas as pd


sales_train = pd.read_csv("data/sales_train_validation.csv")

print(sales_train.head())
print(sales_train.shape)

calendar = pd.read_csv("data/calendar.csv")
print(calendar.head())

