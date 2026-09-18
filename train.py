import pandas as pd
import numpy as np
import joblib

from feature_engineering import CATEGORICAL_COLS, FEATURE_COLS, build_features

df = pd.read_parquet("data/sales_calendar.parquet")

rng = np.random.default_rng(42)
all_ids = df["id"].unique()
sample_ids = rng.choice(all_ids, size=int(len(all_ids) * 0.05), replace=False)
np.save("models/sample_ids.npy", sample_ids)
df = df[df["id"].isin(sample_ids)]

df = build_features(df)


split_date = df["date"].max() - pd.Timedelta(days=28)
train = df[df["date"] <= split_date]
test = df[df["date"] > split_date]


print(train.shape, test.shape)

X_train = train[FEATURE_COLS]
y_train = train["sales"]
X_test = test[FEATURE_COLS]
y_test = test["sales"]


# import lightgbm as lgb

# model_lgb = lgb.LGBMRegressor(random_state=42)
# model_lgb.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)
# joblib.dump(model_lgb, "models/model_lgb.pkl")


# from catboost import CatBoostRegressor

# model_cat = CatBoostRegressor(random_state=42, verbose=100)
# model_cat.fit(X_train, y_train, cat_features=CATEGORICAL_COLS)

# joblib.dump(model_cat, "models/model_cat.pkl")


# import xgboost as xgb

# X_train_xgb = X_train.copy()
# for col in CATEGORICAL_COLS:
#     X_train_xgb[col] = X_train_xgb[col].astype("category")

# X_test_xgb = X_test.copy()
# for col in CATEGORICAL_COLS:
#     X_test_xgb[col] = X_test_xgb[col].astype("category")

# model_xgb = xgb.XGBRegressor(random_state=42, enable_categorical=True, tree_method="hist")
# model_xgb.fit(X_train_xgb, y_train)
# joblib.dump(model_xgb, "models/model_xgb.pkl")


