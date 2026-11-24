import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

# ---- Data: snow + month features, train on older months, test on the latest seasons
df = pd.read_csv("vail_model_data.csv")
df["date"] = pd.to_datetime(df["month"])
df["year_index"] = (df["date"].dt.year - 2010) + (df["date"].dt.month - 1) / 12
snow_cols = ["new_snow", "avg_snow_depth", "peak_swe", "big_storm_days", "last_month_snow"]
months = pd.get_dummies(df["month_num"], prefix="m", dtype=int)
X = pd.concat([df[snow_cols], months, df[["covid", "year_index"]]], axis=1)
y = df["lodging_revenue"]

test = df["date"] >= "2024-05-01"
X_train, X_test, y_train, y_test = X[~test], X[test], y[~test], y[test]

cv = TimeSeriesSplit(n_splits=5)
def scaled_y(model):
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())

# ---- Model 1: Ridge (tune alpha)
ridge = GridSearchCV(
    Pipeline([("scale", StandardScaler()), ("m", scaled_y(Ridge()))]),
    {"m__regressor__alpha": np.logspace(-2, 3, 11)},
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

# ---- Model 2: Random Forest (tune trees, max depth, min samples per leaf)
rf = GridSearchCV(
    Pipeline([("m", RandomForestRegressor(random_state=42))]),
    {"m__n_estimators": [100, 300, 500], "m__max_depth": [3, 5, 8, None], "m__min_samples_leaf": [1, 2, 5]},
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

# ---- Model 3: SVR (tune C and gamma; features must be scaled)
svr = GridSearchCV(
    Pipeline([("scale", StandardScaler()), ("m", scaled_y(SVR(kernel="rbf")))]),
    {"m__regressor__C": [0.1, 1, 10, 100, 1000], "m__regressor__gamma": [0.001, 0.01, 0.1, "scale"]},
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

# ---- Fit and score on the test months
for name, model in {"Ridge": ridge, "Random Forest": rf, "SVR": svr}.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    print(f"{name:14s} R2 = {r2_score(y_test, p):.3f} | MAE = ${mean_absolute_error(y_test, p):,.0f} | best settings: {model.best_params_}")
