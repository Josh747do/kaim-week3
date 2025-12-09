import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def load_data(path: str):
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df["bmi_group"] = np.where(df["bmi"] >= 30, "high_bmi", "normal_bmi")

    X = df.drop("charges", axis=1)
    y = df["charges"]

    cat_cols = ["sex", "smoker", "region", "bmi_group"]
    num_cols = ["age", "bmi", "children"]

    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first"), cat_cols),
            ("numerical", "passthrough", num_cols),
        ]
    )

    return X, y, transformer


def build_models(transformer):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }

    pipelines = {
        name: Pipeline([("transform", transformer), ("model", model)])
        for name, model in models.items()
    }

    return pipelines


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }


def train_and_evaluate_all(df):
    X, y, transformer = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipelines = build_models(transformer)

    results = {}

    for name, pipe in pipelines.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        results[name] = evaluate_model(pipe, X_test, y_test)

    return results
