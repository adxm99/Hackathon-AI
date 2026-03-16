import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

DATA_PATH = Path("data/processed/hr_model_ready.csv")
MODEL_DIR = Path("models")
TARGET = "Termd"

# sensitive columns kept for fairness audit but excluded from model training
SENSITIVE_COLS = ["Sex", "RaceDesc", "HispanicLatino"]


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def main():
    df = pd.read_csv(DATA_PATH)

    fairness_df = df[SENSITIVE_COLS].copy()

    # remove sensitive cols from training
    features_to_drop = [TARGET] + [c for c in SENSITIVE_COLS if c in df.columns]
    X = df.drop(columns=features_to_drop, errors="ignore")
    y = df[TARGET]

    preprocessor = build_preprocessor(X)

    logistic_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test, fair_train, fair_test = train_test_split(
        X, y, fairness_df, test_size=0.25, stratify=y, random_state=42
    )

    logistic_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    results = [
        evaluate_model("logistic_regression", logistic_pipeline, X_test, y_test),
        evaluate_model("random_forest", rf_pipeline, X_test, y_test),
    ]

    results_df = pd.DataFrame(results)
    print(results_df)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(logistic_pipeline, MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(rf_pipeline, MODEL_DIR / "random_forest.joblib")

    eval_df = X_test.copy()
    eval_df[TARGET] = y_test.values

    eval_df["pred_logit"] = logistic_pipeline.predict(X_test)
    eval_df["proba_logit"] = logistic_pipeline.predict_proba(X_test)[:, 1]

    eval_df["pred_rf"] = rf_pipeline.predict(X_test)
    eval_df["proba_rf"] = rf_pipeline.predict_proba(X_test)[:, 1]

    for col in fair_test.columns:
        eval_df[col] = fair_test[col].values

    eval_df.to_csv(MODEL_DIR / "evaluation_dataset.csv", index=False)
    results_df.to_csv(MODEL_DIR / "model_metrics.csv", index=False)


if __name__ == "__main__":
    main()