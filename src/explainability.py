import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models/random_forest.joblib")
DATA_PATH = Path("data/processed/hr_model_ready.csv")
TARGET = "Termd"
SENSITIVE_COLS = ["Sex", "RaceDesc", "HispanicLatino"]


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET] + [c for c in SENSITIVE_COLS if c in df.columns], errors="ignore")

    # global feature importance if classifier supports it
    clf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False)

        print("Top 20 important features:")
        print(importances.head(20))
        importances.to_csv("models/feature_importance.csv", index=False)
    else:
        print("This classifier does not expose feature importances directly.")

    # example local prediction
    sample = X.iloc[[0]]
    proba = model.predict_proba(sample)[0, 1]
    pred = model.predict(sample)[0]

    print("\nExample employee prediction:")
    print(f"Predicted class: {pred}")
    print(f"Predicted attrition probability: {proba:.4f}")


if __name__ == "__main__":
    main()