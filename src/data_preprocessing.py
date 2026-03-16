import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/HRDataset_v14.csv")
PROCESSED_PATH = Path("data/processed/hr_model_ready.csv")

TARGET = "Termd"

ID_COLS = [
    "Employee_Name",
    "EmpID",
    "ManagerName",
    "ManagerID",
]

LEAKAGE_COLS = [
    "DateofTermination",
    "TermReason",
    "EmploymentStatus",
    "EmpStatusID",
]

DROP_EXTRA = [
    "Zip",
    "DOB",
    "DateofHire",
    "LastPerformanceReview_Date",
]

SENSITIVE_COLS = [
    "Sex",
    "RaceDesc",
    "HispanicLatino",
]

DATE_COLS = [
    "DOB",
    "DateofHire",
    "LastPerformanceReview_Date",
]

REFERENCE_DATE = pd.Timestamp("2019-01-01")


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip spaces from column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "DOB" in df.columns:
        df["Age"] = ((REFERENCE_DATE - df["DOB"]).dt.days / 365.25).round(1)

    if "DateofHire" in df.columns:
        df["TenureYears"] = ((REFERENCE_DATE - df["DateofHire"]).dt.days / 365.25).round(2)

    if "LastPerformanceReview_Date" in df.columns:
        df["DaysSinceLastReview"] = (REFERENCE_DATE - df["LastPerformanceReview_Date"]).dt.days

    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [c for c in ID_COLS + LEAKAGE_COLS + DROP_EXTRA if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    df = load_data()
    print("Raw shape:", df.shape)

    df = basic_cleaning(df)
    df = convert_dates(df)
    df = engineer_features(df)
    df = drop_unusable_columns(df)

    print("Processed shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    save_processed(df)
    print(f"\nSaved processed dataset to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()