import pandas as pd
from pathlib import Path

EVAL_PATH = Path("models/evaluation_dataset.csv")
OUTPUT_PATH = Path("models/fairness_metrics.csv")
TARGET = "Termd"
PRED = "pred_logit"


def compute_group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []

    for group_value, g in df.groupby(group_col):
        tp = ((g[TARGET] == 1) & (g[PRED] == 1)).sum()
        tn = ((g[TARGET] == 0) & (g[PRED] == 0)).sum()
        fp = ((g[TARGET] == 0) & (g[PRED] == 1)).sum()
        fn = ((g[TARGET] == 1) & (g[PRED] == 0)).sum()

        total = len(g)
        accuracy = (tp + tn) / total if total else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0

        rows.append({
            "group_column": group_col,
            "group_value": group_value,
            "count": total,
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
        })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(EVAL_PATH)

    all_results = []
    for group_col in ["Sex", "RaceDesc", "HispanicLatino"]:
        if group_col in df.columns:
            all_results.append(compute_group_metrics(df, group_col))

    fairness_df = pd.concat(all_results, ignore_index=True)
    fairness_df.to_csv(OUTPUT_PATH, index=False)
    print(fairness_df)


if __name__ == "__main__":
    main()