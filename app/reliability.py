import pandas as pd
import numpy as np
from typing import Dict


def cronbach_alpha(df: pd.DataFrame) -> float:
    df = df.dropna(axis=0, how="any")
    k = df.shape[1]
    if k < 2:
        return float("nan")
    item_var = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    alpha = (k / (k - 1.0)) * (1.0 - item_var.sum() / total_var)
    return float(alpha)


def reliability_report(df: pd.DataFrame) -> Dict[str, float]:
    numeric = df.select_dtypes(include=["number"])
    cols = numeric.columns.tolist()
    result: Dict[str, float] = {}
    if len(cols) >= 2:
        result["all_numeric"] = cronbach_alpha(numeric)
    return result
