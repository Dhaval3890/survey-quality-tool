from typing import Dict

import numpy as np
import pandas as pd


def cronbach_alpha(df: pd.DataFrame) -> float:
    # drop rows with any missing values
    df = df.dropna(axis=0, how="any")
    k = df.shape[1]
    if k < 2:
        return float("nan")

    item_var = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)

    if total_var == 0 or np.isnan(total_var):
        return float("nan")

    alpha = (k / (k - 1.0)) * (1.0 - item_var.sum() / total_var)
    return float(alpha)


def reliability_report(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Cronbach's alpha for all numeric columns.
    Returns JSON-safe values (NaN -> None).
    """
    numeric = df.select_dtypes(include=["number"])
    cols = numeric.columns.tolist()
    result: Dict[str, float] = {}

    if len(cols) >= 2:
        alpha = cronbach_alpha(numeric)
        if np.isnan(alpha):
            result["all_numeric"] = None
        else:
            result["all_numeric"] = float(round(alpha, 4))

    return result
