from typing import Dict, Any

import numpy as np
import pandas as pd
import pingouin as pg
from factor_analyzer.factor_analyzer import calculate_kmo


def cronbach_alpha_pingouin(df: pd.DataFrame) -> float:
    """
    Wrapper around pingouin.cronbach_alpha.
    Returns NaN if it can't be computed.
    """
    if df.shape[1] < 2:
        return float("nan")

    # Drop rows with missing to keep pingouin happy
    clean = df.dropna(axis=0, how="any")
    if clean.shape[0] < 2:
        return float("nan")

    alpha, _ = pg.cronbach_alpha(clean)
    return float(alpha)


def kmo_index(df: pd.DataFrame) -> float:
    """
    Kaiser-Meyer-Olkin measure of sampling adequacy (0..1).
    Higher is better; usually >= 0.6 is acceptable.
    """
    # FactorAnalyzer expects no missing → simple impute with column means
    clean = df.copy()
    clean = clean.fillna(clean.mean(numeric_only=True))

    try:
        _, kmo_model = calculate_kmo(clean)
        return float(kmo_model)
    except Exception:
        # If anything goes wrong (e.g. singular matrix), return NaN
        return float("nan")


def reliability_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute reliability + factor-suitability metrics for all numeric items.

    Returns:
      {
        "n_numeric_items": int,
        "cronbach_alpha": float or None,
        "kmo": float or None
      }
    """
    numeric = df.select_dtypes(include=["number"])
    result: Dict[str, Any] = {
        "n_numeric_items": int(numeric.shape[1])
    }

    if numeric.shape[1] >= 2:
        alpha = cronbach_alpha_pingouin(numeric)
        kmo = kmo_index(numeric)

        # Let to_json_safe handle NaN → None,
        # but we also guard here just to be nice.
        result["cronbach_alpha"] = (
            float(alpha) if not np.isnan(alpha) else None
        )
        result["kmo"] = float(kmo) if not np.isnan(kmo) else None
    else:
        result["cronbach_alpha"] = None
        result["kmo"] = None

    return result
