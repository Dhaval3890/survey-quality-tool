import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Any, List


def load_table_from_upload(data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(data))
    return pd.read_excel(BytesIO(data))


def basic_validation(
    df: pd.DataFrame,
    likert_min: float,
    likert_max: float,
    id_cols: List[str],
    straight_line_threshold: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    # drop ID columns from analysis (keep them in original df)
    analysis_df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

    numeric_cols = analysis_df.select_dtypes(include=["number"]).columns.tolist()
    result["n_rows"] = int(df.shape[0])
    result["n_cols"] = int(df.shape[1])
    result["numeric_columns"] = numeric_cols

    # missing %
    result["missing_pct"] = {
        col: float(df[col].isna().mean() * 100.0) for col in df.columns
    }

    # out-of-range counts
    oor_counts: Dict[str, int] = {}
    for col in numeric_cols:
        s = analysis_df[col]
        oor = ((s < likert_min) | (s > likert_max)).sum()
        oor_counts[col] = int(oor)
    result["out_of_range_counts"] = oor_counts

    # straight-lining flags (row indices)
    flags: List[int] = []
    values = analysis_df[numeric_cols].to_numpy(dtype=float)
    for i, row in enumerate(values):
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            continue
        _, counts = np.unique(valid, return_counts=True)
        if counts.max() >= straight_line_threshold:
            flags.append(i)
    result["straight_line_flags"] = flags

    return result
