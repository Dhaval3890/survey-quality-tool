import pandas as pd
import numpy as np
import pingouin as pg
from factor_analyzer.factor_analyzer import calculate_kmo

def load_table_from_upload(uploaded_file):
    df = pd.read_csv(uploaded_file.file)
    return df

def basic_validation(df, likert_min=1, likert_max=5):
    issues = []

    # Likert range check
    out_of_bounds = df[(df < likert_min) | (df > likert_max)]
    if not out_of_bounds.empty:
        issues.append("Some values are outside the allowed Likert range.")

    # Missing data
    missing_ratio = df.isnull().mean().mean()
    if missing_ratio > 0:
        issues.append(f"Missing responses detected ({missing_ratio:.2%}).")

    return {
        "missing_ratio": missing_ratio,
        "out_of_bounds": int(out_of_bounds.size),
        "issues": issues or ["No quality issues detected"]
    }

def reliability_analysis(df):
    # Cronbach’s Alpha via Pingouin
    alpha_result = pg.cronbach_alpha(data=df)
    alpha = float(alpha_result[0])

    # KMO — suitability for factor analysis
    kmo_all, kmo_model = calculate_kmo(df)

    return {
        "cronbach_alpha": alpha,
        "kmo": float(kmo_model)
    }
