import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, chi2_contingency


def load_data(path: str):
    """
    Loads CSV data with error handling.
    """
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("ERROR: File not found. Check the path.")
        return None
    except pd.errors.EmptyDataError:
        print("ERROR: File is empty.")
        return None


# ===============================
#  SEGMENTATION FUNCTIONS
# ===============================

def create_bmi_category(df: pd.DataFrame):
    """
    Adds BMI category column: overweight vs normal.
    """
    df = df.copy()
    df["bmi_group"] = np.where(df["bmi"] >= 30, "high_bmi", "normal_bmi")
    return df


def segment_groups(df: pd.DataFrame, col: str, group_a, group_b):
    """
    Returns two series for A/B comparison.
    """
    A = df[df[col] == group_a]["charges"]
    B = df[df[col] == group_b]["charges"]
    return A, B


# ===============================
#  STATISTICAL TESTS
# ===============================

def t_test_groups(group_a, group_b):
    """
    Runs an independent t-test.
    """
    t_stat, p_val = ttest_ind(group_a, group_b, equal_var=False)
    return t_stat, p_val


def anova_test(df: pd.DataFrame, col: str):
    """
    Runs ANOVA on charges across multiple groups (e.g., regions).
    """
    groups = [df[df[col] == g]["charges"] for g in df[col].unique()]
    f_stat, p_val = f_oneway(*groups)
    return f_stat, p_val


def chi_square_test(df: pd.DataFrame, col: str):
    """
    Chi-square test for categorical distributions.
    Example: region vs smoker distribution.
    """
    contingency = pd.crosstab(df[col], df["smoker"])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    return chi2, p_val, contingency, expected


# ===============================
#  INTERPRETATION HELPERS
# ===============================

def interpret_p_value(p_val: float, alpha: float = 0.05):
    """
    Basic interpretation function.
    """
    if p_val < alpha:
        return f"Reject H₀ (p = {p_val:.4f})"
    return f"Fail to reject H₀ (p = {p_val:.4f})"
