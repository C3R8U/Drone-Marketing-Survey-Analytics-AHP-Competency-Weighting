#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TITLE:
    Drone Marketing Survey Analytics + AHP Competency Weighting (Runnable Reference Implementation)

WHAT THIS SCRIPT DOES:
    1) Loads the paper's raw survey dataset (CSV)
    2) Reproduces key descriptive statistics used in the Results section:
         - Figure 6 (binary questions)
         - Figure 7 (multi-select shopping channels)
         - Table 2 (marketing evaluation + multi-select improvement items)
    3) Runs the paper's competency-weighting algorithm using AHP (Analytic Hierarchy Process):
         - Primary indicator weights from the Figure 8 pairwise comparison matrix
         - Consistency metrics (lambda_max, CI, CR)
         - Global weights by combining primary AHP weights with Table 3 secondary weights

DEPENDENCIES:
    - Python 3.9+ recommended
    - pandas, numpy

INSTALL:
    pip install pandas numpy

RUN:
    Option A (CSV in the same folder):
        python paper_algorithm_run.py

    Option B (CSV path provided explicitly):
        python paper_algorithm_run.py /path/to/drone_marketing_raw_synthetic_dataset.csv

NOTES:
    - The script prints complete tables without truncation.
    - If you use a different dataset, make sure it contains the required columns.
"""

import os
import sys
import numpy as np
import pandas as pd


# ---------------------------
# Utility: AHP implementation
# ---------------------------

def ahp_weights_and_consistency(pairwise_matrix: np.ndarray):
    """
    Compute AHP weights using the principal eigenvector method,
    and compute CI/CR consistency metrics.

    Returns:
        weights (np.ndarray): normalized priority vector
        lambda_max (float): principal eigenvalue
        CI (float): consistency index
        CR (float): consistency ratio
    """
    A = np.array(pairwise_matrix, dtype=float)
    n = A.shape[0]

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_index = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues.real[max_index])

    principal_eigenvector = eigenvectors[:, max_index].real
    principal_eigenvector = np.abs(principal_eigenvector)  # ensure non-negative
    weights = principal_eigenvector / principal_eigenvector.sum()

    # Consistency Index (CI)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    # Random Index (RI) table (Saaty)
    RI_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }
    RI = float(RI_table.get(n, 1.49))  # fallback for n > 10
    CR = (CI / RI) if RI > 0 else 0.0

    return weights, lambda_max, CI, CR


def print_full_df(df: pd.DataFrame, title: str):
    """
    Print a dataframe without truncation.
    """
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", None
    ):
        print(df.to_string(index=False))


# -----------------------------------
# 1) Load the paper's raw survey data
# -----------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


# -------------------------------------------------------
# 2) Reproduce key descriptive results (Figure 6/7/Table2)
# -------------------------------------------------------

def reproduce_figure6(df: pd.DataFrame):
    """
    Figure 6 uses two binary variables:
        purchased_drone_before  (1=Yes, 0=No)
        intend_purchase_future  (1=Yes, 0=No)

    Output: counts + proportions (%)
    """
    def summarize_binary(col_name: str) -> pd.DataFrame:
        counts = df[col_name].value_counts(dropna=False).sort_index()
        total = int(counts.sum())
        yes_count = int(counts.get(1, 0))
        no_count = int(counts.get(0, 0))

        yes_pct = (yes_count / total) * 100 if total else 0.0
        no_pct = (no_count / total) * 100 if total else 0.0

        out = pd.DataFrame({
            "Response": ["Yes", "No"],
            "Count": [yes_count, no_count],
            "Proportion(%)": [round(yes_pct, 1), round(no_pct, 1)]
        })
        return out

    fig6_a = summarize_binary("purchased_drone_before")
    fig6_b = summarize_binary("intend_purchase_future")

    print_full_df(fig6_a, "Figure 6 (Part A): Purchased Drone Before")
    print_full_df(fig6_b, "Figure 6 (Part B): Intend to Purchase in the Future")


def reproduce_figure7(df: pd.DataFrame):
    """
    Figure 7 uses multi-select channel variables:
        channel_ecommerce
        channel_live_broadcasting
        channel_physical_store

    Output: counts + proportions (%), relative to N respondents
    """
    n = int(len(df))
    items = [
        ("E-commerce platform", "channel_ecommerce"),
        ("Live broadcasting platform", "channel_live_broadcasting"),
        ("Physical store", "channel_physical_store")
    ]

    rows = []
    for label, col in items:
        count = int(df[col].fillna(0).astype(int).sum())
        pct = (count / n) * 100 if n else 0.0
        rows.append([label, count, round(pct, 1)])

    out = pd.DataFrame(rows, columns=["Channel", "Count", "Proportion(%)"])
    print_full_df(out, "Figure 7: Main Shopping Channels (Multiple Choices Allowed)")


def reproduce_table2(df: pd.DataFrame):
    """
    Table 2 contains:
    (A) Promotional evaluation (single choice):
        promo_evaluation in {"Sufficient Promotion", "Insufficient Promotion"}

    (B) Improvement areas (multi-select):
        improve_unreasonable_pricing
        improve_insufficient_promotion
        improve_low_awareness
        improve_limited_sales_channels
        improve_inadequate_design_development
        improve_insufficient_pre_post_sales_service
    """
    n = int(len(df))

    # (A) Promo evaluation
    promo_counts = df["promo_evaluation"].value_counts()
    promo_rows = []
    for k in ["Sufficient Promotion", "Insufficient Promotion"]:
        c = int(promo_counts.get(k, 0))
        pct = (c / n) * 100 if n else 0.0
        promo_rows.append([k, c, f"{pct:.1f}%"])
    promo_out = pd.DataFrame(promo_rows, columns=["Evaluation Category", "Number of Respondents", "Proportion"])
    print_full_df(promo_out, "Table 2 (A): Respondents' Evaluation of Promotional Efforts")

    # (B) Improvement areas
    improvements = [
        ("Unreasonable Pricing", "improve_unreasonable_pricing"),
        ("Insufficient Promotional Efforts", "improve_insufficient_promotion"),
        ("Low Product Awareness", "improve_low_awareness"),
        ("Limited Sales Channels", "improve_limited_sales_channels"),
        ("Inadequate Product Design and Development", "improve_inadequate_design_development"),
        ("Insufficient Pre- and Post-Sales Services", "improve_insufficient_pre_post_sales_service"),
    ]

    imp_rows = []
    for label, col in improvements:
        c = int(df[col].fillna(0).astype(int).sum())
        pct = (c / n) * 100 if n else 0.0
        imp_rows.append([label, c, f"{pct:.1f}%"])

    imp_out = pd.DataFrame(imp_rows, columns=["Specific Issues", "Number of Respondents", "Proportion"])
    print_full_df(imp_out, "Table 2 (B): Perceived Areas for Improvement in Product Marketing (Multiple Choices Allowed)")


# ---------------------------------------
# 3) Run the competency-weight algorithm
# ---------------------------------------

def run_ahp_competency_algorithm():
    """
    Figure 8 pairwise comparison matrix among primary indicators:
        Personal traits
        Professional commitment
        Knowledge reserve
        Professional skills
        Cruising ability

    Then Table 3 secondary weights are used to compute global weights.
    """
    # (A) Pairwise matrix from Figure 8
    labels = ["Personal traits", "Professional commitment", "Knowledge reserve", "Professional skills", "Cruising ability"]
    A = np.array([
        [1.000, 0.386, 0.346, 0.376, 0.320],
        [2.591, 1.000, 0.693, 0.810, 0.628],
        [2.891, 1.440, 1.000, 1.015, 0.540],
        [2.659, 1.235, 0.985, 1.000, 0.496],
        [3.125, 1.593, 1.856, 2.015, 1.000],
    ], dtype=float)

    weights, lambda_max, CI, CR = ahp_weights_and_consistency(A)

    primary_df = pd.DataFrame({
        "Primary Indicator": labels,
        "AHP Weight (normalized)": np.round(weights, 6)
    })
    print_full_df(primary_df, "AHP Result: Primary Indicator Weights (from Figure 8 Pairwise Matrix)")

    print("\n" + "=" * 120)
    print("AHP Consistency Check")
    print("=" * 120)
    print(f"lambda_max = {lambda_max}")
    print(f"CI = {CI}")
    print(f"CR = {CR}")
    print("Interpretation: a common rule of thumb is CR < 0.10 for acceptable consistency.")

    # (B) Secondary weights from Table 3 (within each primary indicator)
    table3 = {
        "Personal traits": {
            "Self-control": 0.314,
            "Achievement": 0.132,
            "Execution Ability": 0.554
        },
        "Knowledge reserve": {
            "Aviation Theory Knowledge": 0.111,
            "Drone Knowledge": 0.251,
            "Flight Tactics": 0.398,
            "Flight Regulations": 0.240
        },
        "Cruising ability": {
            "Aerial Exploration": 0.182,
            "Safe Flying": 0.257,
            "Crisis Management": 0.346,
            "Ground-Air Coordination": 0.215
        },
        "Professional commitment": {
            "Professional Loyalty": 0.466,
            "Professional Belief": 0.199,
            "Professional Discipline": 0.335
        },
        "Professional skills": {
            "Flight Control": 0.137,
            "Information Integration": 0.315,
            "Mission Planning": 0.337,
            "Payload Usage": 0.211
        }
    }

    # Compute global weights = primary_weight * secondary_weight
    primary_weight_map = dict(zip(labels, weights))
    global_rows = []
    for primary, sec_dict in table3.items():
        pw = float(primary_weight_map[primary])
        for sec_name, sw in sec_dict.items():
            global_rows.append([primary, sec_name, pw, float(sw), pw * float(sw)])

    global_df = pd.DataFrame(
        global_rows,
        columns=[
            "Primary Indicator",
            "Secondary Indicator",
            "Primary Weight",
            "Secondary Weight (within Primary)",
            "Global Weight"
        ]
    )

    # Normalize global weights so that they sum to 1
    global_df["Global Weight (normalized)"] = global_df["Global Weight"] / global_df["Global Weight"].sum()

    global_df = global_df.sort_values("Global Weight (normalized)", ascending=False).reset_index(drop=True)
    global_df["Primary Weight"] = global_df["Primary Weight"].round(6)
    global_df["Secondary Weight (within Primary)"] = global_df["Secondary Weight (within Primary)"].round(6)
    global_df["Global Weight"] = global_df["Global Weight"].round(6)
    global_df["Global Weight (normalized)"] = global_df["Global Weight (normalized)"].round(6)

    print_full_df(global_df, "Competency Model: Global Weights (Primary AHP Weight × Secondary Weight)")

    return primary_df, global_df


# -------------------------
# Main entry point
# -------------------------

def main():
    default_csv = "drone_marketing_raw_synthetic_dataset.csv"
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

    print("\n" + "=" * 120)
    print("Loading dataset")
    print("=" * 120)
    print(f"CSV path: {csv_path}")

    df = load_dataset(csv_path)

    print("\n" + "=" * 120)
    print("Dataset overview")
    print("=" * 120)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    for c in df.columns:
        print(f" - {c}")

    reproduce_figure6(df)
    reproduce_figure7(df)
    reproduce_table2(df)

    run_ahp_competency_algorithm()

    print("\n" + "=" * 120)
    print("Done")
    print("=" * 120)


if __name__ == "__main__":
    main()
