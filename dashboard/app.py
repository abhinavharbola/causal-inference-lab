import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.db import fetch_estimation_runs
from src.utils.logging_config import configure_logging
from src.utils.power_analysis import mde_comparison_table

configure_logging()

DATA_DIR = "data/processed"

st.set_page_config(page_title="Causal Impact & Heterogeneous Response Analysis", layout="wide")


def load_json(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def missing_data_notice(what: str, notebook: str):
    st.info(f"{what} not found yet. Run {notebook} to generate it.")


st.title("Causal Impact & Heterogeneous Response Analysis")
st.caption("Criteo Uplift Modeling Dataset (v2.1) — validation, heterogeneity, and sensitivity analysis")

tab1, tab1_5, tab2, tab3, tab4 = st.tabs(
    [
        "1. Validation",
        "1.5 Outcome Justification",
        "2. Heterogeneity",
        "3. Statistical Rigor",
        "4. Sensitivity",
    ]
)

# ---------------------------------------------------------------------------
# Section 1: Validation via self-induced confounding
# ---------------------------------------------------------------------------
with tab1:
    st.header("Bias-Severity Curve")
    st.write(
        "Naive OLS, PSM, IPW, and AIPW estimates across confounding severities, "
        "compared against the ground-truth ATE from the full randomized dataset."
    )

    ground_truth = load_json("ground_truth.json")
    runs_df = fetch_estimation_runs()

    if runs_df is None or len(runs_df) == 0:
        missing_data_notice("Estimation run log", "notebooks/01_validation.ipynb")
    else:
        severity_order = ["none", "mild", "moderate", "strong"]
        present_severities = [s for s in severity_order if s in runs_df["severity_label"].unique()]

        fig, ax = plt.subplots(figsize=(8, 5))

        for method, group in runs_df.groupby("method"):
            group = group.set_index("severity_label").reindex(present_severities).reset_index()
            group = group.dropna(subset=["point_estimate"])
            x = range(len(group))
            ax.errorbar(
                x,
                group["point_estimate"],
                yerr=[
                    group["point_estimate"] - group["ci_lower"],
                    group["ci_upper"] - group["point_estimate"],
                ],
                marker="o",
                label=method,
                capsize=3,
            )

        if ground_truth is not None:
            ax.axhline(ground_truth["ate"], color="black", linestyle="--", label="Ground truth ATE")
            ax.axhspan(ground_truth["ci_lower"], ground_truth["ci_upper"], color="gray", alpha=0.2)

        ax.set_xticks(range(len(present_severities)))
        ax.set_xticklabels(present_severities)
        ax.set_xlabel("Confounding severity")
        ax.set_ylabel("Estimated ATE")
        ax.set_title("Estimator bias across confounding severity")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)
        st.dataframe(runs_df[["method", "severity_label", "g2", "point_estimate", "ci_lower", "ci_upper"]])

    st.subheader("Covariate Balance (matching diagnostics)")
    balance_df = load_csv("balance_table.csv")
    if balance_df is None:
        missing_data_notice("Balance table", "notebooks/01_validation.ipynb")
    else:
        st.dataframe(balance_df)

# ---------------------------------------------------------------------------
# Section 1.5: Outcome variable justification
# ---------------------------------------------------------------------------
with tab1_5:
    st.header("Outcome Variable Justification (MDE)")
    st.write(
        "Minimum detectable effect for `visit` vs `conversion` at the planned Section 2 "
        "subsample size. This is computed live since it's cheap; no precomputed file needed."
    )

    col1, col2, col3 = st.columns(3)
    visit_rate = col1.number_input("visit base rate", value=0.045, format="%.4f")
    conversion_rate = col2.number_input("conversion base rate", value=0.003, format="%.4f")
    n_per_group = col3.number_input("planned n per group", value=100_000, step=10_000)

    table = mde_comparison_table({"visit": visit_rate, "conversion": conversion_rate}, int(n_per_group))
    st.dataframe(table)
    st.caption(
        "conversion typically requires a much larger relative effect to be detectable at the same "
        "sample size, which is why it is excluded from segment-level work (Section 2/3) and used only "
        "for the full-dataset ground-truth ATE in Section 1."
    )

# ---------------------------------------------------------------------------
# Section 2: Heterogeneity / segmentation
# ---------------------------------------------------------------------------
with tab2:
    st.header("Segment-Level CATE Breakdown")
    segment_df = load_csv("segment_effects.csv")
    if segment_df is None:
        missing_data_notice("Segment effects table", "notebooks/02_heterogeneity.ipynb")
    else:
        st.dataframe(segment_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(len(segment_df))
        ax.errorbar(
            segment_df["point_estimate"],
            y_pos,
            xerr=[
                segment_df["point_estimate"] - segment_df["ci_lower"],
                segment_df["ci_upper"] - segment_df["point_estimate"],
            ],
            fmt="o",
            capsize=3,
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(segment_df["segment"])
        ax.set_xlabel("Estimated treatment effect")
        ax.set_title("Per-segment treatment effect with bootstrap CI")
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Qini Curve (CATE model quality)")
    qini_result = load_json("qini_curve.json")
    if qini_result is None:
        missing_data_notice("Qini curve", "notebooks/02_heterogeneity.ipynb")
    else:
        st.metric("Qini coefficient", f"{qini_result['qini_coefficient']:.4f}")
        fig, ax = plt.subplots(figsize=(6, 5))
        curve_x = qini_result["curve_x"]
        curve_y = qini_result["curve_y"]
        ax.plot(curve_x, curve_y, label="Model")
        ax.plot([curve_x[0], curve_x[-1]], [curve_y[0], curve_y[-1]], linestyle="--", color="gray", label="Random")
        ax.set_xlabel("Number targeted")
        ax.set_ylabel("Cumulative incremental outcomes")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# Section 3: Statistical rigor layer
# ---------------------------------------------------------------------------
with tab3:
    st.header("Multiple Comparison Correction")
    segment_df = load_csv("segment_effects.csv")
    if segment_df is None or "p_value_adjusted" not in segment_df.columns:
        missing_data_notice("BH-corrected segment table", "notebooks/02_heterogeneity.ipynb")
    else:
        st.dataframe(
            segment_df[["segment", "p_value", "p_value_adjusted", "significant_after_correction"]]
        )
        n_sig = segment_df["significant_after_correction"].sum()
        st.caption(f"{n_sig} of {len(segment_df)} segments remain significant after Benjamini-Hochberg correction.")

    st.header("Per-Segment Power Analysis")
    power_df = load_csv("segment_power.csv")
    if power_df is None:
        missing_data_notice("Segment power table", "notebooks/02_heterogeneity.ipynb")
    else:
        st.dataframe(power_df)
        n_underpowered = power_df["underpowered"].sum()
        if n_underpowered > 0:
            st.warning(
                f"{n_underpowered} of {len(power_df)} segments are underpowered at the assumed effect size. "
                "Per-segment findings for these should be treated as directional, not confirmatory."
            )

# ---------------------------------------------------------------------------
# Section 4: Sensitivity analysis
# ---------------------------------------------------------------------------
with tab4:
    st.header("Rosenbaum Sensitivity Bounds")
    bounds_df = load_csv("rosenbaum_bounds.csv")
    if bounds_df is None:
        missing_data_notice("Rosenbaum bounds table", "notebooks/03_sensitivity.ipynb")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bounds_df["gamma"], bounds_df["worst_case_p_value"], marker="o", markersize=3)
        ax.axhline(0.05, linestyle="--", color="red", linewidth=1, label="alpha = 0.05")
        ax.set_xlabel("Gamma (unmeasured confounding strength)")
        ax.set_ylabel("Worst-case p-value")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        st.dataframe(bounds_df)

    st.header("Diagnostic Critique (LLM)")
    critique = load_json("critique.json")
    if critique is None:
        missing_data_notice("Diagnostic critique", "notebooks/03_sensitivity.ipynb")
    else:
        st.caption(f"Source: {critique['source']}")
        st.markdown(critique["critique_text"])
        if critique["source"] == "rule_based_fallback":
            st.caption(
                "Note: this is the deterministic rule-based fallback, not an actual LLM response "
                "(no Groq/NIM API key was configured when this was generated)."
            )