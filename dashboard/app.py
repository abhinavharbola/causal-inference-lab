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

st.set_page_config(
    page_title="Causal Impact & Heterogeneous Response Analysis",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Shared styling: one consistent look for every matplotlib figure in the app,
# set once here rather than repeated per chart.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    }
)
PALETTE = {
    "naive_ols": "#d62728",
    "psm": "#1f77b4",
    "ipw": "#2ca02c",
    "aipw": "#9467bd",
}


def _new_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _file_mtime(path: str):
    """Returns a file's modification time, or None if it doesn't exist.
    Passed as a cache key argument so cached loaders auto-invalidate the
    moment the underlying file changes (e.g. after rerunning a notebook),
    without needing a manual refresh for file-backed artifacts."""
    return os.path.getmtime(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_json(filename: str, mtime=None):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        st.warning(f"⚠️ Could not read `{filename}` (may be mid-write from a running notebook): {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_csv(filename: str, mtime=None):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError) as exc:
        st.warning(f"⚠️ Could not read `{filename}` (may be mid-write from a running notebook): {exc}")
        return None


def load_json_fresh(filename: str):
    """Convenience wrapper: pass the current mtime so the cache is keyed
    correctly without every call site needing to compute it manually."""
    return load_json(filename, mtime=_file_mtime(os.path.join(DATA_DIR, filename)))


def load_csv_fresh(filename: str):
    return load_csv(filename, mtime=_file_mtime(os.path.join(DATA_DIR, filename)))


@st.cache_data(ttl=30, show_spinner=False)
def cached_fetch_estimation_runs():
    """Database reads have no local file mtime to key on, so this relies
    on a short TTL (30s) plus the manual 'Refresh data' button below for
    on-demand invalidation right after a notebook run finishes."""
    return fetch_estimation_runs()


def missing_data_notice(what: str, notebook: str):
    st.info(f"📭 {what} not found yet. Run `{notebook}` to generate it.")


def freshness_caption(filename: str):
    """Shows when this artifact file was last written, so it's obvious
    if the dashboard is displaying results from an old notebook run."""
    path = os.path.join(DATA_DIR, filename)
    mtime = _file_mtime(path)
    if mtime is not None:
        import datetime
        age = datetime.datetime.now().timestamp() - mtime
        stamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        if age < 60:
            age_str = f"{int(age)}s ago"
        elif age < 3600:
            age_str = f"{int(age / 60)}m ago"
        else:
            age_str = f"{age / 3600:.1f}h ago"
        st.caption(f"🕒 Generated {stamp} ({age_str})")


def download_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=f"⬇️ Download {label} (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Sidebar: project summary + pipeline status, so it's obvious at a glance
# which notebooks still need to be run rather than digging through tabs.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Project Status")
    st.caption("Criteo Uplift v2.1 · validation → heterogeneity → sensitivity")

    _artifact_checklist = [
        ("Ground-truth ATE", "ground_truth.json", "01_validation.ipynb"),
        ("Estimation runs (DB)", None, "01_validation.ipynb"),
        ("Balance diagnostics", "balance_table.csv", "01_validation.ipynb"),
        ("Segment effects", "segment_effects.csv", "02_heterogeneity.ipynb"),
        ("Segment power", "segment_power.csv", "02_heterogeneity.ipynb"),
        ("Qini curve", "qini_curve.json", "02_heterogeneity.ipynb"),
        ("Rosenbaum bounds", "rosenbaum_bounds.csv", "03_sensitivity.ipynb"),
        ("LLM critique", "critique.json", "03_sensitivity.ipynb"),
    ]

    _runs_check = cached_fetch_estimation_runs()
    _runs_present = _runs_check is not None and len(_runs_check) > 0

    _done_count = 0
    for label, fname, nb in _artifact_checklist:
        if fname is None:
            present = _runs_present
        else:
            present = os.path.exists(os.path.join(DATA_DIR, fname))
        _done_count += int(present)
        icon = "✅" if present else "⬜"
        st.markdown(f"{icon} {label}")

    st.progress(_done_count / len(_artifact_checklist))
    st.caption(f"{_done_count} of {len(_artifact_checklist)} pipeline artifacts ready")

    st.divider()

    if st.button("🔄 Refresh data", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    st.caption(
        "File-backed artifacts auto-refresh when their file changes. "
        "Database-backed data (estimation runs) refreshes every 30s, or "
        "immediately after clicking the button above."
    )

    with st.expander("ℹ️ About this dashboard"):
        st.markdown(
            "This is a **visualization layer**, not a recomputation engine, it reads "
            "artifacts the notebooks produce, rather than re-running the pipeline. "
            "If a section looks empty, run the notebook listed next to it above."
        )

st.title("📊 Causal Impact & Heterogeneous Response Analysis")
st.caption("Criteo Uplift Modeling Dataset (v2.1), validation, heterogeneity, and sensitivity analysis")

tab1, tab1_5, tab2, tab3, tab4 = st.tabs(
    [
        "📈 1. Validation",
        "🎯 1.5 Outcome Justification",
        "🧬 2. Heterogeneity",
        "📐 3. Statistical Rigor",
        "🛡️ 4. Sensitivity",
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

    ground_truth = load_json_fresh("ground_truth.json")
    runs_df = cached_fetch_estimation_runs()

    if ground_truth is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Ground-truth ATE", f"{ground_truth['ate']:.4f}")
        m2.metric("95% CI lower", f"{ground_truth['ci_lower']:.4f}")
        m3.metric("95% CI upper", f"{ground_truth['ci_upper']:.4f}")

    if runs_df is None or len(runs_df) == 0:
        missing_data_notice("Estimation run log", "notebooks/01_validation.ipynb")
    else:
        severity_order = ["none", "mild", "moderate", "strong"]
        present_severities = [s for s in severity_order if s in runs_df["severity_label"].unique()]

        fig, ax = _new_fig((8, 5))

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
                color=PALETTE.get(method),
                linewidth=2,
            )

        if ground_truth is not None:
            ax.axhline(ground_truth["ate"], color="black", linestyle="--", linewidth=1.5, label="Ground truth ATE")
            ax.axhspan(ground_truth["ci_lower"], ground_truth["ci_upper"], color="gray", alpha=0.15)

        ax.set_xticks(range(len(present_severities)))
        ax.set_xticklabels(present_severities)
        ax.set_xlabel("Confounding severity")
        ax.set_ylabel("Estimated ATE")
        ax.set_title("Estimator bias across confounding severity")
        ax.legend(frameon=False)
        fig.tight_layout()

        st.pyplot(fig)

        # Headline takeaway: which method is most/least biased at the strongest severity present.
        if ground_truth is not None and present_severities:
            strongest = present_severities[-1]
            at_strongest = runs_df[runs_df["severity_label"] == strongest].copy()
            if len(at_strongest) > 0:
                at_strongest["abs_bias"] = (at_strongest["point_estimate"] - ground_truth["ate"]).abs()
                best = at_strongest.loc[at_strongest["abs_bias"].idxmin()]
                worst = at_strongest.loc[at_strongest["abs_bias"].idxmax()]
                st.success(
                    f"At **{strongest}** confounding severity: **{best['method']}** stayed closest to the "
                    f"ground truth (bias {best['abs_bias']:.4f}), while **{worst['method']}** drifted "
                    f"furthest (bias {worst['abs_bias']:.4f})."
                )

        with st.expander("📋 Show raw estimation run log"):
            display_df = runs_df[["method", "severity_label", "g2", "point_estimate", "ci_lower", "ci_upper"]]
            st.dataframe(
                display_df.style.format(
                    {"g2": "{:.2f}", "point_estimate": "{:.4f}", "ci_lower": "{:.4f}", "ci_upper": "{:.4f}"}
                ),
                width='stretch',
            )
            download_button(display_df, "estimation runs", "estimation_runs.csv")
            freshness_caption("ground_truth.json")

    st.subheader("Covariate Balance (matching diagnostics)")
    balance_df = load_csv_fresh("balance_table.csv")
    if balance_df is None:
        missing_data_notice("Balance table", "notebooks/01_validation.ipynb")
    else:
        n_imbalanced = int(balance_df["still_imbalanced"].sum()) if "still_imbalanced" in balance_df.columns else 0
        if n_imbalanced == 0:
            st.success("✅ All covariates balanced after matching (|SMD| ≤ 0.1).")
        else:
            st.warning(f"⚠️ {n_imbalanced} of {len(balance_df)} covariates remain imbalanced after matching.")

        format_cols = {c: "{:.4f}" for c in ["smd_before", "smd_after"] if c in balance_df.columns}
        st.dataframe(balance_df.style.format(format_cols), width='stretch')
        download_button(balance_df, "balance table", "balance_table.csv")
        freshness_caption("balance_table.csv")

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

    m1, m2 = st.columns(2)
    visit_row = table[table["outcome"] == "visit"].iloc[0]
    conversion_row = table[table["outcome"] == "conversion"].iloc[0]
    m1.metric("visit relative MDE", f"{visit_row['mde_relative_pct']:.1f}%")
    m2.metric(
        "conversion relative MDE",
        f"{conversion_row['mde_relative_pct']:.1f}%",
        delta=f"{conversion_row['mde_relative_pct'] - visit_row['mde_relative_pct']:+.1f}pp vs visit",
        delta_color="inverse",
    )

    st.dataframe(
        table.style.format(
            {"baseline_rate": "{:.4f}", "mde_absolute": "{:.5f}", "mde_relative_pct": "{:.2f}%"}
        ),
        width='stretch',
    )
    st.caption(
        "`conversion` typically requires a much larger relative effect to be detectable at the same "
        "sample size, which is why it is excluded from segment-level work (Section 2/3) and used only "
        "for the full-dataset ground-truth ATE in Section 1."
    )

# ---------------------------------------------------------------------------
# Section 2: Heterogeneity / segmentation
# ---------------------------------------------------------------------------
with tab2:
    st.header("Segment-Level CATE Breakdown")
    segment_df = load_csv_fresh("segment_effects.csv")
    if segment_df is None:
        missing_data_notice("Segment effects table", "notebooks/02_heterogeneity.ipynb")
    else:
        format_cols = {
            c: "{:.4f}"
            for c in ["point_estimate", "ci_lower", "ci_upper", "mean_predicted_cate"]
            if c in segment_df.columns
        }
        styled = segment_df.style.format(format_cols)
        if "point_estimate" in segment_df.columns:
            styled = styled.highlight_max(subset=["point_estimate"], color="#d4f4dd")
        st.dataframe(styled, width='stretch')
        download_button(segment_df, "segment effects", "segment_effects.csv")
        freshness_caption("segment_effects.csv")

        fig, ax = _new_fig((8, 4))
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
            color="#1f77b4",
            markersize=7,
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(segment_df["segment"])
        ax.set_xlabel("Estimated treatment effect")
        ax.set_title("Per-segment treatment effect with bootstrap CI")
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Qini Curve (CATE model quality)")
    qini_result = load_json_fresh("qini_curve.json")
    if qini_result is None:
        missing_data_notice("Qini curve", "notebooks/02_heterogeneity.ipynb")
    else:
        qini_coef = qini_result["qini_coefficient"]
        m1, m2 = st.columns([1, 2])
        m1.metric("Qini coefficient", f"{qini_coef:.4f}")
        if qini_coef > 0.02:
            m2.success("Model ranks units by uplift meaningfully better than random targeting.")
        elif qini_coef > 0:
            m2.info("Model beats random targeting, but the margin is modest, interpret segments with care.")
        else:
            m2.warning("Model is not clearly better than random targeting, segment findings below may not reflect real heterogeneity.")

        fig, ax = _new_fig((6, 5))
        curve_x = qini_result["curve_x"]
        curve_y = qini_result["curve_y"]
        ax.plot(curve_x, curve_y, label="Model", color="#1f77b4", linewidth=2)
        ax.plot(
            [curve_x[0], curve_x[-1]], [curve_y[0], curve_y[-1]],
            linestyle="--", color="gray", label="Random targeting",
        )
        ax.fill_between(curve_x, curve_y, np.linspace(curve_y[0], curve_y[-1], len(curve_x)), alpha=0.08, color="#1f77b4")
        ax.set_xlabel("Number targeted")
        ax.set_ylabel("Cumulative incremental outcomes")
        ax.set_title("Qini Curve")
        fig.tight_layout()
        st.pyplot(fig)
        freshness_caption("qini_curve.json")
with tab3:
    st.header("Multiple Comparison Correction")
    segment_df = load_csv_fresh("segment_effects.csv")
    if segment_df is None or "p_value_adjusted" not in segment_df.columns:
        missing_data_notice("BH-corrected segment table", "notebooks/02_heterogeneity.ipynb")
    else:
        n_sig = int(segment_df["significant_after_correction"].sum())
        n_total = len(segment_df)
        m1, m2 = st.columns(2)
        m1.metric("Segments significant after BH correction", f"{n_sig} / {n_total}")
        n_sig_raw = int((segment_df["p_value"] < 0.05).sum())
        m2.metric("Segments significant before correction", f"{n_sig_raw} / {n_total}", delta=n_sig - n_sig_raw)

        display_cols = ["segment", "p_value", "p_value_adjusted", "significant_after_correction"]
        st.dataframe(
            segment_df[display_cols].style.format({"p_value": "{:.4f}", "p_value_adjusted": "{:.4f}"}),
            width='stretch',
        )
        download_button(segment_df[display_cols], "BH-corrected segments", "segment_significance.csv")
        freshness_caption("segment_effects.csv")

    st.header("Per-Segment Power Analysis")
    power_df = load_csv_fresh("segment_power.csv")
    if power_df is None:
        missing_data_notice("Segment power table", "notebooks/02_heterogeneity.ipynb")
    else:
        n_underpowered = int(power_df["underpowered"].sum())
        format_cols = {c: "{:.4f}" for c in ["achieved_power"] if c in power_df.columns}
        st.dataframe(power_df.style.format(format_cols), width='stretch')
        download_button(power_df, "segment power", "segment_power.csv")
        freshness_caption("segment_power.csv")

        if n_underpowered > 0:
            st.warning(
                f"⚠️ {n_underpowered} of {len(power_df)} segments are underpowered at the assumed effect size. "
                "Per-segment findings for these should be treated as directional, not confirmatory."
            )
        else:
            st.success("✅ All segments are adequately powered at the assumed effect size.")

# ---------------------------------------------------------------------------
# Section 4: Sensitivity analysis
# ---------------------------------------------------------------------------
with tab4:
    st.header("Rosenbaum Sensitivity Bounds")
    bounds_df = load_csv_fresh("rosenbaum_bounds.csv")
    if bounds_df is None:
        missing_data_notice("Rosenbaum bounds table", "notebooks/03_sensitivity.ipynb")
    else:
        alpha = 0.05
        crossing = bounds_df[bounds_df["worst_case_p_value"] >= alpha]
        approx_critical_gamma = float(crossing["gamma"].iloc[0]) if len(crossing) > 0 else None

        m1, m2 = st.columns(2)
        if approx_critical_gamma is not None:
            m1.metric("Approx. critical Gamma", f"{approx_critical_gamma:.2f}")
            if approx_critical_gamma < 1.5:
                m2.warning("Fragile: only mild unmeasured confounding would overturn this conclusion.")
            elif approx_critical_gamma < 3:
                m2.info("Moderately robust to unmeasured confounding.")
            else:
                m2.success("Robust: substantial unmeasured confounding would be needed to overturn this.")
        else:
            m1.metric("Approx. critical Gamma", f"> {bounds_df['gamma'].max():.2f}")
            m2.success("Robust to every confounding strength checked in this table.")
        st.caption("Approximate value read off the saved bounds table's grid resolution, not re-solved exactly.")

        fig, ax = _new_fig((6, 4))
        ax.plot(bounds_df["gamma"], bounds_df["worst_case_p_value"], marker="o", markersize=3, color="#1f77b4", linewidth=2)
        ax.axhline(alpha, linestyle="--", color="#d62728", linewidth=1.5, label=f"alpha = {alpha}")
        ax.set_xlabel("Gamma (unmeasured confounding strength)")
        ax.set_ylabel("Worst-case p-value")
        ax.set_title("Rosenbaum Sensitivity Bounds")
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("📋 Show raw bounds table"):
            st.dataframe(bounds_df.style.format({"worst_case_p_value": "{:.4f}"}), width='stretch')
            download_button(bounds_df, "Rosenbaum bounds", "rosenbaum_bounds.csv")
            freshness_caption("rosenbaum_bounds.csv")

    st.header("Diagnostic Critique (LLM)")
    critique = load_json_fresh("critique.json")
    if critique is None:
        missing_data_notice("Diagnostic critique", "notebooks/03_sensitivity.ipynb")
    else:
        if critique["source"] == "rule_based_fallback":
            st.warning(
                "⚙️ Rule-based fallback, not an actual LLM response "
                "(no Groq/NIM API key was configured when this was generated)."
            )
        else:
            st.success(f"🤖 Live LLM response via **{critique['source']}**")
        st.markdown(critique["critique_text"])
        freshness_caption("critique.json")