import datetime
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

# Absolute, not relative to cwd: relative paths broke if the dashboard was ever
# launched from a directory other than the project root.
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

st.set_page_config(
    page_title="Causal Impact & Heterogeneous Response Analysis",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
#
# A "research instrument" identity, not a generic SaaS dashboard: a cool
# neutral paper background, a deep ink-blue for structure, a restrained gold
# for the handful of signal moments (active tab, key emphasis), and a
# muted, jewel-toned quartet of method colors used consistently across every
# chart, table, and badge in the app instead of matplotlib's default tab10.
#
# Type: Source Serif 4 for headings (an editorial, rigorous voice), Public
# Sans for body/UI text, IBM Plex Mono for anything that reads as a
# measurement, an estimate, a p-value, a gamma.
# ---------------------------------------------------------------------------
INK = "#1B1E24"
INK_SOFT = "#5B6270"
PAPER = "#EEF0F3"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F6F7F9"
BORDER = "#DBDFE6"
PRIMARY = "#223A5E"
PRIMARY_SOFT = "#3D5A80"
ACCENT = "#B8862B"

PALETTE = {
    "naive_ols": "#9B2226",
    "psm": "#3D6E8C",
    "ipw": "#3F7D5C",
    "aipw": "#5B4B8A",
}

METHOD_LABELS = {
    "naive_ols": "Naive OLS",
    "psm": "PSM",
    "ipw": "IPW",
    "aipw": "AIPW",
}


def inject_custom_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {{
            --ink: {INK};
            --ink-soft: {INK_SOFT};
            --paper: {PAPER};
            --surface: {SURFACE};
            --surface-alt: {SURFACE_ALT};
            --border: {BORDER};
            --primary: {PRIMARY};
            --primary-soft: {PRIMARY_SOFT};
            --accent: {ACCENT};
            --radius: 10px;
        }}

        html, body, [class*="css"] {{
            font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{ background: var(--paper); }}

        [data-testid="stAppViewContainer"] .block-container {{
            padding-top: 4.5rem !important;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}

        /* Streamlit's fixed top toolbar sits above the content at scroll position 0;
           without enough clearance above it clips the first element (the masthead
           eyebrow). Blend it into the page background instead of hiding it, since it
           still holds the sidebar toggle. */
        [data-testid="stHeader"] {{
            background: var(--paper);
        }}

        /* ---- Typography ---- */
        h1, h2, h3, h4 {{
            font-family: 'Source Serif 4', Georgia, serif !important;
            color: var(--ink) !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }}
        h2[data-testid="stHeadingWithActionElements"], h2 {{
            padding-bottom: 0.55rem;
            border-bottom: 1px solid var(--border);
            margin-top: 2.75rem !important;
        }}
        h3 {{ margin-top: 1.6rem !important; }}
        p, li, span, label {{ color: var(--ink); }}

        [data-testid="stCaptionContainer"] {{
            font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
            color: var(--ink-soft) !important;
            font-size: 0.8rem !important;
            letter-spacing: 0.01em;
        }}

        code, [data-testid="stCode"] {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}

        /* ---- Eyebrow labels (section kicker, mono + gold) ---- */
        .eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 600;
            margin: 0 0 0.2rem 0;
        }}
        .eyebrow.sub {{ color: var(--primary-soft); }}

        /* ---- Masthead ---- */
        .masthead {{
            text-align: center;
            max-width: 860px;
            margin: 0 auto 0.5rem auto;
        }}
        .masthead-eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--primary-soft);
            margin-bottom: 0.6rem;
        }}
        .masthead-title {{
            font-family: 'Source Serif 4', serif !important;
            font-size: 3.3rem !important;
            font-weight: 700 !important;
            color: var(--ink) !important;
            margin: 0 !important;
            line-height: 1.2 !important;
        }}
        .masthead-sub {{
            font-family: 'Public Sans', sans-serif;
            color: var(--ink-soft);
            font-size: 1rem;
            margin-top: 0.6rem;
        }}
        .masthead-rule {{
            height: 3px;
            width: 60px;
            background: var(--accent);
            margin: 1.1rem auto 0.4rem auto;
            border-radius: 2px;
        }}

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {{
            background: var(--surface);
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] h3 {{
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600 !important;
            border-bottom: none;
            margin-top: 0.25rem !important;
        }}

        .status-row {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.22rem 0;
            font-family: 'Public Sans', sans-serif;
            font-size: 0.86rem;
            color: var(--ink);
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .status-dot.done {{ background: #3F7D5C; }}
        .status-dot.pending {{ background: var(--surface); border: 1.5px solid #C3C9D2; }}
        .status-row.pending {{ color: var(--ink-soft); }}

        /* ---- Metrics ---- */
        [data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.95rem 1.1rem 0.85rem 1.1rem;
            height: 128px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
            overflow: hidden;
        }}
        [data-testid="stMetricLabel"] p {{
            font-family: 'IBM Plex Mono', monospace !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.7rem !important;
            color: var(--ink-soft) !important;
            white-space: normal !important;
            line-height: 1.35;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: var(--primary) !important;
            font-weight: 600 !important;
            font-size: 1.55rem !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            line-height: 1.25;
        }}
        [data-testid="stMetricDelta"] {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}
        /* Every metric card in a row gets the same fixed height (set once, above)
           regardless of whether it has a delta pill or a two-line label, rather
           than relying on flex/percentage stretch through Streamlit's nested
           wrapper divs, which doesn't reliably equalize sibling heights. */

        /* ---- Tabs ---- */
        [data-testid="stTabs"] [role="tablist"] {{
            display: flex;
            width: 100%;
            border-bottom: 1px solid var(--border);
            gap: 0;
        }}
        [data-testid="stTab"] {{
            flex: 1 1 0;
            display: flex !important;
            align-items: center;
            justify-content: center;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.8rem !important;
            letter-spacing: 0.02em;
            color: var(--ink-soft) !important;
            border-bottom: 2.5px solid transparent !important;
            padding: 0.7rem 0.5rem !important;
        }}
        [data-testid="stTab"] p {{
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.8rem !important;
            text-align: center;
        }}
        [data-testid="stTab"][aria-selected="true"] {{
            color: var(--primary) !important;
            border-bottom: 2.5px solid var(--accent) !important;
        }}
        [data-testid="stTab"][aria-selected="true"] p {{
            color: var(--primary) !important;
            font-weight: 600 !important;
        }}

        /* ---- DataFrame / tables ---- */
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }}

        /* ---- Buttons ---- */
        [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {{
            border-radius: 8px !important;
            border: 1px solid var(--primary) !important;
            color: var(--primary) !important;
            font-family: 'Public Sans', sans-serif !important;
            font-weight: 600 !important;
        }}
        [data-testid="stBaseButton-secondary"]:hover {{
            background: var(--primary) !important;
            color: white !important;
        }}
        [data-testid="stDownloadButton"] button {{
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            color: var(--primary-soft) !important;
            font-family: 'Public Sans', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }}
        [data-testid="stDownloadButton"] button:hover {{
            border-color: var(--primary) !important;
            color: var(--primary) !important;
        }}

        /* ---- Alerts ---- */
        [data-testid="stAlert"] {{
            border-radius: var(--radius);
            font-family: 'Public Sans', sans-serif;
        }}

        /* ---- Number input (MDE toggles) ---- */
        [data-testid="stNumberInputContainer"] {{
            background: var(--surface) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: 8px !important;
        }}
        [data-testid="stNumberInputContainer"]:focus-within {{
            border-color: var(--primary-soft) !important;
        }}
        [data-testid="stNumberInputField"] {{
            color: var(--ink) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            background: var(--surface) !important;
        }}
        [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{
            background: var(--surface-alt) !important;
            color: var(--primary) !important;
        }}
        [data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover {{
            background: var(--border) !important;
        }}

        /* ---- Expander ---- */
        [data-testid="stExpander"] {{
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--surface);
        }}

        /* ---- Progress bar ---- */
        [data-testid="stProgress"] > div > div {{ background: var(--accent) !important; }}

        hr {{ border-color: var(--border) !important; }}

        /* ---- Card wrapper (used for pipeline overview) ---- */
        .ci-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.1rem 1.25rem;
        }}

        /* ---- Critique tab: source badge pill ---- */
        .source-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-weight: 600;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 1rem;
        }}
        .source-pill.live {{
            background: #E4F1E8;
            color: #2F6B47;
            border: 1px solid #BFDFC9;
        }}
        .source-pill.fallback {{
            background: #FBF1DE;
            color: #8A6416;
            border: 1px solid #EEDCB0;
        }}
        .critique-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--primary-soft);
            border-radius: var(--radius);
            padding: 1.3rem 1.5rem;
            font-family: 'Public Sans', sans-serif;
            line-height: 1.65;
        }}
        .critique-card ul {{ margin: 0; padding-left: 1.2rem; }}
        .critique-card li {{ margin-bottom: 0.55rem; }}
        .critique-card li:last-child {{ margin-bottom: 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def eyebrow(text: str, sub: bool = False):
    cls = "eyebrow sub" if sub else "eyebrow"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


inject_custom_css()

# ---------------------------------------------------------------------------
# Shared matplotlib styling: one consistent look for every chart in the app,
# tuned to match the CSS token palette above rather than matplotlib defaults.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": BORDER,
        "axes.grid": True,
        "grid.color": BORDER,
        "grid.alpha": 0.7,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "legend.frameon": False,
    }
)


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
        st.warning(f"Could not read `{filename}` (may be mid-write from a running notebook): {exc}", icon=":material/warning:")
        return None


@st.cache_data(show_spinner=False)
def load_csv(filename: str, mtime=None):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError) as exc:
        st.warning(f"Could not read `{filename}` (may be mid-write from a running notebook): {exc}", icon=":material/warning:")
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
    st.info(f"{what} not found yet. Run `{notebook}` to generate it.", icon=":material/inbox:")


def freshness_caption(filename: str):
    """Shows when this artifact file was last written, so it's obvious
    if the dashboard is displaying results from an old notebook run."""
    path = os.path.join(DATA_DIR, filename)
    mtime = _file_mtime(path)
    if mtime is not None:
        age = datetime.datetime.now().timestamp() - mtime
        stamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        if age < 60:
            age_str = f"{int(age)}s ago"
        elif age < 3600:
            age_str = f"{int(age / 60)}m ago"
        else:
            age_str = f"{age / 3600:.1f}h ago"
        st.caption(f"GENERATED {stamp} · {age_str}")


def download_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=f"Download {label} (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        icon=":material/download:",
    )


def status_pill(present: bool, label: str):
    dot_cls = "done" if present else "pending"
    row_cls = "status-row" if present else "status-row pending"
    st.markdown(
        f'<div class="{row_cls}"><span class="status-dot {dot_cls}"></span>{label}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: project summary + pipeline status, so it's obvious at a glance
# which notebooks still need to be run rather than digging through tabs.
# ---------------------------------------------------------------------------
with st.sidebar:
    eyebrow("Criteo Uplift v2.1")
    st.markdown("### Pipeline Status")
    st.caption("validation &rarr; heterogeneity &rarr; sensitivity", unsafe_allow_html=True)
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

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
        status_pill(present, label)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.progress(_done_count / len(_artifact_checklist))
    st.caption(f"{_done_count} OF {len(_artifact_checklist)} ARTIFACTS READY")

    st.divider()

    if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption(
        "File-backed artifacts auto-refresh when their file changes. "
        "Database-backed data (estimation runs) refreshes every 30s, or "
        "immediately after clicking the button above."
    )

    with st.expander("About this dashboard"):
        st.markdown(
            "This is a **visualization layer**, not a recomputation engine, it reads "
            "artifacts the notebooks produce, rather than re-running the pipeline. "
            "If a section looks empty, run the notebook listed next to it above."
        )

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="masthead-eyebrow">Causal Inference &middot; Validation Report</div>
        <p class="masthead-title">Causal Impact &amp; Heterogeneous Response Analysis</p>
        <div class="masthead-rule"></div>
        <p class="masthead-sub">Criteo Uplift Modeling Dataset (v2.1) &middot; validation, heterogeneity, and sensitivity analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab1_5, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "01 &middot; Validation",
        "01.5 &middot; Outcome",
        "02 &middot; Heterogeneity",
        "03 &middot; Rigor",
        "04 &middot; Sensitivity",
        "05 &middot; Critique",
    ]
)

# ---------------------------------------------------------------------------
# Section 1: Validation via self-induced confounding
# ---------------------------------------------------------------------------
with tab1:
    eyebrow("Stage 01")
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
                label=METHOD_LABELS.get(method, method),
                capsize=3,
                color=PALETTE.get(method, PRIMARY),
                linewidth=2,
            )

        if ground_truth is not None:
            ax.axhline(ground_truth["ate"], color=INK, linestyle="--", linewidth=1.5, label="Ground truth ATE")
            ax.axhspan(ground_truth["ci_lower"], ground_truth["ci_upper"], color=PRIMARY, alpha=0.08)

        ax.set_xticks(range(len(present_severities)))
        ax.set_xticklabels([s.capitalize() for s in present_severities])
        ax.set_xlabel("Confounding severity")
        ax.set_ylabel("Estimated ATE")
        ax.set_title("Estimator bias across confounding severity")
        ax.legend()
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
                    f"At **{strongest}** confounding severity: **{METHOD_LABELS.get(best['method'], best['method'])}** "
                    f"stayed closest to the ground truth (bias {best['abs_bias']:.4f}), while "
                    f"**{METHOD_LABELS.get(worst['method'], worst['method'])}** drifted furthest "
                    f"(bias {worst['abs_bias']:.4f}).",
                    icon=":material/check_circle:",
                )

        with st.expander("Show raw estimation run log"):
            display_df = runs_df[["method", "severity_label", "g2", "point_estimate", "ci_lower", "ci_upper"]]
            st.dataframe(
                display_df.style.format(
                    {"g2": "{:.2f}", "point_estimate": "{:.4f}", "ci_lower": "{:.4f}", "ci_upper": "{:.4f}"}
                ),
                width="stretch",
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
            st.success("All covariates balanced after matching (|SMD| \u2264 0.1).", icon=":material/check_circle:")
        else:
            st.warning(
                f"{n_imbalanced} of {len(balance_df)} covariates remain imbalanced after matching.",
                icon=":material/warning:",
            )

        match_diag = load_json_fresh("match_diagnostics.json")
        if match_diag is not None:
            # Matching is strict 1:1 without replacement, so with ~85% of the sample
            # treated, most treated units are expected to go unmatched — this reports
            # exactly how much of the treated group the matched sample above is based on.
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "Treated units matched",
                f"{100 * match_diag['match_rate']:.1f}%",
                help=f"{match_diag['n_pairs']:,} of {match_diag['n_treated_total']:,} treated units",
            )
            mc2.metric("Distinct controls used", f"{match_diag['n_control_unique']:,}")
            controls_match = match_diag["n_control_unique"] == match_diag["n_pairs"]
            mc3.metric("Each control used once", "Yes" if controls_match else "No")

        format_cols = {c: "{:.4f}" for c in ["smd_before", "smd_after"] if c in balance_df.columns}
        st.dataframe(balance_df.style.format(format_cols), width="stretch")
        download_button(balance_df, "balance table", "balance_table.csv")
        freshness_caption("balance_table.csv")

# ---------------------------------------------------------------------------
# Section 1.5: Outcome variable justification
# ---------------------------------------------------------------------------
with tab1_5:
    eyebrow("Stage 01.5")
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
        width="stretch",
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
    eyebrow("Stage 02")
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
            styled = styled.highlight_max(subset=["point_estimate"], color="#DCE9E1")
        st.dataframe(styled, width="stretch")
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
            color=PRIMARY_SOFT,
            markersize=7,
        )
        ax.axvline(0, color=INK_SOFT, linestyle="--", linewidth=1)
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
            m2.success("Model ranks units by uplift meaningfully better than random targeting.", icon=":material/check_circle:")
        elif qini_coef > 0:
            m2.info("Model beats random targeting, but the margin is modest, interpret segments with care.", icon=":material/info:")
        else:
            m2.warning(
                "Model is not clearly better than random targeting, segment findings below may not reflect real heterogeneity.",
                icon=":material/warning:",
            )

        fig, ax = _new_fig((6, 5))
        curve_x = qini_result["curve_x"]
        curve_y = qini_result["curve_y"]
        ax.plot(curve_x, curve_y, label="Model", color=PRIMARY_SOFT, linewidth=2)
        ax.plot(
            [curve_x[0], curve_x[-1]], [curve_y[0], curve_y[-1]],
            linestyle="--", color=INK_SOFT, label="Random targeting",
        )
        ax.fill_between(curve_x, curve_y, np.linspace(curve_y[0], curve_y[-1], len(curve_x)), alpha=0.08, color=PRIMARY_SOFT)
        ax.set_xlabel("Number targeted")
        ax.set_ylabel("Cumulative incremental outcomes")
        ax.set_title("Qini Curve")
        fig.tight_layout()
        st.pyplot(fig)
        freshness_caption("qini_curve.json")

# ---------------------------------------------------------------------------
# Section 3: Statistical rigor
# ---------------------------------------------------------------------------
with tab3:
    eyebrow("Stage 03")
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
            width="stretch",
        )
        download_button(segment_df[display_cols], "BH-corrected segments", "segment_significance.csv")
        freshness_caption("segment_effects.csv")

    eyebrow("Stage 03", sub=True)
    st.header("Per-Segment Power Analysis")
    power_df = load_csv_fresh("segment_power.csv")
    if power_df is None:
        missing_data_notice("Segment power table", "notebooks/02_heterogeneity.ipynb")
    else:
        n_underpowered = int(power_df["underpowered"].sum())
        format_cols = {c: "{:.4f}" for c in ["achieved_power"] if c in power_df.columns}
        st.dataframe(power_df.style.format(format_cols), width="stretch")
        download_button(power_df, "segment power", "segment_power.csv")
        freshness_caption("segment_power.csv")

        if n_underpowered > 0:
            st.warning(
                f"{n_underpowered} of {len(power_df)} segments are underpowered at the assumed effect size. "
                "Per-segment findings for these should be treated as directional, not confirmatory.",
                icon=":material/warning:",
            )
        else:
            st.success("All segments are adequately powered at the assumed effect size.", icon=":material/check_circle:")

# ---------------------------------------------------------------------------
# Section 4: Sensitivity analysis
# ---------------------------------------------------------------------------
with tab4:
    eyebrow("Stage 04")
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
                m2.warning("Fragile: only mild unmeasured confounding would overturn this conclusion.", icon=":material/warning:")
            elif approx_critical_gamma < 3:
                m2.info("Moderately robust to unmeasured confounding.", icon=":material/info:")
            else:
                m2.success("Robust: substantial unmeasured confounding would be needed to overturn this.", icon=":material/check_circle:")
        else:
            m1.metric("Approx. critical Gamma", f"> {bounds_df['gamma'].max():.2f}")
            m2.success("Robust to every confounding strength checked in this table.", icon=":material/check_circle:")
        st.caption("Approximate value read off the saved bounds table's grid resolution, not re-solved exactly.")

        fig, ax = _new_fig((6, 4))
        ax.plot(bounds_df["gamma"], bounds_df["worst_case_p_value"], marker="o", markersize=3, color=PRIMARY_SOFT, linewidth=2)
        ax.axhline(alpha, linestyle="--", color="#9B2226", linewidth=1.5, label=f"alpha = {alpha}")
        ax.set_xlabel("Gamma (unmeasured confounding strength)")
        ax.set_ylabel("Worst-case p-value")
        ax.set_title("Rosenbaum Sensitivity Bounds")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("Show raw bounds table"):
            st.dataframe(bounds_df.style.format({"worst_case_p_value": "{:.4f}"}), width="stretch")
            download_button(bounds_df, "Rosenbaum bounds", "rosenbaum_bounds.csv")
            freshness_caption("rosenbaum_bounds.csv")

# ---------------------------------------------------------------------------
# Section 5: Diagnostic critique (LLM)
# ---------------------------------------------------------------------------
with tab5:
    eyebrow("Stage 05")
    st.header("Diagnostic Critique")
    st.write(
        "An automated second read of the diagnostics above, checking whether the numbers "
        "actually support the conclusions drawn from them, generated after Section 4."
    )

    critique = load_json_fresh("critique.json")
    if critique is None:
        missing_data_notice("Diagnostic critique", "notebooks/03_sensitivity.ipynb")
    else:
        is_fallback = critique["source"] == "rule_based_fallback"
        pill_cls = "fallback" if is_fallback else "live"
        pill_text = "Rule-based fallback" if is_fallback else f"Live LLM &middot; {critique['source']}"
        st.markdown(f'<div class="source-pill {pill_cls}">{pill_text}</div>', unsafe_allow_html=True)

        if is_fallback:
            st.caption(
                "No Groq/NIM API key was configured when this was generated, so this is a "
                "rule-based fallback, not an actual LLM response."
            )

        with st.container(border=True):
            st.markdown(critique["critique_text"])

        freshness_caption("critique.json")