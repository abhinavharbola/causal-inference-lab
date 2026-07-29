"""
The one and only place an LLM appears in this project.

Reads the balance table, overlap diagnostics, and Rosenbaum sensitivity
output from Section 1 and Section 4, and produces a short plain-language
flag of likely assumption violations for a non-technical stakeholder.

This does NOT generate narrative reports, does NOT summarize the whole
project, does NOT write the README, and is not used anywhere in Section
2 or 3. Any temptation to reach for the LLM elsewhere in the build should
be resisted; solve it with code and statistics instead.

Provider: Groq (primary) or NVIDIA NIM (fallback), both free tier.
Both expose an OpenAI-compatible chat completions interface, so the same
request shape works for either with only the base_url/model swapped.

If neither provider is reachable (no API key set, network unavailable),
falls back to a deterministic rule-based flag generator so the pipeline
doesn't hard-fail end-to-end just because a key isn't configured, e.g.
during local testing.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

CRITIQUE_SYSTEM_PROMPT = """You are a methodology reviewer producing a short, plain-language \
critique for a non-technical stakeholder reading a causal inference analysis. You are given \
covariate balance diagnostics, common-support overlap statistics, and a Rosenbaum sensitivity \
bound. Your only job is to flag likely assumption violations in 3-5 short bullet points, in \
plain language, with no jargon left unexplained. Do not summarize the whole analysis, do not \
write a report, do not add caveats beyond what the numbers given to you support. If the \
diagnostics look clean, say so plainly instead of inventing a concern."""

# Model names on free-tier providers change fairly often; confirm current
# availability at build time against https://console.groq.com/docs/models
# or https://build.nvidia.com before relying on these defaults.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_NIM_MODEL = "openai/gpt-oss-120b"


def format_diagnostics_summary(diagnostics: dict, rosenbaum_result: dict) -> str:
    """
    Formats the Section 1/4 diagnostic objects into a compact text block
    for the LLM prompt. Deliberately terse: this is input to a single
    bounded critique step, not a document in its own right.
    """
    balance_df: pd.DataFrame = diagnostics["balance_table"]
    overlap: dict = diagnostics["overlap"]
    critical: dict = rosenbaum_result["critical_gamma_result"]

    balance_lines = []
    for _, row in balance_df.iterrows():
        flag = "IMBALANCED" if row["still_imbalanced"] else "balanced"
        balance_lines.append(
            f"  - {row['covariate']}: SMD before={row['smd_before']:.3f}, "
            f"after={row['smd_after']:.3f} ({flag})"
        )
    balance_text = "\n".join(balance_lines)

    overlap_region = overlap["overlap_region"]
    overlap_text = (
        f"  - {overlap['pct_within_overlap']:.1f}% of matched units within common support region "
        f"({float(overlap_region[0]):.4f}, {float(overlap_region[1]):.4f})"
    )

    if critical["critical_gamma"] is None:
        gamma_text = (
            f"  - Conclusion holds even at Gamma up to {critical['gamma_max_checked']} "
            f"(no unmeasured confounding of that strength or less could overturn it)"
        )
    else:
        gamma_text = (
            f"  - Conclusion could be overturned by unmeasured confounding at Gamma >= "
            f"{critical['critical_gamma']:.2f} (odds-ratio strength)"
        )

    return (
        "Covariate balance (standardized mean difference, |SMD| > 0.1 = imbalanced):\n"
        f"{balance_text}\n\n"
        "Common support overlap:\n"
        f"{overlap_text}\n\n"
        "Rosenbaum sensitivity bound:\n"
        f"{gamma_text}"
    )


def _call_groq(prompt: str, model: str, api_key: str) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content


def _call_nim(prompt: str, model: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content


def _rule_based_fallback(diagnostics: dict, rosenbaum_result: dict) -> str:
    """
    Deterministic, non-LLM fallback used only when no provider is
    reachable. Mirrors the kind of flags an LLM call would produce, so
    the pipeline stays runnable without a configured API key, but this
    path should not be mistaken for the actual LLM critique step in the
    README or dashboard: label output from this path as "fallback"
    wherever it is displayed.
    """
    balance_df: pd.DataFrame = diagnostics["balance_table"]
    overlap: dict = diagnostics["overlap"]
    critical: dict = rosenbaum_result["critical_gamma_result"]

    flags = []

    imbalanced = balance_df[balance_df["still_imbalanced"]]
    if len(imbalanced) > 0:
        cov_list = ", ".join(imbalanced["covariate"].tolist())
        flags.append(f"- Balance was not achieved on: {cov_list}. Treat this estimate with caution.")
    else:
        flags.append("- Covariate balance looks good on all checked covariates after matching.")

    if overlap["pct_within_overlap"] < 90:
        flags.append(
            f"- Only {overlap['pct_within_overlap']:.1f}% of units fall within common support; "
            "a meaningful share of the sample was dropped to enforce overlap."
        )
    else:
        flags.append(f"- Common support overlap is strong ({overlap['pct_within_overlap']:.1f}% retained).")

    if critical["critical_gamma"] is not None and critical["critical_gamma"] < 2.0:
        flags.append(
            f"- The result is sensitive to unmeasured confounding: an unobserved factor with only "
            f"Gamma={critical['critical_gamma']:.2f} strength could overturn the conclusion."
        )
    elif critical["critical_gamma"] is not None:
        flags.append(
            f"- The result is moderately robust to unmeasured confounding (critical Gamma="
            f"{critical['critical_gamma']:.2f})."
        )
    else:
        flags.append(
            f"- The result is robust to unmeasured confounding up to Gamma="
            f"{critical['gamma_max_checked']}, the strongest level checked."
        )

    return "\n".join(flags)


def run_diagnostic_critique(
    diagnostics: dict,
    rosenbaum_result: dict,
    provider: str = "groq",
    groq_api_key: str = None,
    nim_api_key: str = None,
    groq_model: str = DEFAULT_GROQ_MODEL,
    nim_model: str = DEFAULT_NIM_MODEL,
) -> dict:
    """
    Runs the single bounded LLM critique step. Tries the requested
    provider first (default Groq), falls back to the other provider if
    that fails, and falls back to a deterministic rule-based summary if
    neither is reachable (e.g. no API key configured).

    API keys are read from GROQ_API_KEY / NVIDIA_NIM_API_KEY environment
    variables if not passed explicitly.
    """
    groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    nim_api_key = nim_api_key or os.environ.get("NVIDIA_NIM_API_KEY")

    prompt = format_diagnostics_summary(diagnostics, rosenbaum_result)

    providers_order = [provider, "nim" if provider == "groq" else "groq"]

    for p in providers_order:
        try:
            if p == "groq":
                if not groq_api_key:
                    raise RuntimeError("GROQ_API_KEY not set")
                text = _call_groq(prompt, groq_model, groq_api_key)
            elif p == "nim":
                if not nim_api_key:
                    raise RuntimeError("NVIDIA_NIM_API_KEY not set")
                text = _call_nim(prompt, nim_model, nim_api_key)
            else:
                continue

            logger.info("Diagnostic critique generated via %s", p)
            return {"critique_text": text, "source": p, "prompt_used": prompt}

        except Exception as exc:
            logger.warning("Critique via %s failed (%s), trying next option", p, exc)

    logger.warning("No LLM provider reachable, falling back to rule-based critique")
    fallback_text = _rule_based_fallback(diagnostics, rosenbaum_result)
    return {"critique_text": fallback_text, "source": "rule_based_fallback", "prompt_used": prompt}