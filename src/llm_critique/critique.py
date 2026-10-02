import logging
import os

import pandas as pd

from src.sensitivity.rosenbaum import classify_gamma
from src.validation.diagnostics import OVERLAP_COEFFICIENT_WARN

logger = logging.getLogger(__name__)

CRITIQUE_SYSTEM_PROMPT = """You are a methodology reviewer producing a short, plain-language \
critique for a non-technical stakeholder reading a causal inference analysis. You are given \
covariate balance diagnostics, propensity overlap statistics, and a Rosenbaum sensitivity \
bound. Your only job is to flag likely assumption violations in 3-5 short bullet points, in \
plain language, with no jargon left unexplained. Gamma is an odds ratio describing how much \
more likely one of two matched units could be to receive treatment because of an unmeasured \
factor; never convert it into a percentage. Do not summarize the whole analysis, do not \
write a report, do not add caveats beyond what the numbers given to you support. If the \
diagnostics look clean, say so plainly instead of inventing a concern."""

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_NIM_MODEL = "mistralai/mistral-nemotron"
MAX_COMPLETION_TOKENS = 2000


def format_diagnostics_summary(diagnostics: dict, rosenbaum_result: dict) -> str:
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
        f"  - {overlap['pct_within_overlap']:.1f}% of the candidate pool lies within the propensity range "
        f"shared by both arms ({float(overlap_region[0]):.4f}, {float(overlap_region[1]):.4f}); "
        "this range check is lenient by construction\n"
        f"  - Overlap coefficient between treated and control propensity distributions: "
        f"{overlap['overlap_coefficient']:.3f} (1.0 = identical, 0.0 = disjoint)"
    )

    if not critical["significant_at_gamma_1"]:
        gamma_text = "  - The matched-pair result is not statistically significant even with no unmeasured confounding"
    elif critical["critical_gamma"] is None:
        gamma_text = (
            f"  - Conclusion holds even at Gamma up to {critical['gamma_max_checked']} "
            f"(no unmeasured confounding of that strength or less could overturn it)"
        )
    else:
        gamma_text = (
            f"  - Conclusion could be overturned by unmeasured confounding at Gamma >= "
            f"{critical['critical_gamma']:.2f} (odds-ratio strength), based on "
            f"{critical['n_discordant']} discordant pairs"
        )

    return (
        "Covariate balance (standardized mean difference, |SMD| > 0.1 = imbalanced):\n"
        f"{balance_text}\n\n"
        "Propensity overlap:\n"
        f"{overlap_text}\n\n"
        "Rosenbaum sensitivity bound:\n"
        f"{gamma_text}"
    )


def _extract_text(response) -> str:
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError("Completion truncated at the token limit")
    text = choice.message.content
    if not text or not text.strip():
        raise RuntimeError("Completion returned no text")
    return text


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
        max_tokens=MAX_COMPLETION_TOKENS,
    )
    return _extract_text(response)


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
        max_tokens=MAX_COMPLETION_TOKENS,
    )
    return _extract_text(response)


def _rule_based_fallback(diagnostics: dict, rosenbaum_result: dict) -> str:
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

    coefficient = overlap["overlap_coefficient"]
    if coefficient < OVERLAP_COEFFICIENT_WARN:
        flags.append(
            f"- Treated and control propensity distributions overlap weakly "
            f"(overlap coefficient {coefficient:.2f}); estimates rely on extrapolation."
        )
    else:
        flags.append(f"- Treated and control propensity distributions overlap reasonably (coefficient {coefficient:.2f}).")

    status = classify_gamma(critical)
    if status == "not_significant":
        flags.append(
            "- The matched-pair result is not statistically significant even under the no-confounding assumption, "
            "so a sensitivity bound adds nothing."
        )
    elif status == "fragile":
        flags.append(
            f"- The result is sensitive to unmeasured confounding: an unobserved factor with only "
            f"Gamma={critical['critical_gamma']:.2f} strength could overturn the conclusion."
        )
    elif status == "moderate":
        flags.append(
            f"- The result is moderately robust to unmeasured confounding (critical Gamma="
            f"{critical['critical_gamma']:.2f})."
        )
    elif critical["critical_gamma"] is None:
        flags.append(
            f"- The result is robust to unmeasured confounding up to Gamma="
            f"{critical['gamma_max_checked']}, the strongest level checked."
        )
    else:
        flags.append(
            f"- The result is robust to unmeasured confounding (critical Gamma="
            f"{critical['critical_gamma']:.2f})."
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

    logger.warning("No LLM provider produced a critique, falling back to rule-based critique")
    fallback_text = _rule_based_fallback(diagnostics, rosenbaum_result)
    return {"critique_text": fallback_text, "source": "rule_based_fallback", "prompt_used": prompt}
