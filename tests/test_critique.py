import pandas as pd
import pytest

from src.llm_critique import critique
from src.llm_critique.critique import (
    _extract_text,
    _rule_based_fallback,
    format_diagnostics_summary,
    run_diagnostic_critique,
)


def _diagnostics(coefficient: float = 0.9, imbalanced: bool = False) -> dict:
    balance = pd.DataFrame(
        {
            "covariate": ["f0", "f1"],
            "smd_before": [0.3, 0.1],
            "smd_after": [0.2 if imbalanced else 0.01, 0.01],
            "still_imbalanced": [imbalanced, False],
        }
    )
    overlap = {
        "overlap_region": (0.001, 0.999),
        "pct_within_overlap": 99.9,
        "overlap_coefficient": coefficient,
    }
    return {"balance_table": balance, "overlap": overlap}


def _rosenbaum(significant: bool = True, critical=1.2) -> dict:
    return {
        "critical_gamma_result": {
            "significant_at_gamma_1": significant,
            "critical_gamma": critical,
            "gamma_max_checked": 10.0,
            "n_discordant": 1_390,
        }
    }


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_Choice(content, finish_reason)]


def test_summary_uses_candidate_pool_wording_and_overlap_coefficient():
    text = format_diagnostics_summary(_diagnostics(), _rosenbaum())

    assert "candidate pool" in text
    assert "matched units" not in text
    assert "Overlap coefficient" in text


def test_summary_states_when_result_is_not_significant():
    text = format_diagnostics_summary(_diagnostics(), _rosenbaum(significant=False, critical=None))

    assert "not statistically significant" in text


def test_fallback_flags_weak_overlap_by_coefficient_not_range_share():
    text = _rule_based_fallback(_diagnostics(coefficient=0.3), _rosenbaum())

    assert "overlap weakly" in text


def test_fallback_reports_fragile_moderate_and_robust_results_distinctly():
    fragile = _rule_based_fallback(_diagnostics(), _rosenbaum(critical=1.2))
    moderate = _rule_based_fallback(_diagnostics(), _rosenbaum(critical=2.0))
    robust = _rule_based_fallback(_diagnostics(), _rosenbaum(critical=None))

    assert "sensitive to unmeasured confounding" in fragile
    assert "moderately robust" in moderate
    assert "robust to unmeasured confounding up to" in robust


def test_fallback_does_not_present_a_non_significant_result_as_overturnable():
    text = _rule_based_fallback(_diagnostics(), _rosenbaum(significant=False, critical=None))

    assert "not statistically significant" in text
    assert "could overturn" not in text


def test_extract_text_rejects_truncated_and_empty_completions():
    with pytest.raises(RuntimeError):
        _extract_text(_Response("partial", finish_reason="length"))
    with pytest.raises(RuntimeError):
        _extract_text(_Response(None))
    with pytest.raises(RuntimeError):
        _extract_text(_Response("   "))
    assert _extract_text(_Response("ok")) == "ok"


def test_run_diagnostic_critique_falls_back_when_no_keys_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)

    result = run_diagnostic_critique(_diagnostics(), _rosenbaum())

    assert result["source"] == "rule_based_fallback"


def test_run_diagnostic_critique_falls_back_when_provider_output_is_truncated(monkeypatch):
    def truncated(prompt, model, api_key):
        return _extract_text(_Response("cut off", finish_reason="length"))

    monkeypatch.setattr(critique, "_call_groq", truncated)
    monkeypatch.setattr(critique, "_call_nim", truncated)

    result = run_diagnostic_critique(_diagnostics(), _rosenbaum(), groq_api_key="x", nim_api_key="y")

    assert result["source"] == "rule_based_fallback"
