"""
Structured logging setup for the project. Every module in src/ already
uses Python's standard `logging` module (logger = logging.getLogger(__name__)).
Rather than rewriting each call site to a Logfire-specific API, this
module attaches a LogfireLoggingHandler to the root logger, so existing
logger.info/warning calls throughout validation/, heterogeneity/,
sensitivity/, and llm_critique/ are forwarded to Logfire as structured
log entries with no other code changes required.

Falls back to plain console logging if LOGFIRE_TOKEN is not set or the
logfire package/configuration fails, consistent with the free-tier /
no-hard-dependency-on-a-single-provider pattern used elsewhere in utils/
(data_loader.py's HF/sklift fallback, db.py's Supabase/SQLite fallback).
"""

import logging
import os

_CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(
    service_name: str = "causal-uplift-project",
    level: int = logging.INFO,
    logfire_token: str = None,
) -> bool:
    """
    Configures logging for the whole project. Call this once, at the
    start of a notebook or the dashboard entrypoint, before any other
    project module is imported and starts logging.

    Returns True if Logfire was successfully configured, False if the
    console fallback is in use. Callers generally don't need to check
    this; it's returned mainly for the dashboard to show which backend
    is active.
    """
    logfire_token = logfire_token or os.environ.get("LOGFIRE_TOKEN")

    if logfire_token:
        try:
            import logfire

            logfire.configure(token=logfire_token, service_name=service_name)

            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            root_logger.addHandler(logfire.LogfireLoggingHandler())

            logging.getLogger(__name__).info("Structured logging configured via Logfire")
            return True

        except Exception as exc:
            logging.basicConfig(level=level, format=_CONSOLE_FORMAT)
            logging.getLogger(__name__).warning(
                "Logfire configuration failed (%s), falling back to console logging", exc
            )
            return False

    logging.basicConfig(level=level, format=_CONSOLE_FORMAT)
    logging.getLogger(__name__).info(
        "LOGFIRE_TOKEN not set, using console logging fallback"
    )
    return False


def get_span(name: str, **attributes):
    """
    Returns a context manager that creates a Logfire span if Logfire is
    active, or a no-op context manager otherwise. Useful for wrapping a
    whole estimation step (e.g. one confounding severity's full
    estimator comparison) as a single structured span with attributes,
    rather than relying only on individual log lines.

    Usage:
        with get_span("estimator_comparison", severity="strong", g2=3.0):
            run_estimator_comparison(...)
    """
    try:
        import logfire
        return logfire.span(name, **attributes)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()