import logging
import os

_CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(
    service_name: str = "causal-uplift-project",
    level: int = logging.INFO,
    logfire_token: str = None,
) -> bool:
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
    try:
        import logfire
        return logfire.span(name, **attributes)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()