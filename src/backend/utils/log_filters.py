import logging


class AllowOnlyAppLogs(logging.Filter):
    """
    Logging filter that allows only records whose logger name
    starts with one of the specified prefixes.

    Intended to be used on console handlers in notebook or
    interactive environments to suppress noisy third-party logs
    while preserving DEBUG logs from the application itself.
    """

    def __init__(self, allowed_prefixes: tuple[str, ...]):
        super().__init__()
        self.allowed_prefixes = allowed_prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        logger_name = record.name or ""
        return logger_name.startswith(self.allowed_prefixes)
