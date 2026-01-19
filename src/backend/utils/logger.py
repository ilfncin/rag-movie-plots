import logging
from datetime import datetime
from pathlib import Path

_LOGGER_INITIALIZED = False


def setup_logging():
    project_root = Path(__file__).resolve().parents[3]  
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"rag_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Log file created at: {log_file.resolve()}")

    _LOGGER_INITIALIZED = True