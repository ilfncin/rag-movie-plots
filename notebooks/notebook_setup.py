"""
Description:
This script configures the Jupyter Notebook environment to recognize the
project's package structure, allowing imports from the `src/` directory.

Usage:
%run notebook_setup.py
"""

import os
import sys
from pathlib import Path


# Detects the project root directory (one level above the `notebooks` folder)
project_root = Path(__file__).resolve().parents[1]

# Define paths for both src and backend
src_path = project_root / "src"
backend_path = src_path / "backend"

# Ensure both paths are added to sys.path (if not already present)
for path in [src_path, backend_path]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ["PYTHONPATH"] = str(src_path)

try:
    import logging
    from backend.utils.logger import setup_logging

    setup_logging()
    logger = logging.getLogger("NOTEBOOK")
except Exception as e:
    print(f"Logger not initialized: {e}")

logger.info("Notebook environment configured successfully")
logger.info(f"Project root: {project_root}")
logger.debug(f"Added to sys.path:\n  - {src_path}\n  - {backend_path}")
logger.debug(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
logger.debug(f"Current working directory: {Path.cwd()}")
