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

print("Notebook environment configured successfully!\n")
print(f"Project root: {project_root}")
print(f"Added to sys.path:\n  - {src_path}\n  - {backend_path}")
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
print(f"Current working directory: {Path.cwd()}")
