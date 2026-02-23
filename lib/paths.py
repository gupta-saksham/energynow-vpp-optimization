"""
Project path constants. All scripts should use these for data, results, and outputs.
"""
from pathlib import Path

# Project root is parent of lib/ (this file lives in lib/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_dirs():
    """Create data, results, and outputs directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "scenario_outputs").mkdir(parents=True, exist_ok=True)
