"""Project root and local paths so all outputs stay under the repo."""

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local output dirs
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DETECT = RUNS_DIR / "detect"
RUNS_SEGMENT = RUNS_DIR / "segment"
RUNS_WARN = RUNS_DIR / "warn"

# Default weight paths (trained models)
CAMEL_WEIGHTS_PATH = RUNS_DETECT / "camel_detect" / "weights" / "best.pt"
ROAD_WEIGHTS_PATH = RUNS_SEGMENT / "road_segment" / "weights" / "best.pt"
