"""Run the shared hypotension pipeline with a cross-channel PatchTST config."""

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).with_name("fcn_baseline_hypotension.py")),
    run_name="__main__",
)
