"""Run the validated supervised hypotension pipeline with PatchTST."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).with_name("fcn_baseline_hypotension.py")),
    run_name="__main__",
)
