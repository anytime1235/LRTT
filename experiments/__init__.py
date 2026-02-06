"""
Baseline Experiments Package for MobileBERT on SQuAD QA and GLUE Benchmark.

This package provides tools for running and analyzing baseline experiments
comparing digital and analog (with PCM noise and drift) performance.

Modules:
    - config: Configuration settings and paths
    - run_baseline_experiments: Main runner script
    - collect_results: Results parsing and CSV generation
    - generate_plots: Visualization generation
"""

from .config import (
    BASE_DIR,
    DATA_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    MODEL_NAME,
    GLUE_TASKS,
    GLUE_METRICS,
    ensure_directories,
)

__version__ = "1.0.0"
__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "PLOTS_DIR",
    "MODEL_NAME",
    "GLUE_TASKS",
    "GLUE_METRICS",
    "ensure_directories",
]
