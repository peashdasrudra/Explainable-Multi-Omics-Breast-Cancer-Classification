"""
utils.py — Reproducibility and Console Output Utilities
=========================================================
Provides seed locking for full reproducibility and formatted console
output functions used throughout the pipeline.

Reproducibility Note:
    set_all_seeds() locks Python's hash seed, the built-in random module,
    and NumPy's random state to ensure identical results across runs.
    This is critical for a defensible thesis — reviewers must be able
    to reproduce exact numbers.
"""
import os
import random
import numpy as np


def set_all_seeds(seed=42):
    """
    Lock all random number generators for full reproducibility.

    Parameters
    ----------
    seed : int, default=42
        The random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"[SEED] All seeds locked to {seed}")


def print_section(title, char="═", width=70):
    """
    Print a formatted section header for console output.

    Parameters
    ----------
    title : str
        Section title text.
    char : str, default="═"
        Character used for the horizontal rule.
    width : int, default=70
        Width of the horizontal rule.
    """
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_step(step_num, description):
    """
    Print a numbered pipeline step for progress tracking.

    Parameters
    ----------
    step_num : int or str
        Step number or label (e.g., "FIG" for figure generation steps).
    description : str
        Human-readable description of what this step does.
    """
    if isinstance(step_num, int):
        print(f"  [{step_num:02d}] {description}")
    else:
        print(f"  [{step_num}] {description}")
