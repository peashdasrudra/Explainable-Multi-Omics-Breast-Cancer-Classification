"""
utils.py -- Reproducibility and utility functions.
"""
import os
import random
import numpy as np


def set_all_seeds(seed=42):
    """Lock all random seeds for 100% reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"[SEED] All seeds locked to {seed}")


def print_section(title, char="═", width=70):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_step(step_num, description):
    """Print a numbered step."""
    if isinstance(step_num, int):
        print(f"  [{step_num:02d}] {description}")
    else:
        print(f"  [{step_num}] {description}")
