"""Shared matplotlib/seaborn configuration for headless, reproducible figures.

Importing this module forces the non-interactive ``Agg`` backend (so plots save
correctly from a CLI with no display) and applies a consistent, colour-blind
friendly style. Every figure in the project is saved through
:func:`save_figure`, which also closes the figure to keep memory flat when many
plots are produced in one run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import seaborn as sns  # noqa: E402

# A small, colour-blind-friendly palette used consistently across the project.
LEGIT_COLOR = "#2e86ab"   # blue  -> legitimate / majority class
FRAUD_COLOR = "#d1495b"   # red   -> fraud / minority class
ACCENT_COLOR = "#edae49"  # amber accent
CLASS_PALETTE = {0: LEGIT_COLOR, 1: FRAUD_COLOR}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def save_figure(fig: "plt.Figure", path: str | Path) -> Path:
    """Save *fig* to *path* (creating parent dirs) and close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
