"""Consistent, colorblind-safe matplotlib styling for benchmark figures."""

from __future__ import annotations

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "font.size": 10,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
