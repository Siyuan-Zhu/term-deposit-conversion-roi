"""Two-panel figure: gain curve and net value versus call threshold."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.policy import UnitEconomics, gain_curve, threshold_frontier  # noqa: E402


def make_figure(y, p_hat, path: Path, econ: UnitEconomics = UnitEconomics()) -> Path:
    share, captured = gain_curve(y, p_hat)
    frontier = threshold_frontier(y, p_hat, econ)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot(share * 100, captured * 100, color="#1f4e79", label="Model")
    a.plot([0, 100], [0, 100], color="#999999", linestyle="--", label="Random")
    a.set_xlabel("Prospects called (%)")
    a.set_ylabel("Conversions captured (%)")
    a.set_title("(a) Cumulative gain, held-out test set")
    a.legend(frameon=False)
    b.plot([f["threshold"] for f in frontier], [f["net_value"] for f in frontier], color="#1f4e79")
    b.axvline(econ.breakeven_prob, color="#b03a2e", linestyle=":", label=f"break-even p = {econ.breakeven_prob:.2f}")
    b.set_xlabel("Call threshold on calibrated p_hat")
    b.set_ylabel("Net value (relative units)")
    b.set_title("(b) Net value by threshold")
    b.legend(frameon=False)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)
    return path
