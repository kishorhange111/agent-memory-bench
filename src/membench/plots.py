"""Plots for the three sweeps (matplotlib optional)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_all(agg_b: list[dict], agg_l: list[dict], agg_e: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

    for name in dict.fromkeys(r["strategy"] for r in agg_b):
        for reader, ls in (("oracle", "-"), ("gullible", ":")):
            rs = sorted(
                (r for r in agg_b if r["strategy"] == name and r["reader"] == reader),
                key=lambda r: r["budget"],
            )
            ax[0].plot(
                [r["budget"] for r in rs],
                [100 * r["overall"] for r in rs],
                ls,
                marker="o",
                label=name if reader == "oracle" else None,
                color=f"C{list(dict.fromkeys(x['strategy'] for x in agg_b)).index(name)}",
            )
    ax[0].set(
        xscale="log",
        xlabel="context budget (tokens)",
        ylabel="accuracy (%)",
        title="accuracy vs budget (solid: oracle reader, dotted: gullible)",
    )
    ax[0].legend(fontsize=8)

    for i, name in enumerate(dict.fromkeys(r["strategy"] for r in agg_l)):
        rs = sorted((r for r in agg_l if r["strategy"] == name), key=lambda r: r["sessions"])
        ax[1].plot(
            [r["sessions"] for r in rs],
            [100 * r["overall"] for r in rs],
            marker="o",
            label=name,
            color=f"C{i}",
        )
    ax[1].set(
        xscale="log",
        xlabel="history length (sessions)",
        ylabel="accuracy (%)",
        title="accuracy vs history length (budget 256)",
    )

    for name, ls in (("fact_store", "-"), ("hybrid", "--")):
        for j, c in enumerate(sorted({r["confusion"] for r in agg_e})):
            rs = sorted(
                (r for r in agg_e if r["strategy"] == name and r["confusion"] == c),
                key=lambda r: r["miss"],
            )
            ax[2].plot(
                [100 * r["miss"] for r in rs],
                [100 * r["overall"] for r in rs],
                ls,
                marker="o",
                color=f"C{j}",
                label=f"{name}, confusion {c:.0%}",
            )
    ax[2].set(
        xlabel="extractor miss rate (%)",
        ylabel="accuracy (%)",
        title="structured memory vs extractor quality",
    )
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sweeps.png", dpi=120)
    plt.close(fig)
