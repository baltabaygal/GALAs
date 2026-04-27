#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_plots.style import apply_paper_style, decimal_log_tick_formatter, viridis_colors


STYLE = apply_paper_style("1col")
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_scan(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    return {
        "theta0": np.array([float(r["theta0"]) for r in rows], dtype=float),
        "pt": np.array([float(r["hi_over_fphi_max_pt"]) for r in rows], dtype=float),
        "nopt": np.array([float(r["hi_over_fphi_max_standard"]) for r in rows], dtype=float),
        "ratio": np.array([float(r["pt_over_standard"]) for r in rows], dtype=float),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Overlay pre-inflation PT/noPT H_I/f_phi bounds for several H_*/M_phi values.")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV files from scan_preinflation_noneq_bound.py, ordered by increasing H_*/M_phi.",
    )
    p.add_argument(
        "--hstars",
        nargs="+",
        required=True,
        help="Labels for the H_*/M_phi values corresponding to --inputs.",
    )
    p.add_argument("--vw", type=float, required=True, help="Wall speed label.")
    p.add_argument("--betaH", type=float, required=True, help="beta/H_* label.")
    p.add_argument("--stem", default="preinflation_noneq_bound_hscan", help="Output stem.")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    if len(ns.inputs) != len(ns.hstars):
        raise ValueError("--inputs and --hstars must have the same length")
    scans = [load_scan(Path(p)) for p in ns.inputs]

    fig, ax = plt.subplots(figsize=(STYLE.width, 2.9), constrained_layout=True)
    colors = viridis_colors(len(scans), start=0.2, end=0.85)

    nopt_theta = scans[0]["theta0"]
    nopt = scans[0]["nopt"]
    ax.plot(nopt_theta, nopt, color="black", lw=1.6, ls="--", label="noPT")

    for color, hlabel, scan in zip(colors, ns.hstars, scans):
        ax.plot(scan["theta0"], scan["pt"], color=color, lw=1.8, label=rf"PT, $H_*/M_\phi={hlabel}$")

    ax.set_yscale("log")
    ax.set_xlim(float(nopt_theta.min()), float(nopt_theta.max()))
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\left(H_I/f_\phi\right)_{\max}$")
    ax.yaxis.set_major_formatter(decimal_log_tick_formatter())
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.03,
        0.04,
        "\n".join(
            [
                rf"$v_w={ns.vw:g}$",
                rf"$\beta/H_*={ns.betaH:g}$",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE.xtick_labelsize,
    )

    out = OUTDIR / f"{ns.stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
