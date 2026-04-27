#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
OUTDIR = Path(__file__).resolve().parent / "outputs" / "equilibrium"
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Plot equilibrium isocurvature results.")
    parser.add_argument("--input", type=str, required=True, help="JSON file from calc_preinflation_equilibrium.py")
    parser.add_argument("--alpha-iso-max", type=float, default=0.038, help="CMB limit")
    parser.add_argument("--stem", type=str, default="equilibrium_plot")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return 1

    with input_path.open() as f:
        data = json.load(f)

    b = np.array([r["b_param"] for r in data])
    # Fallback to 'p_s' if the file was generated with the old version
    ps_cmb = np.array([r.get("p_s_cmb", r.get("p_s", 0.0)) for r in data])
    ps_patch = np.array([r.get("p_s_patch", 0.0) for r in data])
    
    # Metadata for labels
    hstar = data[0]["hstar"]
    vw = data[0]["vw"]
    betaH = data[0]["beta_over_h"]

    fig, ax = plt.subplots(figsize=(STYLE.width, 3.5), constrained_layout=True)
    
    # Plot CMB and Patch powers
    ax.plot(b, ps_cmb, color="tab:blue", lw=2.0, label=r"$P_S$ (CMB, local response)")
    if ps_patch.any():
        ax.plot(b, ps_patch, color="tab:green", lw=1.5, ls="--", alpha=0.8, label=r"$P_S$ (Patch variance)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$b \equiv 8\pi^2 V_{\rm max} / (3 H_I^4)$")
    ax.set_ylabel(r"Isocurvature Power $P_S$")
    ax.yaxis.set_major_formatter(decimal_log_tick_formatter())
    ax.xaxis.set_major_formatter(decimal_log_tick_formatter())
    
    # Horizontal line for Planck limit
    as_val = 2.1e-9
    ps_limit = (args.alpha_iso_max * as_val) / (1.0 - args.alpha_iso_max)
    ax.axhline(ps_limit, color="tab:red", ls=":", lw=1.2, label=rf"Planck Limit ($\alpha_{{\rm iso}} < {args.alpha_iso_max}$)")

    ax.legend(loc="upper left", frameon=False, fontsize=STYLE.xtick_labelsize - 1)
    
    ax.text(
        0.03,
        0.04,
        "\n".join(
            [
                rf"$H_*/M_\phi={hstar:g}$",
                rf"$v_w={vw:g}$",
                rf"$\beta/H_*={betaH:g}$",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE.xtick_labelsize,
    )

    out_pdf = OUTDIR / f"{args.stem}.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Plot saved to {out_pdf}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

