#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_plots.style import apply_paper_style, decimal_log_tick_formatter, viridis_colors
from cmbconstraints.calc_preinflation_noneq import (
    DEFAULT_ALPHA_ISO_MAX,
    DEFAULT_AS,
    DOMAIN_EPS,
    dln_fanh,
    dln_harmonic_reference,
    dln_no_pt_total,
    dln_potential,
    hi_over_fphi_bound_from_response,
    build_hybrid_theta_grid,
)


STYLE = apply_paper_style("1col")
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


@dataclass
class NoPTBoundRow:
    theta0: float
    dln_potential: float
    dln_fanh: float
    dln_total_no_pt: float
    dln_harmonic: float
    hi_over_fphi_max_no_pt: float
    hi_over_fphi_max_harmonic: float
    no_pt_over_harmonic: float


def build_theta_grid(theta_min: float, theta_max: float, n_theta: int, grid_type: str = "linspace") -> np.ndarray:
    if grid_type == "hybrid":
        return build_hybrid_theta_grid(theta_min, theta_max, n_theta)
    theta_lo = max(float(theta_min), DOMAIN_EPS)
    theta_hi = min(float(theta_max), math.pi - DOMAIN_EPS)
    if theta_hi <= theta_lo:
        raise ValueError("theta grid collapsed; need theta_max > theta_min and theta_max < pi")
    return np.linspace(theta_lo, theta_hi, int(n_theta), dtype=float)


def compute_no_pt_bound_scan(
    theta_grid: np.ndarray,
    *,
    a_s: float,
    alpha_iso_max: float,
) -> list[NoPTBoundRow]:
    rows: list[NoPTBoundRow] = []
    for theta0 in theta_grid:
        theta0_f = float(theta0)
        dpot = dln_potential(theta0_f)
        dfanh = dln_fanh(theta0_f)
        dtotal = dln_no_pt_total(theta0_f)
        dharm = dln_harmonic_reference(theta0_f)
        hi_nopt = hi_over_fphi_bound_from_response(dtotal, a_s=a_s, alpha_iso_max=alpha_iso_max)
        hi_harm = hi_over_fphi_bound_from_response(dharm, a_s=a_s, alpha_iso_max=alpha_iso_max)
        ratio = float(hi_nopt / hi_harm) if math.isfinite(hi_nopt) and math.isfinite(hi_harm) and hi_harm > 0.0 else float("nan")
        rows.append(
            NoPTBoundRow(
                theta0=theta0_f,
                dln_potential=float(dpot),
                dln_fanh=float(dfanh),
                dln_total_no_pt=float(dtotal),
                dln_harmonic=float(dharm),
                hi_over_fphi_max_no_pt=float(hi_nopt),
                hi_over_fphi_max_harmonic=float(hi_harm),
                no_pt_over_harmonic=ratio,
            )
        )
    return rows


def save_outputs(rows: list[NoPTBoundRow], stem: str) -> tuple[Path, Path]:
    csv_path = OUTDIR / f"{stem}.csv"
    json_path = OUTDIR / f"{stem}.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return csv_path, json_path


def make_plot(rows: list[NoPTBoundRow], *, alpha_iso_max: float, stem: str) -> Path:
    theta = np.array([r.theta0 for r in rows], dtype=float)
    hi_nopt = np.array([r.hi_over_fphi_max_no_pt for r in rows], dtype=float)
    hi_harm = np.array([r.hi_over_fphi_max_harmonic for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(STYLE.width, 2.7), constrained_layout=True)
    c_harm, c_nopt = viridis_colors(2, start=0.15, end=0.75)
    ax.plot(theta, hi_harm, color=c_harm, lw=1.8, ls="--", label=r"harmonic reference")
    ax.plot(theta, hi_nopt, color=c_nopt, lw=1.8, label=r"exact noPT")
    ax.set_yscale("log")
    ax.set_xlim(float(theta.min()), float(theta.max()))
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\left(H_I/f_\phi\right)_{\max}$")
    ax.yaxis.set_major_formatter(decimal_log_tick_formatter())
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.03,
        0.04,
        rf"$\alpha_{{\rm iso}}^{{\max}}={alpha_iso_max:g}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE.xtick_labelsize,
    )
    out = OUTDIR / f"{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan the standard noPT pre-inflation H_I/f_phi bound over theta0.")
    p.add_argument("--theta-min", type=float, default=0.0, help="Lower theta0 bound in radians. Physical evaluation uses max(theta_min, 1e-8).")
    p.add_argument("--theta-max", type=float, default=math.pi - 0.1, help="Upper theta0 bound in radians.")
    p.add_argument("--n-theta", type=int, default=5, help="Number of theta0 points.")
    p.add_argument("--grid-type", choices=["linspace", "hybrid"], default="linspace", help="Grid distribution type.")
    p.add_argument("--alpha-iso-max", type=float, default=DEFAULT_ALPHA_ISO_MAX, help="Upper bound on the CDM isocurvature fraction.")
    p.add_argument("--A-s", type=float, default=DEFAULT_AS, help="Adiabatic scalar amplitude at the pivot scale.")
    p.add_argument("--stem", type=str, default="nopt_reference_bound", help="Output stem name.")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    theta_grid = build_theta_grid(ns.theta_min, ns.theta_max, ns.n_theta, grid_type=ns.grid_type)
    rows = compute_no_pt_bound_scan(
        theta_grid,
        a_s=float(ns.A_s),
        alpha_iso_max=float(ns.alpha_iso_max),
    )
    csv_path, json_path = save_outputs(rows, ns.stem)
    plot_path = make_plot(rows, alpha_iso_max=float(ns.alpha_iso_max), stem=ns.stem)

    finite_nopt = np.array([r.hi_over_fphi_max_no_pt for r in rows if math.isfinite(r.hi_over_fphi_max_no_pt)], dtype=float)
    finite_harm = np.array([r.hi_over_fphi_max_harmonic for r in rows if math.isfinite(r.hi_over_fphi_max_harmonic)], dtype=float)
    print(f"theta range               = [{theta_grid.min():.6g}, {theta_grid.max():.6g}]")
    print(f"P_S^max                   = {(ns.alpha_iso_max / (1.0 - ns.alpha_iso_max)) * ns.A_s:.6g}")
    print(f"(H_I/f_phi)_max exact     = [{finite_nopt.min():.6g}, {finite_nopt.max():.6g}]")
    print(f"(H_I/f_phi)_max harmonic  = [{finite_harm.min():.6g}, {finite_harm.max():.6g}]")
    print(f"csv                       = {csv_path}")
    print(f"json                      = {json_path}")
    print(f"plot                      = {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
