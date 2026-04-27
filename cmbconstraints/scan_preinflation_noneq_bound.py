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
    compute_preinflation_noneq,
    hi_over_fphi_bound_from_response,
    build_hybrid_theta_grid,
)


STYLE = apply_paper_style("1col")
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BoundRow:
    theta0: float
    xi: float
    dln_potential: float
    dln_fanh: float
    dln_xi: float
    dln_total_pt: float
    dln_total_standard: float
    hi_over_fphi_max_pt: float
    hi_over_fphi_max_standard: float
    pt_over_standard: float


def build_theta_grid(theta_min: float, theta_max: float, n_theta: int, grid_type: str = "linspace") -> np.ndarray:
    if grid_type == "hybrid":
        return build_hybrid_theta_grid(theta_min, theta_max, n_theta)
    theta_lo = max(float(theta_min), DOMAIN_EPS)
    theta_hi = min(float(theta_max), math.pi - DOMAIN_EPS)
    if theta_hi <= theta_lo:
        raise ValueError("theta grid collapsed; need theta_max > theta_min and theta_max < pi")
    return np.linspace(theta_lo, theta_hi, int(n_theta), dtype=float)


def compute_bound_scan(
    theta_grid: np.ndarray,
    *,
    hstar: float,
    vw: float,
    beta_over_h: float,
    a_s: float,
    alpha_iso_max: float,
) -> list[BoundRow]:
    rows: list[BoundRow] = []
    for theta0 in theta_grid:
        res = compute_preinflation_noneq(
            theta0=float(theta0),
            hstar=hstar,
            vw=vw,
            beta_over_h=beta_over_h,
            hi_over_fphi=1.0e-12,
        )
        dln_standard = float(res.dln_potential + res.dln_fanh)
        hi_pt = hi_over_fphi_bound_from_response(res.dln_rho_total, a_s=a_s, alpha_iso_max=alpha_iso_max)
        hi_std = hi_over_fphi_bound_from_response(dln_standard, a_s=a_s, alpha_iso_max=alpha_iso_max)
        ratio = float(hi_pt / hi_std) if math.isfinite(hi_pt) and math.isfinite(hi_std) and hi_std > 0.0 else float("nan")
        rows.append(
            BoundRow(
                theta0=float(theta0),
                xi=float(res.xi),
                dln_potential=float(res.dln_potential),
                dln_fanh=float(res.dln_fanh),
                dln_xi=float(res.dln_xi),
                dln_total_pt=float(res.dln_rho_total),
                dln_total_standard=dln_standard,
                hi_over_fphi_max_pt=float(hi_pt),
                hi_over_fphi_max_standard=float(hi_std),
                pt_over_standard=ratio,
            )
        )
    return rows


def save_outputs(rows: list[BoundRow], stem: str) -> tuple[Path, Path]:
    csv_path = OUTDIR / f"{stem}.csv"
    json_path = OUTDIR / f"{stem}.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return csv_path, json_path


def make_plot(rows: list[BoundRow], *, hstar: float, vw: float, beta_over_h: float, alpha_iso_max: float, a_s: float, stem: str) -> Path:
    theta = np.array([r.theta0 for r in rows], dtype=float)
    hi_pt = np.array([r.hi_over_fphi_max_pt for r in rows], dtype=float)
    hi_std = np.array([r.hi_over_fphi_max_standard for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(STYLE.width, 2.7), constrained_layout=True)
    c_std, c_pt = viridis_colors(2, start=0.2, end=0.8)
    ax.plot(theta, hi_std, color=c_std, lw=1.8, label=r"standard noPT")
    ax.plot(theta, hi_pt, color=c_pt, lw=1.8, label=r"PT")
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
        "\n".join(
            [
                rf"$H_*/M_\phi={hstar:g}$",
                rf"$v_w={vw:g}$",
                rf"$\beta/H_*={beta_over_h:g}$",
                rf"$\alpha_{{\rm iso}}^{{\max}}={alpha_iso_max:g}$",
            ]
        ),
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
    p = argparse.ArgumentParser(description="Scan the pre-inflation non-equilibrium H_I/f_phi bound over theta0.")
    p.add_argument("--hstar", type=float, required=True, help="H_*/M_phi")
    p.add_argument("--vw", type=float, required=True, help="Wall speed v_w")
    p.add_argument("--betaH", type=float, required=True, help="beta/H_*")
    p.add_argument("--theta-min", type=float, default=0.0, help="Lower theta0 bound in radians. Physical evaluation uses max(theta_min, 1e-8).")
    p.add_argument("--theta-max", type=float, default=math.pi - 0.1, help="Upper theta0 bound in radians.")
    p.add_argument("--n-theta", type=int, default=200, help="Number of theta0 points.")
    p.add_argument("--grid-type", choices=["linspace", "hybrid"], default="linspace", help="Grid distribution type.")
    p.add_argument("--alpha-iso-max", type=float, default=DEFAULT_ALPHA_ISO_MAX, help="Upper bound on the CDM isocurvature fraction.")
    p.add_argument("--A-s", type=float, default=DEFAULT_AS, help="Adiabatic scalar amplitude at the pivot scale.")
    p.add_argument("--stem", type=str, default=None, help="Optional output stem name.")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    theta_grid = build_theta_grid(ns.theta_min, ns.theta_max, ns.n_theta, grid_type=ns.grid_type)
    rows = compute_bound_scan(
        theta_grid,
        hstar=float(ns.hstar),
        vw=float(ns.vw),
        beta_over_h=float(ns.betaH),
        a_s=float(ns.A_s),
        alpha_iso_max=float(ns.alpha_iso_max),
    )
    stem = ns.stem or f"preinflation_noneq_bound_h{ns.hstar:g}_vw{ns.vw:g}_b{ns.betaH:g}".replace(".", "p")
    csv_path, json_path = save_outputs(rows, stem)
    plot_path = make_plot(
        rows,
        hstar=float(ns.hstar),
        vw=float(ns.vw),
        beta_over_h=float(ns.betaH),
        alpha_iso_max=float(ns.alpha_iso_max),
        a_s=float(ns.A_s),
        stem=stem,
    )

    finite_pt = np.array([r.hi_over_fphi_max_pt for r in rows if math.isfinite(r.hi_over_fphi_max_pt)], dtype=float)
    finite_std = np.array([r.hi_over_fphi_max_standard for r in rows if math.isfinite(r.hi_over_fphi_max_standard)], dtype=float)
    print(f"theta range            = [{theta_grid.min():.6g}, {theta_grid.max():.6g}]")
    print(f"hstar                  = {ns.hstar:.6g}")
    print(f"vw                     = {ns.vw:.6g}")
    print(f"beta/H*                = {ns.betaH:.6g}")
    print(f"P_S^max                = {(ns.alpha_iso_max / (1.0 - ns.alpha_iso_max)) * ns.A_s:.6g}")
    print(f"(H_I/f_phi)_max PT     = [{finite_pt.min():.6g}, {finite_pt.max():.6g}]")
    print(f"(H_I/f_phi)_max noPT   = [{finite_std.min():.6g}, {finite_std.max():.6g}]")
    print(f"csv                    = {csv_path}")
    print(f"json                   = {json_path}")
    print(f"plot                   = {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
