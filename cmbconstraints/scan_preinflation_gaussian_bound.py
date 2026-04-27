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
    hi_over_fphi_bound_from_response,
)
from cmbconstraints.calc_preinflation_gaussian import compute_pt_result


STYLE = apply_paper_style("1col")
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GaussianBoundRow:
    theta0: float
    sigma_max: float
    hi_over_fphi_max_gaussian: float
    hi_over_fphi_max_deriv: float
    gaussian_over_deriv: float
    ps_at_sigma_max: float
    ps_limit: float
    rho_bar: float


def isocurvature_power_limit(a_s: float, alpha_iso_max: float) -> float:
    return float((alpha_iso_max / (1.0 - alpha_iso_max)) * a_s)


def build_theta_grid(theta_min: float, theta_max: float, n_theta: int) -> np.ndarray:
    theta_lo = max(float(theta_min), DOMAIN_EPS)
    theta_hi = min(float(theta_max), math.pi - 0.1)
    if theta_hi <= theta_lo:
        raise ValueError("theta grid collapsed; need theta_max > theta_min and theta_max < pi")
    return np.linspace(theta_lo, theta_hi, int(n_theta), dtype=float)


def solve_sigma_max(
    theta0: float,
    *,
    hstar: float,
    vw: float,
    beta_over_h: float,
    p_s_limit: float,
    sigma_lo: float = 1.0e-8,
    sigma_hi: float = 1.0,
    rtol: float = 5.0e-3,
    max_iter: int = 40,
) -> tuple[float, float, float]:
    lo = float(sigma_lo)
    hi = float(sigma_hi)
    res_lo = compute_pt_result(theta0, lo, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
    if res_lo.p_s_var > p_s_limit:
        return lo, res_lo.p_s_var, res_lo.rho_bar
    res_hi = compute_pt_result(theta0, hi, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
    expand = 0
    while res_hi.p_s_var < p_s_limit and hi < 3.0 and expand < 6:
        hi *= 2.0
        res_hi = compute_pt_result(theta0, hi, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
        expand += 1
    if res_hi.p_s_var < p_s_limit:
        return hi, res_hi.p_s_var, res_hi.rho_bar

    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        res_mid = compute_pt_result(theta0, mid, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
        if abs(res_mid.p_s_var / p_s_limit - 1.0) < rtol:
            return mid, res_mid.p_s_var, res_mid.rho_bar
        if res_mid.p_s_var < p_s_limit:
            lo = mid
        else:
            hi = mid
    final = math.sqrt(lo * hi)
    res_final = compute_pt_result(theta0, final, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
    return final, res_final.p_s_var, res_final.rho_bar


def compute_bound_scan(
    theta_grid: np.ndarray,
    *,
    hstar: float,
    vw: float,
    beta_over_h: float,
    a_s: float,
    alpha_iso_max: float,
) -> list[GaussianBoundRow]:
    p_s_limit = isocurvature_power_limit(a_s, alpha_iso_max)
    rows: list[GaussianBoundRow] = []
    for theta0 in theta_grid:
        sigma_max, p_s_at_sigma, rho_bar = solve_sigma_max(
            float(theta0),
            hstar=hstar,
            vw=vw,
            beta_over_h=beta_over_h,
            p_s_limit=p_s_limit,
        )
        hi_gauss = float(2.0 * math.pi * sigma_max)
        pt_deriv_bound = hi_over_fphi_bound_from_response(
            compute_pt_result(float(theta0), 1.0e-6, hstar=hstar, vw=vw, beta_over_h=beta_over_h).p_s_deriv**0.5 / 1.0e-6,
            a_s=a_s,
            alpha_iso_max=alpha_iso_max,
        )
        rows.append(
            GaussianBoundRow(
                theta0=float(theta0),
                sigma_max=float(sigma_max),
                hi_over_fphi_max_gaussian=hi_gauss,
                hi_over_fphi_max_deriv=float(pt_deriv_bound),
                gaussian_over_deriv=float(hi_gauss / max(pt_deriv_bound, 1.0e-300)),
                ps_at_sigma_max=float(p_s_at_sigma),
                ps_limit=float(p_s_limit),
                rho_bar=float(rho_bar),
            )
        )
    return rows


def save_outputs(rows: list[GaussianBoundRow], stem: str) -> tuple[Path, Path]:
    csv_path = OUTDIR / f"{stem}.csv"
    json_path = OUTDIR / f"{stem}.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return csv_path, json_path


def make_plot(rows: list[GaussianBoundRow], *, hstar: float, vw: float, beta_over_h: float, stem: str) -> Path:
    theta = np.array([r.theta0 for r in rows], dtype=float)
    hi_gauss = np.array([r.hi_over_fphi_max_gaussian for r in rows], dtype=float)
    hi_deriv = np.array([r.hi_over_fphi_max_deriv for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(STYLE.width, 2.7), constrained_layout=True)
    c_deriv, c_gauss = viridis_colors(2, start=0.2, end=0.8)
    ax.plot(theta, hi_deriv, color=c_deriv, lw=1.8, ls="--", label=r"derivative limit")
    ax.plot(theta, hi_gauss, color=c_gauss, lw=1.8, label=r"Gaussian")
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
    p = argparse.ArgumentParser(description="Scan the Gaussian pre-inflation H_I/f_phi bound over theta0.")
    p.add_argument("--hstar", type=float, required=True, help="H_*/M_phi")
    p.add_argument("--vw", type=float, required=True, help="Wall speed v_w")
    p.add_argument("--betaH", type=float, required=True, help="beta/H_*")
    p.add_argument("--theta-min", type=float, default=0.0, help="Lower theta0 bound in radians.")
    p.add_argument("--theta-max", type=float, default=math.pi - 0.1, help="Upper theta0 bound in radians.")
    p.add_argument("--n-theta", type=int, default=10, help="Number of theta0 points.")
    p.add_argument("--alpha-iso-max", type=float, default=DEFAULT_ALPHA_ISO_MAX, help="Upper bound on the CDM isocurvature fraction.")
    p.add_argument("--A-s", type=float, default=DEFAULT_AS, help="Adiabatic scalar amplitude at the pivot scale.")
    p.add_argument("--stem", type=str, default=None, help="Optional output stem name.")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    theta_grid = build_theta_grid(ns.theta_min, ns.theta_max, ns.n_theta)
    rows = compute_bound_scan(
        theta_grid,
        hstar=float(ns.hstar),
        vw=float(ns.vw),
        beta_over_h=float(ns.betaH),
        a_s=float(ns.A_s),
        alpha_iso_max=float(ns.alpha_iso_max),
    )
    stem = ns.stem or f"preinflation_gaussian_bound_h{ns.hstar:g}_vw{ns.vw:g}_b{ns.betaH:g}_{ns.n_theta}pt".replace(".", "p")
    csv_path, json_path = save_outputs(rows, stem)
    plot_path = make_plot(rows, hstar=float(ns.hstar), vw=float(ns.vw), beta_over_h=float(ns.betaH), stem=stem)

    print("theta0,sigma_max,HI/fphi_max_gaussian,HI/fphi_max_deriv,ratio")
    for row in rows:
        print(
            f"{row.theta0:.6g},{row.sigma_max:.6e},{row.hi_over_fphi_max_gaussian:.6e},"
            f"{row.hi_over_fphi_max_deriv:.6e},{row.gaussian_over_deriv:.6f}"
        )
    print(f"csv = {csv_path}")
    print(f"json = {json_path}")
    print(f"plot = {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
