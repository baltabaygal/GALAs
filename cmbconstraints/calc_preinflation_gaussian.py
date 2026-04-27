#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_plots.style import apply_paper_style, decimal_log_tick_formatter, viridis_colors
from cmbconstraints.calc_preinflation_noneq import (
    DOMAIN_EPS,
    compute_preinflation_noneq,
    fanh_no_pt,
    MODEL,
    xi_value,
)


STYLE = apply_paper_style("1col")
OUTDIR = Path(__file__).resolve().parent / "outputs" / "preinflation_gaussian_debug"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = Path(__file__).resolve().parent / "cache"
CACHEDIR.mkdir(parents=True, exist_ok=True)

THETA_SPLIT = 2.5


@dataclass
class GaussianResult:
    theta0: float
    sigma: float
    rho_bar: float
    p_s_var: float
    p_s_deriv: float
    ratio_var_to_deriv: float
    prob_norm: float
    rho_theta0: float
    rho_bar_over_rho_theta0: float
    max_rho_on_grid: float
    boundary_tail_prob: float


def build_hybrid_theta_grid(
    *,
    theta_min: float = DOMAIN_EPS,
    theta_max: float = math.pi - DOMAIN_EPS,
    n_linear: int = 700,
    n_tail: int = 700,
) -> np.ndarray:
    theta_lo = max(theta_min, DOMAIN_EPS)
    theta_hi = min(theta_max, math.pi - DOMAIN_EPS)
    split = min(max(theta_lo + 1.0e-6, THETA_SPLIT), theta_hi)
    linear = np.linspace(theta_lo, split, n_linear, dtype=float)
    if split >= theta_hi - 1.0e-8:
        return linear
    deltas = np.geomspace(math.pi - split, math.pi - theta_hi, n_tail, dtype=float)
    tail = math.pi - deltas
    return np.unique(np.concatenate([linear, tail]))


def build_master_theta_grid() -> np.ndarray:
    return build_hybrid_theta_grid(n_linear=80, n_tail=120)


def build_adaptive_theta_grid(theta0: float, sigma: float) -> np.ndarray:
    local_half_width = min(max(10.0 * sigma, 2.0e-3), math.pi)
    step = min(max(sigma / 8.0, 2.0e-6), 5.0e-3)
    theta_lo = max(DOMAIN_EPS, theta0 - local_half_width)
    theta_hi = min(math.pi - DOMAIN_EPS, theta0 + local_half_width)
    split = min(max(theta_lo + 1.0e-6, THETA_SPLIT), theta_hi)
    if theta_hi <= split + 1.0e-8:
        n_local = int(min(2501, max(201, math.ceil((theta_hi - theta_lo) / step) + 1)))
        return np.linspace(theta_lo, theta_hi, n_local, dtype=float)
    n_linear = int(min(1200, max(101, math.ceil((split - theta_lo) / step) + 1)))
    linear = np.linspace(theta_lo, split, n_linear, dtype=float)
    n_tail = int(min(1200, max(101, math.ceil((theta_hi - split) / step) + 1)))
    deltas = np.geomspace(math.pi - split, math.pi - theta_hi, n_tail, dtype=float)
    tail = math.pi - deltas
    return np.unique(np.concatenate([linear, tail]))


def rho_pt(theta: np.ndarray, *, hstar: float, vw: float, beta_over_h: float) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    # This Gaussian-smearing case represents inflationary fluctuations around a
    # fixed background theta0. For scan-like workloads we use the batched theta
    # path in xi_model so the geometry interpolation is evaluated on the full
    # theta array at once rather than one scalar theta at a time.
    batch = MODEL.predict_theta_batch(
        theta0_array=theta_arr,
        hstar=hstar,
        vw=vw,
        beta_over_h=beta_over_h,
        clip=True,
        xi_dm_mode="broken_powerlaw_ftilde",
    )
    xi = np.asarray(batch["xi"], dtype=float)
    fanh = np.array([fanh_no_pt(float(th)) for th in theta_arr], dtype=float)
    potential = 1.0 - np.cos(theta_arr)
    return xi * fanh * potential


def rho_no_pt(theta: np.ndarray) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    return np.array([fanh_no_pt(float(th)) * (1.0 - math.cos(float(th))) for th in theta_arr], dtype=float)


def rho_harmonic(theta: np.ndarray) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    return theta_arr**2


def _cache_key(hstar: float, vw: float, beta_over_h: float) -> str:
    return f"h{hstar:.8g}_vw{vw:.8g}_b{beta_over_h:.8g}".replace(".", "p")


def load_or_build_rho_cache(*, hstar: float, vw: float, beta_over_h: float) -> tuple[np.ndarray, np.ndarray, Path]:
    key = _cache_key(hstar, vw, beta_over_h)
    path = CACHEDIR / f"rho_pt_{key}.npz"
    if path.exists():
        data = np.load(path)
        return np.asarray(data["theta"], dtype=float), np.asarray(data["rho"], dtype=float), path
    theta = build_master_theta_grid()
    t0 = time.perf_counter()
    rho = rho_pt(theta, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
    np.savez_compressed(
        path,
        theta=theta,
        rho=rho,
        hstar=hstar,
        vw=vw,
        beta_over_h=beta_over_h,
        build_seconds=time.perf_counter() - t0,
    )
    return theta, rho, path


def gaussian_pdf(theta: np.ndarray, *, theta0: float, sigma: float) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    # Truncated Gaussian over the physical theta domain. This generalizes the
    # non-equilibrium fixed-theta case. In the narrow-width limit it must reduce
    # to the derivative formula. The equilibrium case would use P_eq(theta)
    # instead of this Gaussian.
    weights = np.exp(-0.5 * ((theta_arr - theta0) / sigma) ** 2)
    norm = np.trapezoid(weights, theta_arr)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Gaussian normalization failed at theta0={theta0:g}, sigma={sigma:g}")
    return weights / norm


def compute_gaussian_power(
    rho_fn: Callable[[np.ndarray], np.ndarray],
    *,
    theta0: float,
    sigma: float,
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    theta_grid = build_adaptive_theta_grid(theta0, sigma)
    pdf = gaussian_pdf(theta_grid, theta0=theta0, sigma=sigma)
    rho = rho_fn(theta_grid)
    rho_bar = float(np.trapezoid(pdf * rho, theta_grid))
    frac = (rho - rho_bar) / max(abs(rho_bar), 1.0e-300)
    frac = np.clip(frac, -math.sqrt(sys.float_info.max), math.sqrt(sys.float_info.max))
    frac_sq = np.square(frac, dtype=float)
    frac_sq = np.nan_to_num(frac_sq, nan=sys.float_info.max, posinf=sys.float_info.max, neginf=sys.float_info.max)
    integrand = np.nan_to_num(pdf * frac_sq, nan=sys.float_info.max, posinf=sys.float_info.max, neginf=sys.float_info.max)
    p_s_var = float(np.trapezoid(integrand, theta_grid))
    prob_norm = float(np.trapezoid(pdf, theta_grid))
    tail_mask = theta_grid >= (math.pi - 0.05)
    tail_prob = float(np.trapezoid(pdf[tail_mask], theta_grid[tail_mask])) if np.any(tail_mask) else 0.0
    return rho_bar, p_s_var, prob_norm, float(np.max(rho)), tail_prob, theta_grid, pdf


def compute_gaussian_power_from_cache(
    theta_master: np.ndarray,
    rho_master: np.ndarray,
    *,
    theta0: float,
    sigma: float,
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    theta_grid = build_adaptive_theta_grid(theta0, sigma)
    pdf = gaussian_pdf(theta_grid, theta0=theta0, sigma=sigma)
    rho = np.interp(theta_grid, theta_master, rho_master)
    rho_bar = float(np.trapezoid(pdf * rho, theta_grid))
    frac = (rho - rho_bar) / max(abs(rho_bar), 1.0e-300)
    frac = np.clip(frac, -math.sqrt(sys.float_info.max), math.sqrt(sys.float_info.max))
    frac_sq = np.square(frac, dtype=float)
    frac_sq = np.nan_to_num(frac_sq, nan=sys.float_info.max, posinf=sys.float_info.max, neginf=sys.float_info.max)
    integrand = np.nan_to_num(pdf * frac_sq, nan=sys.float_info.max, posinf=sys.float_info.max, neginf=sys.float_info.max)
    p_s_var = float(np.trapezoid(integrand, theta_grid))
    prob_norm = float(np.trapezoid(pdf, theta_grid))
    tail_mask = theta_grid >= (math.pi - 0.05)
    tail_prob = float(np.trapezoid(pdf[tail_mask], theta_grid[tail_mask])) if np.any(tail_mask) else 0.0
    return rho_bar, p_s_var, prob_norm, float(np.max(rho)), tail_prob, theta_grid, pdf


def compute_pt_result(theta0: float, sigma: float, *, hstar: float, vw: float, beta_over_h: float) -> GaussianResult:
    theta_master, rho_master, _cache_path = load_or_build_rho_cache(hstar=hstar, vw=vw, beta_over_h=beta_over_h)
    rho_bar, p_s_var, prob_norm, max_rho, tail_prob, _grid, _pdf = compute_gaussian_power_from_cache(
        theta_master,
        rho_master,
        theta0=theta0,
        sigma=sigma,
    )
    deriv = compute_preinflation_noneq(
        theta0=theta0,
        hstar=hstar,
        vw=vw,
        beta_over_h=beta_over_h,
        hi_over_fphi=2.0 * math.pi * sigma,
    )
    rho_theta0 = float(np.interp(theta0, theta_master, rho_master))
    return GaussianResult(
        theta0=theta0,
        sigma=sigma,
        rho_bar=rho_bar,
        p_s_var=p_s_var,
        p_s_deriv=float(deriv.p_s),
        ratio_var_to_deriv=float(p_s_var / max(deriv.p_s, 1.0e-300)),
        prob_norm=prob_norm,
        rho_theta0=rho_theta0,
        rho_bar_over_rho_theta0=float(rho_bar / max(rho_theta0, 1.0e-300)),
        max_rho_on_grid=max_rho,
        boundary_tail_prob=tail_prob,
    )


def compute_harmonic_check(theta0: float, sigma: float) -> GaussianResult:
    rho_bar, p_s_var, prob_norm, max_rho, tail_prob, _grid, _pdf = compute_gaussian_power(
        rho_harmonic,
        theta0=theta0,
        sigma=sigma,
    )
    p_s_deriv = float((sigma * (2.0 / max(theta0, 1.0e-300))) ** 2)
    rho_theta0 = float(theta0**2)
    return GaussianResult(
        theta0=theta0,
        sigma=sigma,
        rho_bar=rho_bar,
        p_s_var=p_s_var,
        p_s_deriv=p_s_deriv,
        ratio_var_to_deriv=float(p_s_var / max(p_s_deriv, 1.0e-300)),
        prob_norm=prob_norm,
        rho_theta0=rho_theta0,
        rho_bar_over_rho_theta0=float(rho_bar / max(rho_theta0, 1.0e-300)),
        max_rho_on_grid=max_rho,
        boundary_tail_prob=tail_prob,
    )


def format_result_rows(rows: list[GaussianResult]) -> list[dict[str, float]]:
    return [asdict(r) for r in rows]


def save_table(rows: list[GaussianResult], name: str) -> Path:
    out = OUTDIR / f"{name}.csv"
    data = format_result_rows(rows)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return out


def make_power_plot(
    *,
    theta_scan: np.ndarray,
    sigma_values: list[float],
    hstar: float,
    vw: float,
    beta_over_h: float,
) -> Path:
    colors = viridis_colors(len(sigma_values), start=0.15, end=0.85)
    deriv_vals = []
    sigma_ref = min(sigma_values)
    for theta0 in theta_scan:
        deriv = compute_preinflation_noneq(
            theta0=float(theta0),
            hstar=hstar,
            vw=vw,
            beta_over_h=beta_over_h,
            hi_over_fphi=2.0 * math.pi * sigma_ref,
        )
        deriv_vals.append(deriv.p_s / max((sigma_ref**2), 1.0e-300))
    deriv_vals = np.array(deriv_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(STYLE.width, 2.8), constrained_layout=True)
    for color, sigma in zip(colors, sigma_values):
        vals = []
        for theta0 in theta_scan:
            res = compute_pt_result(float(theta0), sigma, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
            vals.append(res.p_s_var / max(sigma**2, 1.0e-300))
        ax.plot(theta_scan, np.array(vals), color=color, lw=1.6, label=rf"$\sigma={sigma:g}$")
    ax.plot(theta_scan, deriv_vals, color="black", lw=1.5, ls="--", label=r"derivative limit")
    ax.set_yscale("log")
    ax.set_xlim(float(theta_scan.min()), float(theta_scan.max()))
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$P_S/\sigma^2$")
    ax.yaxis.set_major_formatter(decimal_log_tick_formatter())
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False)
    out = OUTDIR / "pt_power_vs_theta.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_ratio_plot(
    *,
    theta_scan: np.ndarray,
    sigma_values: list[float],
    hstar: float,
    vw: float,
    beta_over_h: float,
) -> Path:
    colors = viridis_colors(len(sigma_values), start=0.15, end=0.85)
    fig, ax = plt.subplots(figsize=(STYLE.width, 2.8), constrained_layout=True)
    for color, sigma in zip(colors, sigma_values):
        ratios = []
        for theta0 in theta_scan:
            res = compute_pt_result(float(theta0), sigma, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
            ratios.append(res.ratio_var_to_deriv)
        ax.plot(theta_scan, np.array(ratios), color=color, lw=1.6, label=rf"$\sigma={sigma:g}$")
    ax.axhline(1.0, color="black", lw=1.2, ls="--")
    ax.set_xlim(float(theta_scan.min()), float(theta_scan.max()))
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$P_S^{\rm var}/P_S^{\rm deriv}$")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False)
    out = OUTDIR / "pt_ratio_vs_theta.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def print_debug_table(title: str, rows: list[GaussianResult], *, warning_tol: float = 0.02) -> None:
    print(title)
    print("theta0,sigma,rho_bar,P_S_var,P_S_deriv,ratio")
    for row in rows:
        print(
            f"{row.theta0:.6g},{row.sigma:.6g},{row.rho_bar:.6e},{row.p_s_var:.6e},{row.p_s_deriv:.6e},{row.ratio_var_to_deriv:.6f}"
        )
        if abs(row.ratio_var_to_deriv - 1.0) > warning_tol:
            print(
                f"WARNING narrow-limit mismatch at theta0={row.theta0:.6g}, sigma={row.sigma:.6g}: "
                f"ratio={row.ratio_var_to_deriv:.6f}"
            )


def run_checks(hstar: float, vw: float, beta_over_h: float) -> dict[str, object]:
    narrow_sigmas = [1.0e-6, 1.0e-5, 1.0e-4]
    large_sigmas = [0.3, 1.0]
    theta_narrow = [0.5, 1.5, 3.041592653589793]
    theta_large = [0.5, 3.041592653589793]

    pt_narrow_rows: list[GaussianResult] = []
    harmonic_rows: list[GaussianResult] = []
    large_rows: list[GaussianResult] = []

    for theta0 in theta_narrow:
        for sigma in narrow_sigmas:
            pt_narrow_rows.append(compute_pt_result(theta0, sigma, hstar=hstar, vw=vw, beta_over_h=beta_over_h))
            harmonic_rows.append(compute_harmonic_check(theta0, sigma))

    for theta0 in theta_large:
        for sigma in large_sigmas:
            large_rows.append(compute_pt_result(theta0, sigma, hstar=hstar, vw=vw, beta_over_h=beta_over_h))

    print_debug_table("CHECK 1 + CHECK 3 + CHECK 6: PT narrow-Gaussian limit", pt_narrow_rows)
    print_debug_table("CHECK 2: harmonic narrow-Gaussian limit", harmonic_rows)

    print("CHECK 4: large-sigma normalization and boundary leakage")
    print("theta0,sigma,prob_norm,boundary_tail_prob,rho_bar")
    for row in large_rows:
        print(f"{row.theta0:.6g},{row.sigma:.6g},{row.prob_norm:.8f},{row.boundary_tail_prob:.6e},{row.rho_bar:.6e}")

    print("CHECK 5: hilltop stability")
    print("theta0,sigma,max_rho,rho_bar_over_rho_theta0,P_S_deriv")
    for row in [r for r in pt_narrow_rows + large_rows if r.theta0 >= 2.8]:
        print(
            f"{row.theta0:.6g},{row.sigma:.6g},{row.max_rho_on_grid:.6e},{row.rho_bar_over_rho_theta0:.6f},{row.p_s_deriv:.6e}"
        )
        if not all(math.isfinite(x) for x in [row.max_rho_on_grid, row.rho_bar, row.p_s_var, row.p_s_deriv]):
            print(f"WARNING non-finite hilltop quantity at theta0={row.theta0:.6g}, sigma={row.sigma:.6g}")

    pt_csv = save_table(pt_narrow_rows, "check_pt_narrow")
    harm_csv = save_table(harmonic_rows, "check_harmonic_narrow")
    large_csv = save_table(large_rows, "check_large_sigma")

    theta_scan = np.linspace(0.1, math.pi - 0.1, 10, dtype=float)
    sigma_plot = [1.0e-4, 1.0e-3, 1.0e-2]
    power_plot = make_power_plot(theta_scan=theta_scan, sigma_values=sigma_plot, hstar=hstar, vw=vw, beta_over_h=beta_over_h)
    ratio_plot = make_ratio_plot(theta_scan=theta_scan, sigma_values=sigma_plot, hstar=hstar, vw=vw, beta_over_h=beta_over_h)

    summary = {
        "hstar": hstar,
        "vw": vw,
        "beta_over_h": beta_over_h,
        "comments": [
            "Gaussian P(theta) models inflationary fluctuations around fixed theta0.",
            "sigma = H_I / (2 pi f_phi).",
            "The variance expression must reduce to the derivative formula in the narrow-Gaussian limit.",
            "The equilibrium case would replace the Gaussian by P_eq(theta).",
        ],
        "files": {
            "pt_narrow_csv": str(pt_csv),
            "harmonic_narrow_csv": str(harm_csv),
            "large_sigma_csv": str(large_csv),
            "power_plot": str(power_plot),
            "ratio_plot": str(ratio_plot),
        },
        "max_abs_narrow_mismatch_pt": max(abs(r.ratio_var_to_deriv - 1.0) for r in pt_narrow_rows),
        "max_abs_narrow_mismatch_harmonic": max(abs(r.ratio_var_to_deriv - 1.0) for r in harmonic_rows),
    }
    summary_path = OUTDIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gaussian-smeared pre-inflation CMB isocurvature debug module.")
    p.add_argument("--hstar", type=float, default=0.05, help="H_*/M_phi")
    p.add_argument("--vw", type=float, default=0.8, help="Wall speed v_w")
    p.add_argument("--betaH", type=float, default=4.0, help="beta/H_*")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    summary = run_checks(hstar=float(ns.hstar), vw=float(ns.vw), beta_over_h=float(ns.betaH))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
