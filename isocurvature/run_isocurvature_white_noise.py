#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_isocurvature_white_noise")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib

from paper_plots.style import apply_paper_style, get_style

STYLE_1COL = apply_paper_style("1col")
STYLE_2COL = get_style("2col")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XI_MODEL_DIR = ROOT / "paper_codes" / "xi_model"
if str(XI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(XI_MODEL_DIR))

from xi_model import load_default_model


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

PK_PATH = ROOT / "paper_codes" / "Pk_CDM.dat"
STATUS_PATH = ROOT / "status.md"

H_LIST = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
VW_LIST = [0.3, 0.5, 0.7, 0.9]
BETA_LIST = [4.0, 8.0, 12.0, 20.0, 40.0]
THETA_GRID = np.linspace(0.01, np.pi - 0.01, 500, dtype=float)
THETA0_REF = 1.0
KREF_MPC = 1.0
TOSC_NO_PT = 1.5
FAST_PT_BETA_REF = 1.0e6
FIG_W = STYLE_1COL.width
FIG_H = 2.55


def save_both(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def fanh_no_pt(theta: np.ndarray) -> np.ndarray:
    cos2 = np.cos(theta / 2.0) ** 2
    base = 1.0 - np.log(np.maximum(cos2, 1.0e-300))
    return np.power(base, 2.216)


def delta2_from_rho(rho: np.ndarray, theta: np.ndarray) -> float:
    rho_bar = float((1.0 / np.pi) * np.trapezoid(rho, theta))
    delta_sq = ((rho - rho_bar) / max(abs(rho_bar), 1.0e-300)) ** 2
    return float((1.0 / np.pi) * np.trapezoid(delta_sq, theta))


def load_cdm_full_and_tail() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(PK_PATH)
    k_mpc = 1000.0 * np.asarray(arr[:, 0], dtype=float)
    p_cdm = np.asarray(arr[:, 1], dtype=float)
    peak_idx = int(np.argmax(p_cdm))
    return k_mpc, p_cdm, k_mpc[peak_idx:], p_cdm[peak_idx:]


def loglog_interp(x: np.ndarray, y: np.ndarray, x_new: float) -> float:
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.maximum(np.asarray(y, dtype=float), 1.0e-300))
    return float(np.exp(np.interp(np.log(float(x_new)), lx, ly)))


def invert_cdm_tail(k_tail: np.ndarray, p_tail: np.ndarray, target_p: float) -> float:
    # tail is monotone decreasing in P with increasing k
    rev_p = p_tail[::-1]
    rev_k = k_tail[::-1]
    lp = np.log(np.maximum(rev_p, 1.0e-300))
    lk = np.log(np.maximum(rev_k, 1.0e-300))
    lt = np.log(max(float(target_p), 1.0e-300))
    if lt <= lp[0]:
        i0, i1 = 0, 1
        slope = (lk[i1] - lk[i0]) / (lp[i1] - lp[i0])
        return float(np.exp(lk[i0] + slope * (lt - lp[i0])))
    if lt >= lp[-1]:
        i0, i1 = -2, -1
        slope = (lk[i1] - lk[i0]) / (lp[i1] - lp[i0])
        return float(np.exp(lk[i0] + slope * (lt - lp[i0])))
    return float(np.exp(np.interp(lt, lp, lk)))


def compute_point(model, hstar: float, vw: float, beta_over_h: float, delta2_no_pt: float) -> dict[str, float]:
    xi_vals = np.array(
        [model._eval_core(theta0=float(theta), vw=vw, hstar=hstar, beta_over_h=beta_over_h)["xi"] for theta in THETA_GRID],
        dtype=float,
    )
    xi_fast_vals = np.array(
        [model._eval_core(theta0=float(theta), vw=vw, hstar=hstar, beta_over_h=FAST_PT_BETA_REF)["xi"] for theta in THETA_GRID],
        dtype=float,
    )
    xi_rel = xi_vals / np.maximum(xi_fast_vals, 1.0e-300)
    rho_common = (1.0 - np.cos(THETA_GRID)) * fanh_no_pt(THETA_GRID)
    rho_pt = xi_rel * rho_common
    delta2_pt = delta2_from_rho(rho_pt, THETA_GRID)

    tp_pt = float(model._eval_core(theta0=THETA0_REF, vw=vw, hstar=hstar, beta_over_h=beta_over_h)["tp"])
    tp_no_pt = float(TOSC_NO_PT)
    t_eff_pt = float(max(tp_no_pt, tp_pt))
    p0_ratio = float((delta2_pt / max(delta2_no_pt, 1.0e-300)) * ((t_eff_pt / max(tp_no_pt, 1.0e-300)) ** 1.5))
    return {
        "H": float(hstar),
        "v_w": float(vw),
        "beta_over_H": float(beta_over_h),
        "delta2_pt": float(delta2_pt),
        "delta2_no_pt": float(delta2_no_pt),
        "xi_fast_ref_beta_over_H": float(FAST_PT_BETA_REF),
        "tp_pt": float(tp_pt),
        "tp_no_pt": float(tp_no_pt),
        "t_eff_pt": float(t_eff_pt),
        "p0_ratio_pt_to_no_pt": float(p0_ratio),
    }


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_pk_comparison(rows: list[dict[str, float]], k_full: np.ndarray, p_full: np.ndarray, k_tail: np.ndarray, p_tail: np.ndarray) -> dict[str, float]:
    apply_paper_style("2col")
    p_ref = loglog_interp(k_tail, p_tail, KREF_MPC)
    p0_no_pt = p_ref
    h_benchmarks = [0.05, 2.0]
    beta_lo = 4.0
    beta_hi = 40.0
    vw_ref = 0.5
    colors = ["#d62728", "#1f77b4"]

    fig, axes = plt.subplots(1, 2, figsize=(STYLE_2COL.width, 2.6), sharex=True, sharey=True, constrained_layout=True)
    benchmark_payload: dict[str, dict[str, float]] = {}

    for ax, hstar, color in zip(axes, h_benchmarks, colors):
        row_lo = next(r for r in rows if abs(r["H"] - hstar) < 1e-12 and abs(r["v_w"] - vw_ref) < 1e-12 and abs(r["beta_over_H"] - beta_lo) < 1e-12)
        row_hi = next(r for r in rows if abs(r["H"] - hstar) < 1e-12 and abs(r["v_w"] - vw_ref) < 1e-12 and abs(r["beta_over_H"] - beta_hi) < 1e-12)
        p0_pt_lo = float(row_lo["p0_ratio_pt_to_no_pt"] * p_ref)
        p0_pt_hi = float(row_hi["p0_ratio_pt_to_no_pt"] * p_ref)
        p_band_lo = min(p0_pt_lo, p0_pt_hi)
        p_band_hi = max(p0_pt_lo, p0_pt_hi)
        k_eq_lo = invert_cdm_tail(k_tail, p_tail, p0_pt_lo)
        k_eq_hi = invert_cdm_tail(k_tail, p_tail, p0_pt_hi)

        ax.plot(k_full, p_full, color="black", lw=1.1, label=r"$P_{\rm CDM}(k)$")
        ax.axhline(p0_no_pt, color="#444444", lw=0.95, ls="--", label=r"$P_0^{\rm noPT}$")
        ax.fill_between(k_full, p_band_lo, p_band_hi, color=color, alpha=0.20, label=r"$P_0^{\rm PT}$, $\beta/H_*\in[4,40]$")
        ax.axhline(p0_pt_lo, color=color, lw=0.8, ls="-")
        ax.axhline(p0_pt_hi, color=color, lw=0.8, ls=":")
        ax.axvline(k_eq_lo, color=color, lw=0.75, ls="-", alpha=0.9)
        ax.axvline(k_eq_hi, color=color, lw=0.75, ls=":", alpha=0.9)
        ax.text(0.03, 0.94, rf"$H_*/M_\phi={hstar:g}$", transform=ax.transAxes, va="top")
        ax.text(0.03, 0.86, rf"$\beta/H_*=4$", color=color, transform=ax.transAxes, va="top")
        ax.text(0.03, 0.78, rf"$\beta/H_*=40$", color=color, transform=ax.transAxes, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
        benchmark_payload[f"H_{hstar:g}"] = {
            "H": float(hstar),
            "v_w": float(vw_ref),
            "beta_low": float(beta_lo),
            "beta_high": float(beta_hi),
            "p0_ratio_beta_low": float(row_lo["p0_ratio_pt_to_no_pt"]),
            "p0_ratio_beta_high": float(row_hi["p0_ratio_pt_to_no_pt"]),
            "k_eq_ratio_beta_low": float(k_eq_lo / KREF_MPC),
            "k_eq_ratio_beta_high": float(k_eq_hi / KREF_MPC),
            "k_cut_ratio_beta_low": float(row_lo["k_cut_ratio_pt_to_no_pt"]),
            "k_cut_ratio_beta_high": float(row_hi["k_cut_ratio_pt_to_no_pt"]),
        }

    axes[0].set_ylabel(r"$P(k)$")
    axes[1].legend(loc="lower left", frameon=True, fancybox=False, edgecolor="black")
    save_both(fig, OUTDIR / "Pk_comparison")
    plt.close(fig)
    return benchmark_payload


def plot_heatmap(rows: list[dict[str, float]]) -> None:
    apply_paper_style("2col")
    h_vals = H_LIST
    beta_vals = BETA_LIST
    fig, axes = plt.subplots(2, 2, figsize=(STYLE_2COL.width, STYLE_2COL.height), constrained_layout=True, sharex=True, sharey=True)
    for ax, vw in zip(axes.ravel(), VW_LIST):
        grid = np.zeros((len(h_vals), len(beta_vals)), dtype=float)
        for i, h in enumerate(h_vals):
            for j, beta in enumerate(beta_vals):
                row = next(r for r in rows if abs(r["H"] - h) < 1e-12 and abs(r["v_w"] - vw) < 1e-12 and abs(r["beta_over_H"] - beta) < 1e-12)
                grid[i, j] = float(row["p0_ratio_pt_to_no_pt"])
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[0, len(beta_vals) - 1, 0, len(h_vals) - 1],
        )
        ax.set_title(rf"$v_w={vw:.1f}$")
        ax.set_xticks(range(len(beta_vals)), [f"{b:g}" for b in beta_vals], rotation=0)
        ax.set_yticks(range(len(h_vals)), [f"{h:g}" for h in h_vals])
    axes[1, 0].set_xlabel(r"$\beta/H_*$")
    axes[1, 1].set_xlabel(r"$\beta/H_*$")
    axes[0, 0].set_ylabel(r"$H_*/M_\phi$")
    axes[1, 0].set_ylabel(r"$H_*/M_\phi$")
    cbar = fig.colorbar(im, ax=axes, shrink=0.92)
    cbar.set_label(r"$P_0^{\rm PT}/P_0^{\rm noPT}$")
    save_both(fig, OUTDIR / "P0_ratio_heatmap")
    plt.close(fig)


def plot_keq_shift(rows: list[dict[str, float]], k_tail: np.ndarray, p_tail: np.ndarray) -> None:
    apply_paper_style("1col")
    p_ref = loglog_interp(k_tail, p_tail, KREF_MPC)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(H_LIST)))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    for color, h in zip(colors, H_LIST):
        sub = sorted(
            [r for r in rows if abs(r["H"] - h) < 1e-12 and abs(r["v_w"] - 0.5) < 1.0e-12],
            key=lambda x: x["beta_over_H"],
        )
        x = np.array([r["beta_over_H"] for r in sub], dtype=float)
        y = np.array([invert_cdm_tail(k_tail, p_tail, float(r["p0_ratio_pt_to_no_pt"] * p_ref)) / KREF_MPC for r in sub], dtype=float)
        ax.plot(x, y, "-o", color=color, lw=1.0, ms=2.2, label=rf"$H_*/M_\phi={h:g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta/H_*$")
    ax.set_ylabel(r"$k_{\rm eq,iso}^{\rm PT}/k_{\rm eq,iso}^{\rm noPT}$")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    save_both(fig, OUTDIR / "keq_shift_vs_betaH")
    plt.close(fig)


def append_status(summary: dict[str, object]) -> None:
    text = STATUS_PATH.read_text() if STATUS_PATH.exists() else ""
    bench = summary["benchmark"]
    block = (
        "\n## Task 84 — Isocurvature White Noise vs CDM\n"
        f"- outputs: `paper_codes/isocurvature/outputs`\n"
        f"- H*=0.05, vw=0.5: P0 ratios [beta=4,40] = [{bench['H_0.05']['p0_ratio_beta_low']:.4g}, {bench['H_0.05']['p0_ratio_beta_high']:.4g}]\n"
        f"- H*=2.0, vw=0.5: P0 ratios [beta=4,40] = [{bench['H_2']['p0_ratio_beta_low']:.4g}, {bench['H_2']['p0_ratio_beta_high']:.4g}]\n"
        f"- no-PT reference: `t_osc = {TOSC_NO_PT:g} M_phi^-1`\n"
        "- keq normalization convention: `k_eq,noPT = 1 Mpc^-1` on CDM tail, PT shift from corrected P0 ratio\n"
    )
    if block not in text:
        STATUS_PATH.write_text(text.rstrip() + "\n" + block)


def main() -> None:
    model = load_default_model()
    rho_no_pt = (1.0 - np.cos(THETA_GRID)) * fanh_no_pt(THETA_GRID)
    delta2_no_pt = delta2_from_rho(rho_no_pt, THETA_GRID)
    rows: list[dict[str, float]] = []
    for h in H_LIST:
        for vw in VW_LIST:
            for beta in BETA_LIST:
                rows.append(compute_point(model, h, vw, beta, delta2_no_pt))

    k_full, p_full, k_tail, p_tail = load_cdm_full_and_tail()
    for row in rows:
        row["k_eq_ratio_pt_to_no_pt"] = float(
            invert_cdm_tail(k_tail, p_tail, float(row["p0_ratio_pt_to_no_pt"] * loglog_interp(k_tail, p_tail, KREF_MPC))) / KREF_MPC
        )
        row["k_cut_ratio_pt_to_no_pt"] = float((row["tp_no_pt"] / max(row["t_eff_pt"], 1.0e-300)) ** 0.5)

    write_csv(rows, OUTDIR / "isocurvature_summary.csv")
    benchmark = plot_pk_comparison(rows, k_full, p_full, k_tail, p_tail)
    plot_heatmap(rows)
    plot_keq_shift(rows, k_tail, p_tail)

    summary = {
        "status": "ok",
        "theta_grid_min": float(THETA_GRID.min()),
        "theta_grid_max": float(THETA_GRID.max()),
        "n_theta": int(len(THETA_GRID)),
        "delta2_no_pt": float(delta2_no_pt),
        "t_osc_no_pt": float(TOSC_NO_PT),
        "k_ref_no_pt_mpc_inv": float(KREF_MPC),
        "benchmark": benchmark,
        "outputs": {
            "csv": str((OUTDIR / "isocurvature_summary.csv").resolve()),
            "pk_comparison": str((OUTDIR / "Pk_comparison.pdf").resolve()),
            "p0_heatmap": str((OUTDIR / "P0_ratio_heatmap.pdf").resolve()),
            "keq_shift": str((OUTDIR / "keq_shift_vs_betaH.pdf").resolve()),
        },
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    append_status(summary)


if __name__ == "__main__":
    main()
