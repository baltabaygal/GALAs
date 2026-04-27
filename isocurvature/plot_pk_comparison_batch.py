#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_isocurvature as iso


OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

H_LIST = [1.5, 0.05, 1.0e-4]
MASS_LIST = [1.0e-6, 1.0e-10, 1.0e-15, 1.0e-20]
VW = 0.5
BETA_LIST = [4.0, 40.0]
COLORS = ["#d62728", "#1f77b4"]
LINESTYLES = ["-", ":"]


def sanitize(x: float) -> str:
    s = f"{x:.0e}" if x < 1.0e-3 or x >= 1.0e3 else f"{x:g}"
    return s.replace("-", "m").replace("+", "").replace(".", "p")


def case_row(hstar: float, beta: float, var_no: float) -> dict[str, float]:
    xi = iso.xi_vec(iso.THETA_GRID, hstar, VW, beta)
    rho = iso.rho_from_xi(iso.THETA_GRID, xi)
    _, var, _ = iso.mean_and_var(iso.THETA_GRID, rho)
    tp, t_onset, kcut_ratio = iso.t_onset_and_kcut_ratio(hstar, beta, VW)
    p0_ratio = (var / var_no) * ((t_onset / iso.TOSC) ** 1.5)
    return {
        "beta": beta,
        "var": var,
        "var_ratio": var / var_no,
        "tp": tp,
        "t_onset": t_onset,
        "kcut_ratio": kcut_ratio,
        "p0_ratio": p0_ratio,
    }


def make_plot(hstar: float, mphi_ev: float, var_no: float, k_dat: np.ndarray, p_cdm: np.ndarray) -> Path:
    rows = [case_row(hstar, beta, var_no) for beta in BETA_LIST]
    kcut0 = iso.kcut_no_pt_mpc_inv(mphi_ev)

    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    p_no = iso.p_white_noise(var_no, kcut0, k_dat)
    ax.plot(k_dat, p_cdm, color="#888888", lw=1.6, label=r"$P_{\rm CDM}(k)$ from data")
    ax.plot(k_dat, p_no, color="black", ls="--", label="noPT")
    ax.axvline(kcut0, color="black", ls="--", lw=1.0, alpha=0.8)

    for row, color, ls in zip(rows, COLORS, LINESTYLES):
        kcut = row["kcut_ratio"] * kcut0
        p_pt = iso.p_white_noise(row["var"], kcut, k_dat)
        ax.plot(
            k_dat,
            p_pt,
            color=color,
            ls="-",
            lw=1.8,
            label=rf"$\beta/H_*={row['beta']:g}$  [$P_0$ ratio={row['p0_ratio']:.2f}]",
        )
        ax.axvline(kcut, color=color, ls=ls, lw=1.0)

    ax.text(0.03, 0.95, rf"$H_*/M_\phi = {hstar:g}$", transform=ax.transAxes, va="top")
    ax.text(0.03, 0.87, rf"$v_w = {VW:g}$", transform=ax.transAxes, va="top")
    ax.text(0.97, 0.06, rf"$M_\phi = {mphi_ev:.0e}\ \mathrm{{eV}}$", transform=ax.transAxes, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-4, 1.0e2)
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)$")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")

    out = OUTDIR / f"Pk_comparison_H{sanitize(hstar)}_mphi_{sanitize(mphi_ev)}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rho_no = iso.rho_from_xi(iso.THETA_GRID, np.ones_like(iso.THETA_GRID))
    _, var_no, _ = iso.mean_and_var(iso.THETA_GRID, rho_no)
    k_dat, p_cdm = iso.load_pcdm_dat()
    outputs = []
    for hstar in H_LIST:
        for mphi_ev in MASS_LIST:
            out = make_plot(hstar, mphi_ev, var_no, k_dat, p_cdm)
            outputs.append(out)
            print(out)
    print(f"generated {len(outputs)} plots")


if __name__ == "__main__":
    main()
