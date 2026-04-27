#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XI_MODEL_DIR = ROOT / "paper_codes" / "xi_model"
if str(XI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(XI_MODEL_DIR))

from xi_model import load_default_model


MODEL = load_default_model()
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

A0_FANH = 0.373
GAMMA0_FANH = 1.20
THETA_MIN = 0.262
THETA_MAX = 2.88
THETA = np.linspace(THETA_MIN, THETA_MAX, 120)
THETA_SCAN = np.linspace(0.262, 2.88, 7)
THETA_TABS = np.array([0.261799, 0.785398, 1.308997, 1.832596, 2.356194, 2.879793], dtype=float)

BENCHMARKS = [
    ("DM-dominated", 0.2, 0.5, 8.0, "#0077BB"),
    ("BM-dominated", 0.05, 0.5, 4.0, "#EE7733"),
    ("Moderate", 1.0, 0.5, 20.0, "#009988"),
]
H_SCAN = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
VW_SCAN = 0.5
BETA_SCAN = 8.0


def fanh_no_pt(theta: np.ndarray) -> np.ndarray:
    cos2 = np.cos(theta / 2.0) ** 2
    base = 1.0 - np.log(np.maximum(cos2, 1.0e-300))
    return A0_FANH * np.power(base, GAMMA0_FANH)


def xi_vec(theta: np.ndarray, hstar: float, vw: float, beta_over_h: float) -> np.ndarray:
    out = np.ones_like(theta, dtype=float)
    for i, th in enumerate(theta):
        out[i] = MODEL.predict(
            hstar=hstar,
            vw=vw,
            theta0=float(th),
            beta_over_h=beta_over_h,
            clip=True,
            xi_dm_mode="broken_powerlaw_ftilde",
        ).xi
    return out


def rho_from_xi(theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    return xi * fanh_no_pt(theta) * (1.0 - np.cos(theta))


def delta_integrand(rho: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, float]:
    rho_bar = float((1.0 / math.pi) * np.trapezoid(rho, theta))
    integrand = ((rho - rho_bar) / rho_bar) ** 2
    return integrand, rho_bar


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTDIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def print_tabulated_xi() -> None:
    print("Tabulated xi(theta0) values")
    print("theta0 nodes:", " ".join(f"{x:.6f}" for x in THETA_TABS))
    for label, hstar, vw, beta, _ in BENCHMARKS:
        vals = xi_vec(THETA_TABS, hstar, vw, beta)
        print(f"{label:12s} (H*={hstar:g}, vw={vw:g}, beta/H*={beta:g})")
        print("  " + " ".join(f"{x:.6f}" for x in vals))


def test1_shapes() -> None:
    rho_no = rho_from_xi(THETA, np.ones_like(THETA))
    rho_no_norm = rho_no / rho_no[0]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)

    for label, hstar, vw, beta, color in BENCHMARKS:
        xi = xi_vec(THETA, hstar, vw, beta)
        rho_pt = rho_from_xi(THETA, xi)
        rho_pt_norm = rho_pt / rho_pt[0]

        axes[0].plot(THETA, xi, lw=1.8, color=color, label=rf"{label}: $(H_*,\beta/H_*)=({hstar:g},{beta:g})$")
        axes[1].plot(THETA, rho_pt / rho_no, lw=1.8, color=color, label=label)
        axes[2].plot(THETA, rho_pt_norm, lw=1.8, color=color, label=label)

    axes[0].set_xlabel(r"$\theta_0$")
    axes[0].set_ylabel(r"$\xi(\theta_0)$")
    axes[0].legend(loc="best", frameon=True)

    axes[1].set_xlabel(r"$\theta_0$")
    axes[1].set_ylabel(r"$\rho_{\rm PT} / \rho_{\rm noPT}$")

    axes[2].plot(THETA, rho_no_norm, lw=1.8, ls="--", color="black", label="noPT")
    axes[2].set_xlabel(r"$\theta_0$")
    axes[2].set_ylabel(r"$\rho(\theta_0) / \rho(\theta_{\rm min})$")
    axes[2].legend(loc="best", frameon=True)

    save(fig, "debug_test1_xi_rho_shapes")


def test2_integrands() -> None:
    rho_no = rho_from_xi(THETA, np.ones_like(THETA))
    integ_no, _ = delta_integrand(rho_no, THETA)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True, sharey=True)

    for ax, (label, hstar, vw, beta, color) in zip(axes, BENCHMARKS):
        xi = xi_vec(THETA, hstar, vw, beta)
        rho_pt = rho_from_xi(THETA, xi)
        integ_pt, _ = delta_integrand(rho_pt, THETA)
        ax.plot(THETA, integ_no, lw=1.8, ls="--", color="black", label="noPT")
        ax.plot(THETA, integ_pt, lw=1.8, color=color, label=label)
        ax.set_title(rf"$H_*/M_\phi={hstar:g},\ \beta/H_*={beta:g}$")
        ax.set_xlabel(r"$\theta_0$")
        ax.legend(loc="best", frameon=True)

    axes[0].set_ylabel(r"$[(\rho-\bar\rho)/\bar\rho]^2$")
    save(fig, "debug_test2_variance_integrand")


def test3_hscan() -> None:
    rho_no = rho_from_xi(THETA, np.ones_like(THETA))
    integ_no, _ = delta_integrand(rho_no, THETA)
    var_no = float((1.0 / math.pi) * np.trapezoid(integ_no, THETA))

    ratios = []
    for h in H_SCAN:
        xi = xi_vec(THETA, h, VW_SCAN, BETA_SCAN)
        rho_pt = rho_from_xi(THETA, xi)
        integ_pt, _ = delta_integrand(rho_pt, THETA)
        var_pt = float((1.0 / math.pi) * np.trapezoid(integ_pt, THETA))
        ratios.append(var_pt / var_no)

    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    ax.plot(H_SCAN, ratios, "-o", lw=1.8, ms=4.0, color="#0077BB")
    ax.axhline(1.0, color="black", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$H_*/M_\phi$")
    ax.set_ylabel(r"$\langle\delta^2\rangle_{\rm PT}/\langle\delta^2\rangle_{\rm noPT}$")
    ax.set_title(rf"$v_w={VW_SCAN:g},\ \beta/H_*={BETA_SCAN:g}$")
    save(fig, "debug_test3_variance_ratio_vs_hstar")


def main() -> None:
    print_tabulated_xi()
    test1_shapes()
    test2_integrands()
    test3_hscan()


if __name__ == "__main__":
    main()
