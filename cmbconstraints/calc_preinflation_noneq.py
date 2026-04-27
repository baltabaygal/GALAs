#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XI_MODEL_DIR = ROOT / "xi_model"
if str(XI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(XI_MODEL_DIR))

from xi_model import load_default_model


A0_FANH = 0.373
GAMMA0_FANH = 1.20
DOMAIN_EPS = 1.0e-8
DEFAULT_DTHETA = 1.0e-3
DEFAULT_AS = 2.1e-9
DEFAULT_ALPHA_ISO_MAX = 0.038


def build_hybrid_theta_grid(
    theta_min: float = DOMAIN_EPS,
    theta_max: float = math.pi - DOMAIN_EPS,
    n_total: int = 100,
    split_at: float = 2.0,
) -> np.ndarray:
    """
    Builds a grid that is linear up to split_at, and then 
    geometric (dense) approaching theta_max (near pi).
    """
    theta_lo = max(theta_min, DOMAIN_EPS)
    theta_hi = min(theta_max, math.pi - DOMAIN_EPS)
    
    if theta_hi <= split_at:
        return np.linspace(theta_lo, theta_hi, n_total)
        
    n_low = int(n_total * 0.4)
    n_high = n_total - n_low
    
    low = np.linspace(theta_lo, split_at, n_low, endpoint=False)
    
    # Geometric spacing for the distance from pi
    # This packs points exponentially close to the hilltop
    dist_start = math.pi - split_at
    dist_end = math.pi - theta_hi
    high_dists = np.geomspace(dist_start, dist_end, n_high)
    high = math.pi - high_dists
    
    return np.unique(np.concatenate([low, high]))


@dataclass
class CMBIsoResult:
    theta0: float
    hstar: float
    vw: float
    beta_over_h: float
    HI_over_fphi: float
    delta_theta: float
    xi: float
    dln_potential: float
    dln_fanh: float
    dln_xi: float
    dln_rho_total: float
    p_s: float
    p_s_standard: float
    enhancement_vs_standard: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


MODEL = load_default_model()


def h_theta(theta0: float) -> float:
    return float(1.0 - math.log(max(math.cos(theta0 / 2.0) ** 2, 1.0e-300)))


def fanh_no_pt(theta0: float) -> float:
    return float(A0_FANH * (h_theta(theta0) ** GAMMA0_FANH))


def dln_potential(theta0: float) -> float:
    return float(1.0 / max(math.tan(theta0 / 2.0), 1.0e-300))


def dln_fanh(theta0: float) -> float:
    h = h_theta(theta0)
    return float(GAMMA0_FANH * math.tan(theta0 / 2.0) / h)


def dln_no_pt_total(theta0: float) -> float:
    return float(dln_potential(theta0) + dln_fanh(theta0))


def dln_harmonic_reference(theta0: float) -> float:
    return float(2.0 / max(theta0, 1.0e-300))


def xi_value(theta0: float, hstar: float, vw: float, beta_over_h: float) -> float:
    res = MODEL.predict(
        hstar=hstar,
        vw=vw,
        theta0=theta0,
        beta_over_h=beta_over_h,
        clip=True,
        xi_dm_mode="broken_powerlaw_ftilde",
    )
    return float(res.xi)


def dln_xi_fd(theta0: float, hstar: float, vw: float, beta_over_h: float, dtheta: float = DEFAULT_DTHETA) -> float:
    left = max(0.0 + DOMAIN_EPS, theta0 - dtheta)
    right = min(math.pi - DOMAIN_EPS, theta0 + dtheta)
    if right <= left:
        raise ValueError(f"Finite-difference window collapsed at theta0={theta0:g}")
    xi_l = max(xi_value(left, hstar, vw, beta_over_h), 1.0e-300)
    xi_r = max(xi_value(right, hstar, vw, beta_over_h), 1.0e-300)
    return float((math.log(xi_r) - math.log(xi_l)) / (right - left))


def compute_preinflation_noneq(theta0: float, hstar: float, vw: float, beta_over_h: float, hi_over_fphi: float) -> CMBIsoResult:
    delta_theta = float(hi_over_fphi / (2.0 * math.pi))
    xi = xi_value(theta0, hstar, vw, beta_over_h)
    term_pot = dln_potential(theta0)
    term_fanh = dln_fanh(theta0)
    term_xi = dln_xi_fd(theta0, hstar, vw, beta_over_h)
    dln_total = term_pot + term_fanh + term_xi
    dln_standard = term_pot + term_fanh
    amp_pt = float(delta_theta * dln_total)
    amp_std = float(delta_theta * dln_standard)
    max_amp = math.sqrt(sys.float_info.max)
    p_s = float("inf") if abs(amp_pt) >= max_amp else float(amp_pt**2)
    p_s_standard = float("inf") if abs(amp_std) >= max_amp else float(amp_std**2)
    enhancement = float(p_s / max(p_s_standard, 1.0e-300))
    return CMBIsoResult(
        theta0=theta0,
        hstar=hstar,
        vw=vw,
        beta_over_h=beta_over_h,
        HI_over_fphi=hi_over_fphi,
        delta_theta=delta_theta,
        xi=xi,
        dln_potential=term_pot,
        dln_fanh=term_fanh,
        dln_xi=term_xi,
        dln_rho_total=dln_total,
        p_s=p_s,
        p_s_standard=p_s_standard,
        enhancement_vs_standard=enhancement,
    )


def isocurvature_power_limit(a_s: float = DEFAULT_AS, alpha_iso_max: float = DEFAULT_ALPHA_ISO_MAX) -> float:
    if not (0.0 < alpha_iso_max < 1.0):
        raise ValueError(f"alpha_iso_max={alpha_iso_max:g} must lie in (0, 1)")
    if a_s <= 0.0:
        raise ValueError(f"a_s={a_s:g} must be positive")
    return float((alpha_iso_max / (1.0 - alpha_iso_max)) * a_s)


def hi_over_fphi_bound_from_response(dln_rho_total: float, a_s: float = DEFAULT_AS, alpha_iso_max: float = DEFAULT_ALPHA_ISO_MAX) -> float:
    pref = abs(float(dln_rho_total))
    if pref <= 1.0e-300:
        return float("inf")
    p_s_max = isocurvature_power_limit(a_s=a_s, alpha_iso_max=alpha_iso_max)
    return float(2.0 * math.pi * math.sqrt(p_s_max) / pref)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-inflation, non-equilibrium CMB isocurvature calculator.")
    p.add_argument("--theta0", type=float, required=True, help="Background misalignment angle theta0 in radians.")
    p.add_argument("--hstar", type=float, required=True, help="H_*/M_phi")
    p.add_argument("--vw", type=float, required=True, help="Wall speed v_w")
    p.add_argument("--betaH", type=float, required=True, help="beta/H_*")
    p.add_argument("--HI-over-fphi", type=float, required=True, help="Dimensionless ratio H_I / f_phi")
    p.add_argument("--json", action="store_true", help="Print JSON instead of a compact text summary.")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    res = compute_preinflation_noneq(
        theta0=float(ns.theta0),
        hstar=float(ns.hstar),
        vw=float(ns.vw),
        beta_over_h=float(ns.betaH),
        hi_over_fphi=float(ns.HI_over_fphi),
    )
    if ns.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"theta0               = {res.theta0:.6g}")
        print(f"hstar                = {res.hstar:.6g}")
        print(f"vw                   = {res.vw:.6g}")
        print(f"beta/H*              = {res.beta_over_h:.6g}")
        print(f"H_I/f_phi            = {res.HI_over_fphi:.6g}")
        print(f"delta_theta          = {res.delta_theta:.6g}")
        print(f"xi(theta0)           = {res.xi:.6g}")
        print(f"dln(1-cos theta)     = {res.dln_potential:.6g}")
        print(f"dln f_anh            = {res.dln_fanh:.6g}")
        print(f"dln xi               = {res.dln_xi:.6g}")
        print(f"dln rho total        = {res.dln_rho_total:.6g}")
        print(f"P_S standard         = {res.p_s_standard:.6g}")
        print(f"P_S PT               = {res.p_s:.6g}")
        print(f"PT / standard        = {res.enhancement_vs_standard:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
