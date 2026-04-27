#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmbconstraints.calc_preinflation_noneq import DOMAIN_EPS, DEFAULT_AS, DEFAULT_ALPHA_ISO_MAX
from cmbconstraints.calc_preinflation_gaussian import load_or_build_rho_cache


@dataclass
class EquilibriumResult:
    hstar: float
    vw: float
    beta_over_h: float
    b_param: float  # 8*pi^2*V_max / (3*HI^4)
    theta_typ: float # sqrt(<theta^2>)
    rho_bar: float   # <rho>
    p_s_cmb: float   # Tenkanen-style local response: (delta_theta * dlnrho/dtheta)^2
    p_s_patch: float # Global variance: <rho^2>/<rho>^2 - 1
    alpha_iso: float # Isocurvature fraction for CMB
    is_valid: bool


def compute_equilibrium_stats(
    theta_master: np.ndarray,
    rho_master: np.ndarray,
    b_param: float,
    a_s: float = DEFAULT_AS,
) -> EquilibriumResult:
    """
    Computes density statistics under the stochastic equilibrium distribution.
    Matches the Tenkanen-style 'local response' for CMB isocurvature.
    """
    # 1. Build PDF: P(theta) \propto exp(-b * (1 - cos(theta)))
    weights = np.exp(-b_param * (1.0 - np.cos(theta_master)))
    norm = np.trapezoid(weights, theta_master)
    if norm <= 0:
        return EquilibriumResult(0, 0, 0, b_param, 0, 0, 0, 0, 0, False)
    pdf = weights / norm

    # 2. Compute averages
    rho_bar = float(np.trapezoid(pdf * rho_master, theta_master))
    rho2_bar = float(np.trapezoid(pdf * rho_master**2, theta_master))
    theta2_bar = float(np.trapezoid(pdf * theta_master**2, theta_master))
    theta_typ = math.sqrt(theta2_bar)

    # 3. Patch-to-patch variance (LSS / White noise)
    p_s_patch = (rho2_bar / (rho_bar**2)) - 1.0 if rho_bar > 0 else 0.0

    # 4. CMB Isocurvature (Tenkanen style)
    # P_S = (delta_theta * dln_rho/dtheta)^2 evaluated at theta_typ
    # From b = 8*pi^2*V_max / (3*HI^4), and V(theta) approx 1/2 m^2 f^2 theta^2
    # we have delta_theta^2 = (HI/2pi*f)^2 approx 1 / (sqrt(6)*pi*b) * (m/f) ... 
    # Actually, for simplicity and to match the 'noneq' logic, we define
    # the local fluctuation scale delta_theta from the distribution itself:
    # In the harmonic limit, <theta^2> = 1/b, and we expect delta_theta^2 approx <theta^2> / (Volume factor)
    # More simply, we use the response coefficient:
    dlnrho = np.gradient(np.log(np.clip(rho_master, 1e-300, None)), theta_master)
    dlnrho_typ = float(np.interp(theta_typ, theta_master, dlnrho))
    
    # delta_theta_squared is the fluctuation power per log k. 
    # For equilibrium stochastic case, the 'standard' isocurvature result 
    # relates HI/f and b. We use the identity delta_theta^2 = 1/(2*b) 
    # (harmonic approximation for the fluctuation of the background-equivalent theta)
    delta_theta2 = 1.0 / (2.0 * b_param) if b_param > 0 else 0.0
    p_s_cmb = delta_theta2 * (dlnrho_typ**2)

    # alpha_iso = P_S / (P_S + P_zeta)
    alpha_iso = p_s_cmb / (p_s_cmb + a_s) if (p_s_cmb + a_s) > 0 else 0.0

    return EquilibriumResult(
        hstar=0.0, vw=0.0, beta_over_h=0.0,
        b_param=float(b_param),
        theta_typ=theta_typ,
        rho_bar=rho_bar,
        p_s_cmb=p_s_cmb,
        p_s_patch=p_s_patch,
        alpha_iso=alpha_iso,
        is_valid=True
    )


def main():
    parser = argparse.ArgumentParser(description="Calculate equilibrium axion isocurvature.")
    parser.add_argument("--hstar", type=float, default=0.05, help="H_*/M_phi")
    parser.add_argument("--vw", type=float, default=0.7, help="Wall speed v_w")
    parser.add_argument("--betaH", type=float, default=4.0, help="beta/H_*")
    parser.add_argument("--b-min", type=float, default=0.01, help="Min b = 8*pi^2*Vmax / 3*HI^4")
    parser.add_argument("--b-max", type=float, default=100.0, help="Max b")
    parser.add_argument("--n-b", type=int, default=50, help="Number of b points")
    parser.add_argument("--a-s", type=float, default=DEFAULT_AS)
    parser.add_argument("--alpha-iso-max", type=float, default=DEFAULT_ALPHA_ISO_MAX)
    parser.add_argument("--nopt", action="store_true", help="Calculate the standard noPT case.")
    
    args = parser.parse_args()
    
    # 1. Load or compute the rho(theta) profile
    if args.nopt:
        print("Computing standard noPT rho(theta) profile...")
        # We still need a theta grid; we'll use the one from a typical cache or build one
        from cmbconstraints.calc_preinflation_gaussian import build_master_theta_grid, fanh_no_pt
        theta_master = build_master_theta_grid()
        rho_master = np.array([fanh_no_pt(float(th)) * (1.0 - math.cos(float(th))) for th in theta_master], dtype=float)
        cache_path = "Internal (NoPT)"
    else:
        print(f"Loading rho(theta) profile for H*={args.hstar}, vw={args.vw}, beta/H*={args.betaH}...")
        theta_master, rho_master, cache_path = load_or_build_rho_cache(
            hstar=args.hstar, vw=args.vw, beta_over_h=args.betaH
        )
    print(f"Using source: {cache_path}")
    
    # 2. Scan over the stochasticity parameter b
    b_grid = np.logspace(np.log10(args.b_min), np.log10(args.b_max), args.n_b)
    
    results = []
    print(f"{'b':>10} | {'P_S (CMB)':>12} | {'P_S (Patch)':>12} | {'Status':>10}")
    print("-" * 65)
    
    for b in b_grid:
        res = compute_equilibrium_stats(theta_master, rho_master, b, a_s=args.a_s)
        if args.nopt:
            res.hstar = 0.0
            res.vw = 0.0
            res.beta_over_h = 0.0
        else:
            res.hstar = args.hstar
            res.vw = args.vw
            res.beta_over_h = args.betaH
        
        status = "BOUNDED" if res.alpha_iso < args.alpha_iso_max else "EXCEEDED"
        print(f"{b:10.4f} | {res.p_s_cmb:12.4e} | {res.p_s_patch:12.4e} | {status}")
        results.append(asdict(res))
        
    # 3. Save results
    out_dir = Path("outputs/equilibrium")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.nopt:
        stem = "equilibrium_nopt"
    else:
        stem = f"equilibrium_h{args.hstar}_vw{args.vw}_b{args.betaH}".replace(".", "p")
    out_path = out_dir / f"{stem}.json"
    
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
