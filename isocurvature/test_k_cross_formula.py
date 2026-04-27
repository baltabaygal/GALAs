#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isocurvature import plot_isocurvature as iso


OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTDIR / "k_cross_formula_test.csv"
JSON_PATH = OUTDIR / "k_cross_formula_test_summary.json"
K_DAT_MPC, P_DAT_MPC = iso.load_pcdm_dat_physical()
THETA_TEST = iso.THETA_GRID[np.linspace(0, len(iso.THETA_GRID) - 1, 33, dtype=int)]


@dataclass
class Row:
    mass_ev: float
    hstar: float
    vw: float
    beta_over_h: float
    var: float
    t_onset: float
    kcut_mpc: float
    kcross_num_hmpc: float | None
    kcross_fit_hmpc: float | None
    rel_err: float | None


def numeric_crossing_hmpc(kcut_mpc: float, var: float) -> float | None:
    p_i = iso.WHITE_NOISE_PREFAC * (kcut_mpc ** -3) * var / iso.DI2_EQ
    vals = P_DAT_MPC - p_i
    idx = [i for i in range(len(vals) - 1) if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0.0]
    if not idx:
        return None
    i = int(idx[0])
    x0, x1 = float(K_DAT_MPC[i]), float(K_DAT_MPC[i + 1])
    y0, y1 = float(vals[i]), float(vals[i + 1])
    kcross_mpc = x0 - y0 * (x1 - x0) / (y1 - y0)
    return float(iso.k_mpc_to_hmpc(kcross_mpc))


def fit_formula_hmpc(*, mphi_ev: float, var: float, t_onset: float, var_ref: float, kcross_ref_h: float, g_ref: float, gs_ref: float) -> float:
    t_osc = iso.TOSC
    t_osc_ev = iso.temperature_from_hubble_rd(mphi_ev / 3.0)
    g = iso.gstar_energy(t_osc_ev)
    gs = iso.gstar_entropy(t_osc_ev)
    return float(
        kcross_ref_h
        * (mphi_ev / 1.0e-18) ** 0.5633423806592047
        * (var / var_ref) ** (-0.37556158710613646)
        * (t_osc / t_onset) ** 0.5633423806592047
        * (g / g_ref) ** 0.28167119032960236
        * (gs / gs_ref) ** (-0.37556158710613646)
    )


def fast_pt_summary(hstar: float, vw: float, beta: float) -> dict[str, float]:
    xi = np.empty(len(THETA_TEST), dtype=float)
    for i, th in enumerate(THETA_TEST):
        vals, _, _ = iso.MODEL._prepare_inputs(theta0=float(th), vw=vw, hstar=hstar, beta_over_h=beta, clip=True)
        xi[i] = float(iso.MODEL._eval_core(**vals, xi_dm_mode="broken_powerlaw_ftilde")["xi"])
    rho = iso.rho_from_xi(THETA_TEST, xi)
    _, var, _ = iso.mean_and_var(THETA_TEST, rho)
    tp, t_onset, kcut_ratio = iso.t_onset_and_kcut_ratio(hstar, beta, vw)
    return {
        "var": float(var),
        "tp": float(tp),
        "t_onset": float(t_onset),
        "kcut_ratio": float(kcut_ratio),
    }


def reference_normalization() -> tuple[float, float, float, float]:
    var_ref = 2.0077
    mref = 1.0e-18
    kcut_ref = iso.kcut_no_pt_mpc_inv(mref)
    kcross_ref_h = numeric_crossing_hmpc(kcut_ref, var_ref)
    if kcross_ref_h is None:
        raise RuntimeError("Reference k_cross not found inside the tabulated CDM range")
    t_osc_ev = iso.temperature_from_hubble_rd(mref / 3.0)
    g_ref = iso.gstar_energy(t_osc_ev)
    gs_ref = iso.gstar_entropy(t_osc_ev)
    return var_ref, float(kcross_ref_h), float(g_ref), float(gs_ref)


def main() -> None:
    masses = [1.0e-16]
    var_ref, kcross_ref_h, g_ref, gs_ref = reference_normalization()
    rows: list[Row] = []
    pt_cache: dict[tuple[float, float, float], dict[str, float]] = {}

    for mass_ev in masses:
        kcut0_mpc = iso.kcut_no_pt_mpc_inv(mass_ev)
        for hstar in map(float, iso.MODEL.hstar_grid):
            for vw in map(float, iso.MODEL.vw_grid):
                for beta in map(float, iso.MODEL.beta_grid):
                    key = (hstar, vw, beta)
                    if key not in pt_cache:
                        pt_cache[key] = fast_pt_summary(hstar, vw, beta)
                    pt = pt_cache[key]
                    var = float(pt["var"])
                    t_onset = float(pt["t_onset"])
                    kcut_mpc = float(pt["kcut_ratio"]) * kcut0_mpc
                    kcross_num = numeric_crossing_hmpc(kcut_mpc, var)
                    kcross_fit = fit_formula_hmpc(
                        mphi_ev=mass_ev,
                        var=var,
                        t_onset=t_onset,
                        var_ref=var_ref,
                        kcross_ref_h=kcross_ref_h,
                        g_ref=g_ref,
                        gs_ref=gs_ref,
                    )
                    rel_err = None if kcross_num is None else float((kcross_fit - kcross_num) / kcross_num)
                    rows.append(
                        Row(
                            mass_ev=float(mass_ev),
                            hstar=hstar,
                            vw=vw,
                            beta_over_h=beta,
                            var=var,
                            t_onset=t_onset,
                            kcut_mpc=float(kcut_mpc),
                            kcross_num_hmpc=None if kcross_num is None else float(kcross_num),
                            kcross_fit_hmpc=float(kcross_fit),
                            rel_err=rel_err,
                        )
                    )

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    valid = [r for r in rows if r.kcross_num_hmpc is not None and r.rel_err is not None]
    abs_errs = np.array([abs(r.rel_err) for r in valid], dtype=float)
    by_mass: dict[str, dict[str, float]] = {}
    for mass_ev in masses:
        subset = [r for r in valid if abs(r.mass_ev - mass_ev) < 1.0e-30]
        vals = np.array([abs(r.rel_err) for r in subset], dtype=float)
        by_mass[f"{mass_ev:.0e}"] = {
            "n": int(len(subset)),
            "mean_abs_frac_err": float(vals.mean()),
            "median_abs_frac_err": float(np.median(vals)),
            "max_abs_frac_err": float(vals.max()),
            "p90_abs_frac_err": float(np.quantile(vals, 0.9)),
        }

    worst = sorted(valid, key=lambda r: abs(r.rel_err), reverse=True)[:10]
    summary = {
        "formula": "k_cross ~= kref * (mphi/1e-18)^0.56334 * (var/2.0077)^-0.37556 * (tosc/tonset)^0.56334 * (g/gref)^0.28167 * (gs/gsref)^-0.37556",
        "reference": {
            "kcross_ref_hmpc": kcross_ref_h,
            "var_ref": var_ref,
            "g_ref": g_ref,
            "gs_ref": gs_ref,
        },
        "n_total": int(len(rows)),
        "n_valid": int(len(valid)),
        "mean_abs_frac_err": float(abs_errs.mean()),
        "median_abs_frac_err": float(np.median(abs_errs)),
        "max_abs_frac_err": float(abs_errs.max()),
        "p90_abs_frac_err": float(np.quantile(abs_errs, 0.9)),
        "by_mass": by_mass,
        "worst_cases": [asdict(r) for r in worst],
        "csv_path": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
