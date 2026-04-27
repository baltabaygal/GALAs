#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np

from paper_plots.style import apply_paper_style, get_style, viridis_colors, decimal_log_tick_formatter

STYLE_1COL = apply_paper_style("1col")
XI_MODEL_DIR = ROOT / "paper_codes" / "xi_model"
PERC_DIR = ROOT / "ode" / "hom_ODE"
PK_DAT_PATH = ROOT / "paper_codes" / "common" / "data" / "Pk_CDM 2.dat"
LYA_DAT_PATH = ROOT / "paper_codes" / "common" / "data" / "DR14_Lyman_alpha_Pk.dat"
LRG_DAT_PATH = ROOT / "paper_codes" / "common" / "data" / "LRG_DR7_Pk.dat"
GSTAR_LOOKUP_PATH = ROOT / "paper_codes" / "common" / "gstar_lookup.json"
if str(XI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(XI_MODEL_DIR))
if str(PERC_DIR) not in sys.path:
    sys.path.insert(0, str(PERC_DIR))

from xi_model import load_default_model
from percolation import t_perc_RD


MODEL = load_default_model()
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
PVSK_PER_MPHI_DIR = Path(__file__).resolve().parent / "PvsKperMphi"
PVSK_PER_MPHI_DIR.mkdir(parents=True, exist_ok=True)
PT_CACHE_PATH = Path(__file__).resolve().parent / "pt_case_cache.json"

A0_FANH = 0.373
GAMMA0_FANH = 1.20
TOSC = 1.5
VW_FIXED = 0.5
M_PHI_DEFAULT_EV = 1.0e-5

THETA_MIN = 0.0
THETA_MAX = math.pi
N_THETA = 2000
THETA_GRID = THETA_MAX - np.logspace(-6, 0, N_THETA)[::-1] * (THETA_MAX - 0.262)
THETA_GRID = np.unique(np.concatenate(([0.0, 0.05, 0.1, 0.2, 0.262], np.clip(THETA_GRID, 0.262, THETA_MAX), [math.pi])))

G_S_0 = 3.91
T0_EV = 2.35e-4
MPL_REDUCED_EV = 2.435e27
HBARC_EV_M = 1.973269804e-7
M_PER_MPC = 3.085677581491367e22
EV_TO_MPC_INV = M_PER_MPC / HBARC_EV_M
MPC_INV_TO_KPC_INV = 1.0e-3
MPC3_TO_KPC3 = 1.0e9
WHITE_NOISE_PREFAC = 6.0 * math.pi**2
VAR_HARMONIC_REF = 4.0 / 5.0
KCUT_HARMONIC_REF_MPC = 300.0

A_S = 2.1e-9
N_S = 0.965
K_PIVOT = 0.05
K_EQ = 0.01
Z_EQ = 3402.0
DI_EQ = 3.7e-4
DI2_EQ = DI_EQ**2
DI2_BOOST = 1.0 / DI2_EQ
H0_REDUCED = 0.674
OMEGA_M = 0.315
OMEGA_B = 0.049
GAMMA_EH = OMEGA_M * H0_REDUCED * math.exp(-OMEGA_B * (1.0 + math.sqrt(2.0 * H0_REDUCED) / OMEGA_M))
K_CDM_DATA_LIMIT = 10.0
GSTAR_LOOKUP = json.loads(GSTAR_LOOKUP_PATH.read_text())  # source: https://arxiv.org/pdf/1609.04979
_T_DOF_EV = np.array(GSTAR_LOOKUP["temperature_ev"], dtype=float)
_GSTAR_ENERGY = np.array(GSTAR_LOOKUP["g_energy"], dtype=float)
_GSTAR_ENTROPY = np.array(GSTAR_LOOKUP["g_entropy"], dtype=float)

COLORS = ["#000000", "#0077BB", "#EE7733", "#009988"]

mpl.rcParams.update(
    {
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
    }
)


@dataclass(frozen=True)
class Case:
    label: str
    hstar: float | None
    beta: float | None
    color: str
    ls: str


CASES = [
    Case("noPT", None, None, COLORS[0], "--"),
    Case("DM-dom  H*=0.2 β=8", 0.2, 8.0, COLORS[1], "-"),
    Case("BM-dom  H*=0.05 β=4", 0.05, 4.0, COLORS[2], "-"),
    Case("Moderate H*=1.0 β=20", 1.0, 20.0, COLORS[3], "-"),
]


def fanh_no_pt(theta: np.ndarray) -> np.ndarray:
    cos2 = np.cos(theta / 2.0) ** 2
    base = 1.0 - np.log(np.maximum(cos2, 1.0e-300))
    return A0_FANH * np.power(base, GAMMA0_FANH)


def xi_vec(theta0_arr: np.ndarray, hstar: float, vw: float, beta_over_h: float) -> np.ndarray:
    out = np.ones(len(theta0_arr), dtype=float)
    for i, th in enumerate(theta0_arr):
        r = MODEL.predict(
            hstar=hstar,
            vw=vw,
            theta0=float(th),
            beta_over_h=beta_over_h,
            clip=True,
            xi_dm_mode="broken_powerlaw_ftilde",
        )
        out[i] = r.xi
    return out


def rho_from_xi(theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    return xi * fanh_no_pt(theta) * (1.0 - np.cos(theta))


def mean_and_var(theta: np.ndarray, rho: np.ndarray) -> tuple[float, float, np.ndarray]:
    rho_bar = float((1.0 / math.pi) * np.trapezoid(rho, theta))
    integrand = ((rho - rho_bar) / rho_bar) ** 2
    var = float((1.0 / math.pi) * np.trapezoid(integrand, theta))
    return rho_bar, var, integrand


def t_onset_and_kcut_ratio(hstar: float, beta: float, vw: float) -> tuple[float, float, float]:
    tp = float(t_perc_RD(hstar, beta, vw))
    t_onset = max(tp, TOSC)
    kcut_ratio = math.sqrt(TOSC / t_onset)
    return tp, t_onset, kcut_ratio


def _interp_log_temperature_dof(temp_ev: float, values: np.ndarray) -> float:
    temp_ev = float(max(temp_ev, float(_T_DOF_EV[0])))
    return float(np.interp(np.log(temp_ev), np.log(_T_DOF_EV), np.asarray(values, dtype=float)))


def gstar_energy(temp_ev: float) -> float:
    return _interp_log_temperature_dof(temp_ev, _GSTAR_ENERGY)


def gstar_entropy(temp_ev: float) -> float:
    return _interp_log_temperature_dof(temp_ev, _GSTAR_ENTROPY)


def temperature_from_hubble_rd(hubble_ev: float) -> float:
    temp_ev = float(math.sqrt(hubble_ev * MPL_REDUCED_EV) * (90.0 / (math.pi**2 * 100.0)) ** 0.25)
    for _ in range(32):
        g_eff = gstar_energy(temp_ev)
        updated = float(math.sqrt(hubble_ev * MPL_REDUCED_EV) * (90.0 / (math.pi**2 * g_eff)) ** 0.25)
        if abs(updated - temp_ev) <= 1.0e-10 * max(1.0, temp_ev):
            temp_ev = updated
            break
        temp_ev = updated
    return temp_ev


def kcut_no_pt_mpc_inv(mphi_ev: float) -> float:
    h_osc_ev = mphi_ev / 3.0
    t_osc_ev = temperature_from_hubble_rd(h_osc_ev)
    g_s_osc = gstar_entropy(t_osc_ev)
    kcut_ev = h_osc_ev * (T0_EV / t_osc_ev) * (G_S_0 / g_s_osc) ** (1.0 / 3.0)
    return kcut_ev * EV_TO_MPC_INV


def delta2_cdm(k: np.ndarray) -> np.ndarray:
    return A_S * np.power(k / K_PIVOT, N_S - 1.0)


def load_pcdm_dat() -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(PK_DAT_PATH)
    k_mpc = 1000.0 * np.asarray(arr[:, 0], dtype=float)
    p_cdm = np.asarray(arr[:, 1], dtype=float)
    return k_mpc, p_cdm


def load_pcdm_dat_physical() -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(PK_DAT_PATH)
    k_mpc = 1000.0 * np.asarray(arr[:, 0], dtype=float)
    p_mpc3 = np.asarray(arr[:, 1], dtype=float) / (1000.0 ** 3)
    return k_mpc, p_mpc3


def load_external_pk_dat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float), np.asarray(arr[:, 2], dtype=float)


def _load_pt_cache() -> dict[str, dict[str, float]]:
    if not PT_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(PT_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_pt_cache(cache: dict[str, dict[str, float]]) -> None:
    PT_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _pt_cache_key(hstar: float, vw: float, beta: float) -> str:
    return f"h={hstar:.8g}|vw={vw:.8g}|beta={beta:.8g}|n={len(THETA_GRID)}|mode=broken_powerlaw_ftilde"


def get_pt_case_summary(hstar: float, vw: float, beta: float) -> dict[str, float]:
    cache = _load_pt_cache()
    key = _pt_cache_key(hstar, vw, beta)
    if key in cache:
        return cache[key]
    xi = xi_vec(THETA_GRID, hstar, vw, beta)
    rho = rho_from_xi(THETA_GRID, xi)
    _, var, _ = mean_and_var(THETA_GRID, rho)
    tp, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
    row = {
        "var": float(var),
        "tp": float(tp),
        "t_onset": float(t_onset),
        "kcut_ratio": float(kcut_ratio),
    }
    cache[key] = row
    _save_pt_cache(cache)
    return row


def get_pt_case_summary_beta_infty(hstar: float, vw: float) -> dict[str, float]:
    beta_samples = np.array([12.0, 20.0, 40.0], dtype=float)
    inv_beta = 1.0 / beta_samples
    rows = [get_pt_case_summary(hstar, vw, float(beta)) for beta in beta_samples]
    var_vals = np.array([float(row["var"]) for row in rows], dtype=float)
    coeff = np.polyfit(inv_beta, var_vals, 1)
    var_inf = max(float(coeff[1]), 1.0e-8)
    tp_inf = 1.0 / (2.0 * float(hstar))
    t_onset_inf = max(tp_inf, TOSC)
    kcut_ratio_inf = math.sqrt(TOSC / t_onset_inf)
    return {
        "var": var_inf,
        "tp": tp_inf,
        "t_onset": t_onset_inf,
        "kcut_ratio": kcut_ratio_inf,
    }


def p_lin_eh(k: np.ndarray | float) -> np.ndarray | float:
    kk = np.asarray(k, dtype=float)
    q = kk / (GAMMA_EH * H0_REDUCED)
    q = np.maximum(q, 1.0e-30)
    term1 = np.log(1.0 + 2.34 * q) / (2.34 * q)
    term2 = np.power(1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4, -0.25)
    transfer = term1 * term2
    d2 = A_S * np.power(kk / K_PIVOT, N_S - 1.0) * (transfer**2)
    p = (2.0 * math.pi**2 / (kk**3)) * d2
    if np.isscalar(k):
        return float(p)
    return p


def build_pcdm_segments() -> tuple[np.ndarray, np.ndarray]:
    k_dat, p_dat = load_pcdm_dat_physical()
    use = k_dat <= K_CDM_DATA_LIMIT
    return k_dat[use], p_dat[use]


def p_cdm_stitched(k: np.ndarray | float, k_dat: np.ndarray, p_dat: np.ndarray) -> np.ndarray | float:
    kk = np.asarray(k, dtype=float)
    out = np.empty_like(kk, dtype=float)
    in_dat = kk <= K_CDM_DATA_LIMIT
    if np.any(in_dat):
        out[in_dat] = np.exp(np.interp(np.log(kk[in_dat]), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300))))
    if np.any(~in_dat):
        out[~in_dat] = np.asarray(p_lin_eh(kk[~in_dat]), dtype=float)
    if np.isscalar(k):
        return float(out)
    return out


def p_white_noise(var: float, kcut: float, k: np.ndarray) -> np.ndarray:
    p0 = WHITE_NOISE_PREFAC * (kcut ** -3) * var
    return p0 * (k <= kcut)


def p_white_noise_growth(var: float, kcut: float, k: np.ndarray, di2: float = DI2_EQ) -> np.ndarray:
    p0 = WHITE_NOISE_PREFAC * (kcut ** -3) * var / di2
    return p0 * (k <= kcut)


def delta2_from_p(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    return (k**3 / (2.0 * math.pi**2)) * p


def run_unit_sanity_checks() -> None:
    for k_mpc in [1.0e-4, 1.0e-2, 1.0, 1.0e2]:
        p_mpc = float(p_lin_eh(k_mpc))
        d2_mpc = float(delta2_from_p(np.array([p_mpc]), np.array([k_mpc]))[0])
        k_kpc = k_mpc * MPC_INV_TO_KPC_INV
        p_kpc = p_mpc * MPC3_TO_KPC3
        d2_kpc = float(delta2_from_p(np.array([p_kpc]), np.array([k_kpc]))[0])
        rel = abs(d2_mpc - d2_kpc) / max(abs(d2_mpc), 1.0e-300)
        if rel > 1.0e-12:
            raise RuntimeError(f"Unit mismatch in Delta^2 conversion at k={k_mpc:g} Mpc^-1: rel={rel:.3e}")

    k_dat_mpc, p_dat_mpc = build_pcdm_segments()
    raw = np.loadtxt(PK_DAT_PATH)
    k_raw_kpc = np.asarray(raw[:, 0], dtype=float)
    p_raw_kpc = np.asarray(raw[:, 1], dtype=float)
    use = (k_raw_kpc * 1000.0) <= K_CDM_DATA_LIMIT
    if not np.allclose(k_dat_mpc, k_raw_kpc[use] * 1000.0, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("k conversion mismatch between Mpc^-1 and kpc^-1 tabulated CDM grids")
    if not np.allclose(p_dat_mpc, p_raw_kpc[use] / (1000.0**3), rtol=1.0e-12, atol=0.0):
        raise RuntimeError("P(k) conversion mismatch between Mpc^3 and kpc^3 tabulated CDM grids")


def load_ftilde_params() -> dict[str, float] | None:
    path = ROOT / "paper_codes" / "xi_model" / "data" / "effective_ftilde_by_vw_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    try:
        return {
            "A_inf": float(data["f_inf"]["A_inf"]),
            "gamma_inf": float(data["f_inf"]["gamma_inf"]),
            "A_f": float(data["fixed_f0"]["A_f"]),
            "gamma_f": float(data["fixed_f0"]["gamma_f"]),
            "C0": float(data["C_law"]["C0"]),
            "p_C": float(data["C_law"]["p"]),
            "r0": float(data["r_law"]["r0"]),
            "r1": float(data["r_law"]["r1"]),
        }
    except KeyError:
        return None


def ftilde_ratio(theta: np.ndarray, tp: float, vw: float, pars: dict[str, float]) -> np.ndarray:
    x = 1.0 - np.log(np.maximum(np.cos(theta / 2.0) ** 2, 1.0e-300))
    f0 = pars["A_f"] * np.power(x, pars["gamma_f"])
    finf = pars["A_inf"] * np.power(x, pars["gamma_inf"])
    c_vw = pars["C0"] * (vw ** pars["p_C"])
    r_vw = pars["r0"] + pars["r1"] * vw
    ftilde = np.power((c_vw * finf) ** r_vw + (f0**r_vw) / (((2.0 * tp) / 3.0) ** (1.5 * r_vw)), 1.0 / r_vw)
    return ftilde / f0


def save_pdf(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTDIR / name, bbox_inches="tight")
    plt.close(fig)


def save_pdf_to(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / name, bbox_inches="tight")
    plt.close(fig)


def _set_dynamic_log_yrange(ax: plt.Axes, arrays: list[np.ndarray], floor: float = 1.0e-30) -> None:
    positive = []
    for arr in arrays:
        aa = np.asarray(arr, dtype=float)
        aa = aa[np.isfinite(aa) & (aa > floor)]
        if aa.size:
            positive.append(aa)
    if not positive:
        return
    merged = np.concatenate(positive)
    ymin = 10.0 ** math.floor(math.log10(float(merged.min())) - 0.35)
    ymax = 10.0 ** math.ceil(math.log10(float(merged.max())) + 0.35)
    ax.set_ylim(ymin, ymax)


def k_mpc_to_hmpc(k_mpc: np.ndarray | float) -> np.ndarray | float:
    kk = np.asarray(k_mpc, dtype=float) / H0_REDUCED
    if np.isscalar(k_mpc):
        return float(kk)
    return kk


def p_mpc3_to_hinv3_mpc3(p_mpc3: np.ndarray | float) -> np.ndarray | float:
    pp = np.asarray(p_mpc3, dtype=float) * (H0_REDUCED ** 3)
    if np.isscalar(p_mpc3):
        return float(pp)
    return pp


def build_case_data() -> tuple[dict[str, dict[str, object]], float]:
    case_data: dict[str, dict[str, object]] = {}
    rho_no = rho_from_xi(THETA_GRID, np.ones_like(THETA_GRID))
    rho_bar_no, var_no, integrand_no = mean_and_var(THETA_GRID, rho_no)
    case_data["noPT"] = {
        "theta": THETA_GRID,
        "xi": np.ones_like(THETA_GRID),
        "rho": rho_no,
        "rho_bar": rho_bar_no,
        "rho_norm": rho_no / rho_bar_no,
        "integrand": integrand_no,
        "tp": TOSC,
        "t_onset": TOSC,
        "kcut_ratio": 1.0,
        "var": var_no,
        "var_ratio": 1.0,
        "P0_ratio": 1.0,
    }
    for case in CASES[1:]:
        xi = xi_vec(THETA_GRID, case.hstar, VW_FIXED, case.beta)
        rho = rho_from_xi(THETA_GRID, xi)
        rho_bar, var, integrand = mean_and_var(THETA_GRID, rho)
        tp, t_onset, kcut_ratio = t_onset_and_kcut_ratio(case.hstar, case.beta, VW_FIXED)
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        case_data[case.label] = {
            "theta": THETA_GRID,
            "xi": xi,
            "rho": rho,
            "rho_bar": rho_bar,
            "rho_norm": rho / rho_bar,
            "integrand": integrand,
            "tp": tp,
            "t_onset": t_onset,
            "kcut_ratio": kcut_ratio,
            "var": var,
            "var_ratio": var / var_no,
            "P0_ratio": p0_ratio,
        }
    return case_data, var_no


def print_summary(case_data: dict[str, dict[str, object]], var_no: float) -> None:
    print("Case              | t_p    | t_onset | k_cut_ratio | var_PT  | var_noPT | var_ratio | P0_ratio")
    print(
        f"noPT              | {TOSC:5.3f}  | {TOSC:5.3f}   | {1.0:5.3f}       | "
        f"{var_no:7.4f} | {var_no:7.4f}  | {1.0:7.3f}   | {1.0:7.3f}"
    )
    for case in CASES[1:]:
        row = case_data[case.label]
        print(
            f"H*={case.hstar:<4g} beta={case.beta:<4g} | "
            f"{float(row['tp']):5.3f} | {float(row['t_onset']):6.3f} | {float(row['kcut_ratio']):7.3f}     | "
            f"{float(row['var']):7.4f} | {var_no:7.4f}  | {float(row['var_ratio']):7.3f}   | {float(row['P0_ratio']):7.3f}"
        )
        if float(row["P0_ratio"]) < 1.0:
            print(f"WARNING: {case.label} has P0_ratio < 1 ({float(row['P0_ratio']):.3f})")


def make_proof_figure(case_data: dict[str, dict[str, object]]) -> None:
    pars = load_ftilde_params()
    ncols = 4 if pars is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(18, 4) if ncols == 4 else (14, 4), constrained_layout=True)

    ax = axes[0]
    for case in CASES[1:]:
        row = case_data[case.label]
        ax.plot(THETA_GRID, row["xi"], color=case.color, ls="-", label=case.label)
        xi_bar = float(row["rho_bar"]) / float(case_data["noPT"]["rho_bar"])
        ax.axhline(xi_bar, color=case.color, lw=1.0, ls="--", alpha=0.8)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.annotate(
        "xi decreases with theta0\nanti-correlated with rho_noPT",
        xy=(2.2, 6.0),
        xytext=(1.55, 14.0),
        arrowprops=dict(arrowstyle="->", lw=0.9),
        fontsize=9,
    )
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\xi(\theta_0)$")
    ax.set_title(r"$\xi(\theta_0)$ shape")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(THETA_GRID, case_data["noPT"]["rho_norm"], color="black", ls="--", label="noPT")
    for case in CASES[1:]:
        row = case_data[case.label]
        ax.plot(THETA_GRID, row["rho_norm"], color=case.color, ls="-", label=case.label)
    ax.axhline(1.0, color="#888888", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\rho(\theta_0) / \bar\rho$")
    ax.set_title(r"$\rho(\theta_0)/\bar\rho$ -- flatter means lower variance")

    dm = CASES[1]
    row_dm = case_data[dm.label]
    ax = axes[2]
    rho_no = np.asarray(case_data["noPT"]["rho"], dtype=float)
    rho_pt = np.asarray(row_dm["rho"], dtype=float)
    ref_idx = int(np.argmax(THETA_GRID >= 0.262))
    delta_ln_rho_no = np.log(np.maximum(rho_no, 1.0e-300)) - math.log(float(rho_no[ref_idx]))
    delta_ln_xi = np.log(np.maximum(np.asarray(row_dm["xi"], dtype=float), 1.0e-300)) - math.log(float(row_dm["xi"][ref_idx]))
    delta_ln_rho_pt = np.log(np.maximum(rho_pt, 1.0e-300)) - math.log(float(rho_pt[ref_idx]))
    ax.plot(THETA_GRID, delta_ln_rho_no, color="black", ls="--", label=r"$\ln\rho_{\rm noPT}$")
    ax.plot(THETA_GRID, delta_ln_xi, color="#CC3311", ls=":", label=r"$\ln\xi$")
    ax.plot(THETA_GRID, delta_ln_rho_pt, color="#0077BB", ls="-", label=r"$\ln\rho_{\rm PT}$")
    ax.fill_between(THETA_GRID, delta_ln_rho_no, delta_ln_rho_pt, color="#0077BB", alpha=0.15)
    idx = int(0.62 * len(THETA_GRID))
    ax.annotate(
        "negative slope of ln xi\npartially cancels rho_noPT slope",
        xy=(THETA_GRID[idx], delta_ln_xi[idx]),
        xytext=(1.1, -0.6),
        arrowprops=dict(arrowstyle="->", lw=0.9),
        fontsize=9,
    )
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\Delta \ln \rho$")
    ax.set_title(r"$\ln\rho_{\rm PT} = \ln\xi + \ln\rho_{\rm noPT}$")
    ax.legend(loc="best")

    if pars is not None:
        ax = axes[3]
        tps = [1.5, 5.0, 15.0, 35.0]
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(tps)))
        for color, tp in zip(cmap, tps):
            ratio = ftilde_ratio(THETA_GRID, tp, VW_FIXED, pars)
            ax.plot(THETA_GRID, ratio, color=color, label=rf"$t_p={tp:g}$")
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\tilde f_{\rm anh}/f_{\rm anh}^{\rm noPT}$")
        ax.set_title(r"$\tilde f_{\rm anh}/f_{\rm anh}^{\rm noPT}$: flattens at large $t_p$")
        ax.legend(loc="best")

    save_pdf(fig, "isocurvature_proof.pdf")


def make_normalized_figure(case_data: dict[str, dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    k_arr = np.logspace(-3, math.log10(1.5), 3000)
    y_no = np.where(k_arr <= 1.0, 1.0, 0.0)
    ax.plot(k_arr, y_no, color=CASES[0].color, ls=CASES[0].ls, label="noPT  [P0 ratio = 1.00]")
    ax.axvline(1.0, color=CASES[0].color, ls=":", lw=1.0)

    box_lines = []
    for case in CASES[1:]:
        row = case_data[case.label]
        p0_ratio = float(row["P0_ratio"])
        kcut_ratio = float(row["kcut_ratio"])
        y = p0_ratio * np.where(k_arr <= kcut_ratio, 1.0, 0.0)
        ax.plot(
            k_arr,
            y,
            color=case.color,
            ls=case.ls,
            label=f"{case.label}  [P0 ratio = {p0_ratio:.1f}x,  k_cut = {kcut_ratio:.3f} k_cut^noPT]",
        )
        ax.axvline(kcut_ratio, color=case.color, ls=":", lw=1.0)
        box_lines.append(
            f"{case.label}\n"
            f"var ratio = {float(row['var_ratio']):.3f}\n"
            f"k_cut = {kcut_ratio:.3f}\n"
            f"P0 = {p0_ratio:.2f}x"
        )

    ax.text(
        0.98,
        0.98,
        "\n\n".join(box_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-3, 1.5)
    ax.set_xlabel(r"$k / k_{\rm cut}^{\rm noPT}$")
    ax.set_ylabel(r"Normalized white-noise amplitude")
    ax.legend(loc="lower left")
    save_pdf(fig, "isocurvature_normalized.pdf")


def make_physical_figure(case_data: dict[str, dict[str, object]], mphi_ev: float = M_PHI_DEFAULT_EV) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    k_arr = np.logspace(-1, 8, 5000)
    kcut0 = kcut_no_pt_mpc_inv(mphi_ev)

    p_no = p_white_noise(float(case_data["noPT"]["var"]), kcut0, k_arr)
    d_no = delta2_from_p(p_no, k_arr)
    ax.plot(k_arr, d_no, color=CASES[0].color, ls=CASES[0].ls, label=CASES[0].label)
    ax.axvline(kcut0, color=CASES[0].color, ls=":", lw=1.0)
    ax.text(kcut0 * 1.03, 3.0 * float(case_data["noPT"]["var"]) / (8.0 * math.pi**3), "noPT", rotation=90, va="bottom", ha="left", fontsize=8)

    for case in CASES[1:]:
        row = case_data[case.label]
        kcut = float(row["kcut_ratio"]) * kcut0
        p_pt = p_white_noise(float(row["var"]), kcut, k_arr)
        d_pt = delta2_from_p(p_pt, k_arr)
        ax.plot(k_arr, d_pt, color=case.color, ls=case.ls, label=case.label)
        ax.axvline(kcut, color=case.color, ls=":", lw=1.0)
        peak = 3.0 * float(row["var"]) / (8.0 * math.pi**3)
        ax.text(kcut * 1.03, peak, f"H*={case.hstar:g}", rotation=90, va="bottom", ha="left", fontsize=8, color=case.color)

    ax.plot(k_arr, delta2_cdm(k_arr), color="#888888", lw=1.5, label="CDM adiabatic")
    ax.text(0.98, 0.06, rf"$M_\phi = {mphi_ev:.0e}\ \mathrm{{eV}}$", transform=ax.transAxes, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-1, 1.0e8)
    ax.set_ylim(1.0e-12, 1.0e-4)
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.legend(loc="lower right")
    save_pdf(fig, "isocurvature_physical.pdf")


def make_pk_comparison_figure(case_data: dict[str, dict[str, object]], mphi_ev: float = M_PHI_DEFAULT_EV) -> None:
    k_dat, p_cdm = load_pcdm_dat()
    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    kcut0 = kcut_no_pt_mpc_inv(mphi_ev)

    p_no = p_white_noise(float(case_data["noPT"]["var"]), kcut0, k_dat)
    ax.plot(k_dat, p_cdm, color="#888888", lw=1.6, label=r"$P_{\rm CDM}(k)$ from data")
    ax.plot(k_dat, p_no, color=CASES[0].color, ls=CASES[0].ls, label=CASES[0].label)
    ax.axvline(kcut0, color=CASES[0].color, ls=":", lw=1.0)

    for case in CASES[1:]:
        row = case_data[case.label]
        kcut = float(row["kcut_ratio"]) * kcut0
        p_pt = p_white_noise(float(row["var"]), kcut, k_dat)
        ax.plot(k_dat, p_pt, color=case.color, ls=case.ls, label=case.label)
        ax.axvline(kcut, color=case.color, ls=":", lw=1.0)

    ax.text(0.98, 0.05, rf"$M_\phi = {mphi_ev:.0e}\ \mathrm{{eV}}$", transform=ax.transAxes, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-4, 1.0e2)
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)$")
    ax.legend(loc="best")
    save_pdf(fig, "Pk_comparison.pdf")


def _find_crossing_eh(kmin: float, kcut: float, p0: float) -> tuple[float, bool]:
    def f(x: float) -> float:
        return p0 - float(p_lin_eh(x))

    if f(kmin) >= 0.0:
        return kmin, False
    lo = kmin
    hi = kcut
    flo = f(lo)
    fhi = f(hi)
    if fhi < 0.0:
        return kcut, False
    for _ in range(120):
        mid = math.sqrt(lo * hi)
        fm = f(mid)
        if fm >= 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return hi, True


def make_pk_physical_corrected(
    case_data: dict[str, dict[str, object]],
    mphi_ev: float = 1.0e-20,
    hstar_bench: float = 0.05,
    vw_bench: float = 0.5,
    outdir: Path | None = None,
    filename: str = "Pk_physical_corrected.pdf",
    strict: bool = True,
) -> None:
    k_dat, p_dat = build_pcdm_segments()
    k_arr = np.logspace(-2, 1, 4000)
    p_cdm_arr = np.exp(np.interp(np.log(k_arr), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300))))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(k_arr, p_cdm_arr, color="#888888", lw=1.6)

    no_row = case_data["noPT"]
    kcut0 = kcut_no_pt_mpc_inv(mphi_ev)
    p_no_level = WHITE_NOISE_PREFAC * (kcut0 ** -3) * float(no_row["var"]) / DI2_EQ
    p_no = np.where(k_arr <= min(kcut0, 1.0e1), p_no_level, 0.0)
    ax.plot(k_arr, p_no, color="black", ls="--", label="noPT")
    p_ref_level = WHITE_NOISE_PREFAC * (KCUT_HARMONIC_REF_MPC ** -3) * VAR_HARMONIC_REF / DI2_EQ
    p_ref = np.where(k_arr <= min(KCUT_HARMONIC_REF_MPC, 1.0e1), p_ref_level, 0.0)
    ax.plot(k_arr, p_ref, color="#666666", ls="-.", lw=1.4, label=rf"noPT, var$=4/5$, $k_{{\rm cut}}={KCUT_HARMONIC_REF_MPC:.0f}$")
    ax.axvline(kcut0, color="black", ls=":", lw=1.0)

    hstar = hstar_bench
    vw = vw_bench
    rows = []
    var_no = float(no_row["var"])
    for beta, color in [(4.0, "#d62728"), (40.0, "#1f77b4")]:
        xi = xi_vec(THETA_GRID, hstar, vw, beta)
        rho = rho_from_xi(THETA_GRID, xi)
        _, var, _ = mean_and_var(THETA_GRID, rho)
        tp, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
        kcut = kcut_ratio * kcut0
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        p_level_nogrowth = WHITE_NOISE_PREFAC * (kcut ** -3) * var
        p_level = p_level_nogrowth / DI2_EQ
        p_arr = np.where(k_arr <= min(kcut, 1.0e1), p_level, 0.0)
        ax.plot(k_arr, p_arr, color=color, ls="-", label=rf"$\beta/H_*={beta:g}$  [$k_{{\rm cut}}$ ratio={kcut_ratio:.3f}]")
        ax.axvline(kcut, color=color, ls=":", lw=1.0)
        ratio_at_kcut_dat = p_level / float(np.exp(np.interp(np.log(K_CDM_DATA_LIMIT), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300))))) if kcut > K_CDM_DATA_LIMIT else p_level / float(np.exp(np.interp(np.log(kcut), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300)))))
        rows.append(
            {
                "beta": beta,
                "tp": tp,
                "t_onset": t_onset,
                "kcut": kcut,
                "kcut_ratio": kcut_ratio,
                "var": var,
                "var_ratio": var / var_no,
                "p0_ratio": p0_ratio,
                "p_level": p_level,
                "p_level_nogrowth": p_level_nogrowth,
                "ratio_at_kcut_dat": ratio_at_kcut_dat,
            }
        )

    ratio_no_at_kcut_dat = p_no_level / float(np.exp(np.interp(np.log(K_CDM_DATA_LIMIT), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300)))))

    ax.text(
        0.98,
        0.06,
        rf"$M_\phi = {mphi_ev:.0e}\ \mathrm{{eV}},\ H_*/M_\phi = {hstar_bench:g},\ v_w = {vw_bench:g}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    info_lines = []
    for row in rows:
        info_lines.append(
            rf"$\beta/H_*={row['beta']:g}$" "\n"
            rf"var = {row['var']:.3f}" "\n"
            rf"$D_i^{{-2}}$ var = {row['var'] / DI2_EQ:.3e}" "\n"
            rf"$P_0$ ratio = {row['p0_ratio']:.2f}"
        )
    ax.text(
        0.03,
        0.98,
        rf"$D_i(z_{{\rm eq}})^{{-2}} = (1+z_{{\rm eq}})^2 \approx {DI2_BOOST:.2e}$" + "\n" + r"CDM shown only from tabulated data",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.text(
        0.98,
        0.98,
        "\n\n".join(info_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-2, 1.0e1)
    _set_dynamic_log_yrange(ax, [p_cdm_arr, p_no] + [np.where(k_arr <= min(r["kcut"], 1.0e1), r["p_level"], 0.0) for r in rows])
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\ [{\rm Mpc}^3]$")
    ax.legend(loc="lower left")
    save_pdf_to(fig, OUTDIR if outdir is None else outdir, filename)

    print("Corrected physical P(k) benchmark [dat-only window]")
    print(f"DI2_BOOST = {DI2_BOOST:.0f}")
    print(f"WHITE_NOISE_PREFAC = {WHITE_NOISE_PREFAC:.6f}")
    print("Case       | k_cut [Mpc^-1] | var | P_i(no Di) [Mpc^3] | P_i(boost) [Mpc^3] | P_CDM_dat(k_eval) [Mpc^3] | P_i(boost)/P_dat | P0_ratio")
    p_dat_no = p_no_level / ratio_no_at_kcut_dat
    ratio_ref_at_kcut_dat = p_ref_level / p_dat_no
    print(
        f"noPT       | {kcut0:13.6g} | {float(no_row['var']):.6g} | {WHITE_NOISE_PREFAC * (kcut0 ** -3) * float(no_row['var']):16.6g} | "
        f"{p_no_level:16.6g} | {p_dat_no:20.6g} | {ratio_no_at_kcut_dat:14.6g} | {1.0:8.3f}"
    )
    print(
        f"ref 4/5    | {KCUT_HARMONIC_REF_MPC:13.6g} | {VAR_HARMONIC_REF:.6g} | {WHITE_NOISE_PREFAC * (KCUT_HARMONIC_REF_MPC ** -3) * VAR_HARMONIC_REF:16.6g} | "
        f"{p_ref_level:16.6g} | {p_dat_no:20.6g} | {ratio_ref_at_kcut_dat:14.6g} | {'-':>8}"
    )
    if strict and ratio_no_at_kcut_dat < 1.0:
        raise RuntimeError(f"noPT boosted P_i is below P_CDM_dat at k_eval: ratio={ratio_no_at_kcut_dat:.6g}")
    for row in rows:
        p_dat_eval = row["p_level"] / row["ratio_at_kcut_dat"]
        line = (
            f"beta={row['beta']:<4g} | {row['kcut']:13.6g} | {row['var']:.6g} | {row['p_level_nogrowth']:16.6g} | {row['p_level']:16.6g} | "
            f"{p_dat_eval:20.6g} | {row['ratio_at_kcut_dat']:14.6g} | {row['p0_ratio']:8.3f}"
        )
        print(line)
        if strict and row["ratio_at_kcut_dat"] < 1.0:
            raise RuntimeError(
                f"beta={row['beta']} boosted P_i is below P_CDM_dat at k_eval: ratio={row['ratio_at_kcut_dat']:.6g}"
            )


def make_pk_physical_nodi(
    case_data: dict[str, dict[str, object]],
    mphi_ev: float = 1.0e-20,
    hstar_bench: float = 0.05,
    vw_bench: float = 0.5,
    outdir: Path | None = None,
    filename: str = "Pk_physical_noDi.pdf",
) -> None:
    k_dat, p_dat = build_pcdm_segments()
    k_arr = np.logspace(-2, 1, 4000)
    p_cdm_arr = np.exp(np.interp(np.log(k_arr), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300))))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(k_arr, p_cdm_arr, color="#888888", lw=1.6, label=r"$P_{\rm CDM}(k)$")

    no_row = case_data["noPT"]
    kcut0 = kcut_no_pt_mpc_inv(mphi_ev)
    p_no_level = WHITE_NOISE_PREFAC * (kcut0 ** -3) * float(no_row["var"])
    p_no = np.where(k_arr <= min(kcut0, 1.0e1), p_no_level, 0.0)
    ax.plot(k_arr, p_no, color="black", ls="--", label=rf"noPT  [$k_{{\rm cut}}={kcut0:.1f}$]")
    p_ref_level = WHITE_NOISE_PREFAC * (KCUT_HARMONIC_REF_MPC ** -3) * VAR_HARMONIC_REF
    p_ref = np.where(k_arr <= min(KCUT_HARMONIC_REF_MPC, 1.0e1), p_ref_level, 0.0)
    ax.plot(k_arr, p_ref, color="#666666", ls="-.", lw=1.4, label=rf"noPT, var$=4/5$, $k_{{\rm cut}}={KCUT_HARMONIC_REF_MPC:.0f}$")
    ax.axvline(kcut0, color="black", ls=":", lw=1.0)

    rows = []
    hstar = hstar_bench
    vw = vw_bench
    var_no = float(no_row["var"])
    p_dat_limit = float(np.exp(np.interp(np.log(K_CDM_DATA_LIMIT), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300)))))
    for beta, color in [(4.0, "#d62728"), (40.0, "#1f77b4")]:
        xi = xi_vec(THETA_GRID, hstar, vw, beta)
        rho = rho_from_xi(THETA_GRID, xi)
        _, var, _ = mean_and_var(THETA_GRID, rho)
        _, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
        kcut = kcut_ratio * kcut0
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        p_level = WHITE_NOISE_PREFAC * (kcut ** -3) * var
        p_arr = np.where(k_arr <= min(kcut, 1.0e1), p_level, 0.0)
        ax.plot(k_arr, p_arr, color=color, ls="-", label=rf"$\beta/H_*={beta:g}$  [$k_{{\rm cut}}={kcut:.1f}$, $P_0={p0_ratio:.0f}\times$]")
        ax.axvline(kcut, color=color, ls=":", lw=1.0)
        ratio_at_kcut_dat = p_level / (p_dat_limit if kcut > K_CDM_DATA_LIMIT else float(np.exp(np.interp(np.log(kcut), np.log(k_dat), np.log(np.maximum(p_dat, 1.0e-300))))))
        rows.append(
            {
                "beta": beta,
                "kcut": kcut,
                "p_level": p_level,
                "ratio_at_kcut_dat": ratio_at_kcut_dat,
                "p0_ratio": p0_ratio,
            }
        )

    ratio_no_at_kcut_dat = p_no_level / p_dat_limit

    ax.text(
        0.98,
        0.06,
        rf"$M_\phi = {mphi_ev:.0e}\ \mathrm{{eV}},\ H_*/M_\phi = {hstar_bench:g},\ v_w = {vw_bench:g}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.text(
        0.03,
        0.98,
        r"$D_i(z_{\rm eq})^2$ not included" + "\n" + r"CDM shown only from tabulated data",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-2, 1.0e1)
    _set_dynamic_log_yrange(ax, [p_cdm_arr, p_no] + [np.where(k_arr <= min(r["kcut"], 1.0e1), r["p_level"], 0.0) for r in rows])
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\ [{\rm Mpc}^3]$")
    ax.legend(loc="lower left")
    save_pdf_to(fig, OUTDIR if outdir is None else outdir, filename)

    print("Physical P(k) benchmark without D_i^2 [dat-only window]")
    print("Case       | k_cut [Mpc^-1] | P_i(no Di) [Mpc^3] | P_i/P_dat(k_eval) | P0_ratio")
    print(
        f"noPT       | {kcut0:13.6g} | {p_no_level:16.6g} | {ratio_no_at_kcut_dat:14.6g} | {1.0:8.3f}"
    )
    print(
        f"ref 4/5    | {KCUT_HARMONIC_REF_MPC:13.6g} | {p_ref_level:16.6g} | {p_ref_level / p_dat_limit:14.6g} | {'-':>8}"
    )
    for row in rows:
        print(
            f"beta={row['beta']:<4g} | {row['kcut']:13.6g} | {row['p_level']:16.6g} | {row['ratio_at_kcut_dat']:14.6g} | {row['p0_ratio']:8.3f}"
        )


def make_pk_physical_corrected_hunits(
    case_data: dict[str, dict[str, object]],
    mphi_ev: float = 1.0e-20,
    hstar_bench: float = 0.05,
    vw_bench: float = 0.8,
    outdir: Path | None = None,
    filename: str = "Pk_physical_corrected_hunits.pdf",
) -> None:
    k_plot_max = 5000.0
    k_dat_mpc, p_dat_mpc = load_pcdm_dat_physical()
    k_arr_mpc = np.logspace(math.log10(0.5 * H0_REDUCED), math.log10(min(k_plot_max * H0_REDUCED, float(k_dat_mpc.max()))), 4000)
    p_cdm_arr_mpc = np.exp(np.interp(np.log(k_arr_mpc), np.log(k_dat_mpc), np.log(np.maximum(p_dat_mpc, 1.0e-300))))

    k_dat = k_mpc_to_hmpc(k_dat_mpc)
    k_arr = k_mpc_to_hmpc(k_arr_mpc)
    p_dat = p_mpc3_to_hinv3_mpc3(p_dat_mpc)
    p_cdm_arr = p_mpc3_to_hinv3_mpc3(p_cdm_arr_mpc)

    fig, ax = plt.subplots(figsize=(STYLE_1COL.width, 2.75), constrained_layout=True)
    curve_beta4, curve_betainf = viridis_colors(2, start=0.22, end=0.82)
    data_lya = "#C23B22"
    data_lrg = "#1B6CA8"
    ax.plot(k_arr, p_cdm_arr, color="black", ls="--", lw=1.6)

    no_row = case_data["noPT"]
    kcut0_mpc = kcut_no_pt_mpc_inv(mphi_ev)
    p_no_level_mpc = WHITE_NOISE_PREFAC * (kcut0_mpc ** -3) * float(no_row["var"]) / DI2_EQ
    kcut0 = k_mpc_to_hmpc(kcut0_mpc)
    p_no_level = p_mpc3_to_hinv3_mpc3(p_no_level_mpc)
    p_no = np.where(k_arr <= min(kcut0, k_plot_max), p_no_level, 0.0)
    ax.plot(k_arr, p_cdm_arr + p_no, color="#444444", ls="-", lw=1.5, alpha=0.95)
    ax.axvline(kcut0, color="#444444", ls=":", lw=1.0)

    hstar = hstar_bench
    vw = vw_bench
    rows = []
    var_no = float(no_row["var"])
    for label, beta, color, ls, alpha in [
        ("4", 4.0, curve_beta4, "-", 1.0),
    ]:
        pt = get_pt_case_summary(hstar, vw, beta)
        var = float(pt["var"])
        t_onset = float(pt["t_onset"])
        kcut_mpc = float(pt["kcut_ratio"]) * kcut0_mpc
        p_level_nogrowth_mpc = WHITE_NOISE_PREFAC * (kcut_mpc ** -3) * var
        p_level_mpc = p_level_nogrowth_mpc / DI2_EQ
        kcut = k_mpc_to_hmpc(kcut_mpc)
        p_level = p_mpc3_to_hinv3_mpc3(p_level_mpc)
        p_arr = np.where(k_arr <= min(kcut, k_plot_max), p_level, 0.0)
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        ax.plot(k_arr, p_cdm_arr + p_arr, color=color, ls=ls, alpha=alpha)
        ax.axvline(kcut, color=color, ls=":", lw=0.9, alpha=0.75)
        rows.append({"kcut": kcut, "p_level": p_level, "beta": label, "p0_ratio": p0_ratio})

    pt_inf = get_pt_case_summary_beta_infty(hstar, vw)
    var_inf = float(pt_inf["var"])
    kcut_inf_mpc = float(pt_inf["kcut_ratio"]) * kcut0_mpc
    p_inf_level_mpc = WHITE_NOISE_PREFAC * (kcut_inf_mpc ** -3) * var_inf / DI2_EQ
    kcut_inf = k_mpc_to_hmpc(kcut_inf_mpc)
    p_inf_level = p_mpc3_to_hinv3_mpc3(p_inf_level_mpc)
    p_inf = np.where(k_arr <= min(kcut_inf, k_plot_max), p_inf_level, 0.0)
    ax.plot(k_arr, p_cdm_arr + p_inf, color=curve_betainf, ls="-", alpha=0.98)
    ax.axvline(kcut_inf, color=curve_betainf, ls=":", lw=0.9, alpha=0.75)
    p0_ratio_inf = (var_inf / var_no) * ((float(pt_inf["t_onset"]) / TOSC) ** 1.5)
    rows.append({"kcut": kcut_inf, "p_level": p_inf_level, "beta": r"\infty", "p0_ratio": p0_ratio_inf})

    if LYA_DAT_PATH.exists():
        k_lya, p_lya, s_lya = load_external_pk_dat(LYA_DAT_PATH)
        ax.errorbar(
            k_lya,
            p_lya,
            yerr=s_lya,
            fmt="s",
            ms=2.2,
            lw=0.8,
            elinewidth=0.7,
            capsize=2.0,
            color=data_lya,
            mfc="white",
            mec=data_lya,
            zorder=5,
        )
    if LRG_DAT_PATH.exists():
        k_lrg, p_lrg, s_lrg = load_external_pk_dat(LRG_DAT_PATH)
        ax.errorbar(
            k_lrg,
            p_lrg,
            yerr=s_lrg,
            fmt="o",
            ms=2.2,
            lw=0.8,
            elinewidth=0.7,
            capsize=2.0,
            color=data_lrg,
            mfc="white",
            mec=data_lrg,
            zorder=5,
        )

    key_lines = [
        (curve_beta4, "-", r"$4$"),
        (curve_betainf, "-", r"$\infty$"),
        ("#444444", "-", r"no PT"),
    ]
    x0 = 0.71
    x1 = 0.78
    y0 = 0.875
    dy = 0.072
    ax.text(
        0.71,
        0.95,
        r"$\beta/H_*$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=STYLE_1COL.xtick_labelsize,
        color="black",
        zorder=2,
    )
    for i, (color, ls, label) in enumerate(key_lines):
        y = y0 - i * dy
        ax.plot([x0, x1], [y, y], transform=ax.transAxes, color=color, ls=ls, lw=2.2, clip_on=False, zorder=2)
        ax.text(
            x1 + 0.018,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=STYLE_1COL.xtick_labelsize,
            color="black",
            zorder=2,
        )
    ax.grid(False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, k_plot_max)
    _set_dynamic_log_yrange(ax, [p_cdm_arr, p_cdm_arr + p_no] + [p_cdm_arr + np.where(k_arr <= min(r["kcut"], k_plot_max), r["p_level"], 0.0) for r in rows])
    ax.set_ylim(1.0e-4, 1.0e3)
    ax.xaxis.set_major_formatter(decimal_log_tick_formatter())
    ax.yaxis.set_major_formatter(decimal_log_tick_formatter())
    ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\ [h^{-3}{\rm Mpc}^{3}]$")
    save_pdf_to(fig, OUTDIR if outdir is None else outdir, filename)


def make_pk_physical_nodi_hunits(
    case_data: dict[str, dict[str, object]],
    mphi_ev: float = 1.0e-20,
    hstar_bench: float = 0.05,
    vw_bench: float = 0.5,
    outdir: Path | None = None,
    filename: str = "Pk_physical_noDi_hunits.pdf",
) -> None:
    k_dat_mpc, p_dat_mpc = build_pcdm_segments()
    k_arr_mpc = np.logspace(-2, 1, 4000)
    p_cdm_arr_mpc = np.exp(np.interp(np.log(k_arr_mpc), np.log(k_dat_mpc), np.log(np.maximum(p_dat_mpc, 1.0e-300))))

    k_dat = k_mpc_to_hmpc(k_dat_mpc)
    k_arr = k_mpc_to_hmpc(k_arr_mpc)
    p_dat = p_mpc3_to_hinv3_mpc3(p_dat_mpc)
    p_cdm_arr = p_mpc3_to_hinv3_mpc3(p_cdm_arr_mpc)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(k_arr, p_cdm_arr, color="#888888", lw=1.6, label=r"$P_{\rm CDM}(k)$")

    no_row = case_data["noPT"]
    kcut0_mpc = kcut_no_pt_mpc_inv(mphi_ev)
    kcut0 = k_mpc_to_hmpc(kcut0_mpc)
    p_no_level = p_mpc3_to_hinv3_mpc3(WHITE_NOISE_PREFAC * (kcut0_mpc ** -3) * float(no_row["var"]))
    p_ref_level = p_mpc3_to_hinv3_mpc3(WHITE_NOISE_PREFAC * (KCUT_HARMONIC_REF_MPC ** -3) * VAR_HARMONIC_REF)
    kcut_ref = k_mpc_to_hmpc(KCUT_HARMONIC_REF_MPC)
    p_no = np.where(k_arr <= min(kcut0, k_mpc_to_hmpc(1.0e1)), p_no_level, 0.0)
    p_ref = np.where(k_arr <= min(kcut_ref, k_mpc_to_hmpc(1.0e1)), p_ref_level, 0.0)
    ax.plot(k_arr, p_no, color="black", ls="--", label=rf"noPT  [$k_{{\rm cut}}={kcut0:.1f}$]")
    ax.plot(k_arr, p_ref, color="#666666", ls="-.", lw=1.4, label=rf"noPT, var$=4/5$, $k_{{\rm cut}}={kcut_ref:.0f}$")
    ax.axvline(kcut0, color="black", ls=":", lw=1.0)

    rows = []
    hstar = hstar_bench
    vw = vw_bench
    var_no = float(no_row["var"])
    for beta, color in [(4.0, "#d62728"), (40.0, "#1f77b4")]:
        xi = xi_vec(THETA_GRID, hstar, vw, beta)
        rho = rho_from_xi(THETA_GRID, xi)
        _, var, _ = mean_and_var(THETA_GRID, rho)
        _, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
        kcut_mpc = kcut_ratio * kcut0_mpc
        kcut = k_mpc_to_hmpc(kcut_mpc)
        p_level = p_mpc3_to_hinv3_mpc3(WHITE_NOISE_PREFAC * (kcut_mpc ** -3) * var)
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        p_arr = np.where(k_arr <= min(kcut, k_mpc_to_hmpc(1.0e1)), p_level, 0.0)
        ax.plot(k_arr, p_arr, color=color, ls="-", label=rf"$\beta/H_*={beta:g}$  [$k_{{\rm cut}}={kcut:.1f}$, $P_0={p0_ratio:.0f}\times$]")
        ax.axvline(kcut, color=color, ls=":", lw=1.0)
        rows.append({"kcut": kcut, "p_level": p_level})

    ax.text(
        0.98,
        0.06,
        rf"$M_\phi = {mphi_ev:.0e}\ \mathrm{{eV}},\ H_*/M_\phi = {hstar_bench:g},\ v_w = {vw_bench:g}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.text(
        0.03,
        0.98,
        r"$D_i(z_{\rm eq})$ not included",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(k_mpc_to_hmpc(1.0e-2), k_mpc_to_hmpc(1.0e1))
    _set_dynamic_log_yrange(ax, [p_cdm_arr, p_no, p_ref] + [np.where(k_arr <= min(r["kcut"], k_mpc_to_hmpc(1.0e1)), r["p_level"], 0.0) for r in rows])
    ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\ [h^{-3}{\rm Mpc}^{3}]$")
    ax.legend(loc="lower left")
    save_pdf_to(fig, OUTDIR if outdir is None else outdir, filename)


def make_delta2_physical_nodi(case_data: dict[str, dict[str, object]], mphi_ev: float = 1.0e-20) -> None:
    k_dat, p_dat = build_pcdm_segments()
    k_arr = np.logspace(-4, math.log10(5.0e2), 5000)
    p_cdm_arr = p_cdm_stitched(k_arr, k_dat, p_dat)
    p_eh_arr = np.asarray(p_lin_eh(k_arr), dtype=float)
    d2_cdm_arr = delta2_from_p(p_cdm_arr, k_arr)
    d2_eh_arr = delta2_from_p(p_eh_arr, k_arr)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    mask_dat = k_arr <= K_CDM_DATA_LIMIT
    mask_eh = k_arr >= K_CDM_DATA_LIMIT
    ax.plot(k_arr[mask_dat], d2_cdm_arr[mask_dat], color="#888888", lw=1.6, label=r"$\Delta^2_{\rm CDM}(k)$ [nonlinear | linear EH]")
    ax.plot(k_arr[mask_eh], d2_eh_arr[mask_eh], color="#888888", lw=1.6)
    ax.axvline(K_CDM_DATA_LIMIT, color="#888888", ls=":", lw=1.0)
    ax.text(K_CDM_DATA_LIMIT * 1.04, d2_eh_arr[mask_eh][0] * 1.8, "CDM data limit", rotation=90, color="#666666", fontsize=8, va="bottom")

    no_row = case_data["noPT"]
    kcut0 = kcut_no_pt_mpc_inv(mphi_ev)
    p_no_level = WHITE_NOISE_PREFAC * (kcut0 ** -3) * float(no_row["var"])
    p_no = np.where(k_arr <= kcut0, p_no_level, 0.0)
    d2_no = delta2_from_p(p_no, k_arr)
    ax.plot(k_arr, d2_no, color="black", ls="--", label=rf"noPT  [$k_{{\rm cut}}={kcut0:.1f}$]")
    ax.axvline(kcut0, color="black", ls=":", lw=1.0)

    hstar = 0.05
    vw = 0.5
    var_no = float(no_row["var"])
    for beta, color in [(4.0, "#d62728"), (40.0, "#1f77b4")]:
        xi = xi_vec(THETA_GRID, hstar, vw, beta)
        rho = rho_from_xi(THETA_GRID, xi)
        _, var, _ = mean_and_var(THETA_GRID, rho)
        _, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
        kcut = kcut_ratio * kcut0
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        p_level = WHITE_NOISE_PREFAC * (kcut ** -3) * var
        p_arr = np.where(k_arr <= kcut, p_level, 0.0)
        d2_arr = delta2_from_p(p_arr, k_arr)
        ax.plot(k_arr, d2_arr, color=color, ls="-", label=rf"$\beta/H_*={beta:g}$  [$k_{{\rm cut}}={kcut:.1f}$, $P_0={p0_ratio:.0f}\times$]")
        ax.axvline(kcut, color=color, ls=":", lw=1.0)

    k_cross_no, _ = _find_crossing_eh(1.0e-4, kcut0, p_no_level)
    d2_cross = delta2_from_p(np.array([p_no_level]), np.array([k_cross_no]))[0]
    ax.axvline(k_cross_no, color="black", ls="--", lw=0.9, alpha=0.8)
    ax.annotate(
        rf"$\Delta_i^2 = \Delta_{{\rm CDM}}^2$" + "\n" + rf"$(k={k_cross_no:.1f}\ \mathrm{{Mpc}}^{{-1}})$",
        xy=(k_cross_no, d2_cross),
        xytext=(k_cross_no * 2.8, d2_cross * 6.0),
        arrowprops=dict(arrowstyle="->", lw=0.9),
        fontsize=9,
    )

    k_shade = np.logspace(math.log10(k_cross_no), math.log10(kcut0), 1200)
    d2_eh_shade = delta2_from_p(np.asarray(p_lin_eh(k_shade), dtype=float), k_shade)
    d2_i_shade = delta2_from_p(np.full_like(k_shade, p_no_level), k_shade)
    ax.fill_between(k_shade, d2_eh_shade, d2_i_shade, color="#d62728", alpha=0.12)

    ax.text(
        0.98,
        0.06,
        r"$M_\phi = 10^{-20}\ \mathrm{eV},\ H_*/M_\phi = 0.05,\ v_w = 0.5$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.text(
        0.03,
        0.98,
        r"$D_i(z_{\rm eq})^2$ not included",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-4, 5.0e2)
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.legend(loc="upper left")
    save_pdf(fig, "Delta2_physical_noDi.pdf")


def make_pk_physical_corrected_kpc(case_data: dict[str, dict[str, object]], mphi_ev: float = 1.0e-20) -> None:
    k_dat_mpc, p_dat_mpc = build_pcdm_segments()
    k_arr_mpc = np.logspace(-4, math.log10(5.0e2), 5000)
    p_cdm_arr_mpc = p_cdm_stitched(k_arr_mpc, k_dat_mpc, p_dat_mpc)
    p_eh_arr_mpc = np.asarray(p_lin_eh(k_arr_mpc), dtype=float)

    k_arr = k_arr_mpc * MPC_INV_TO_KPC_INV
    k_dat = k_dat_mpc * MPC_INV_TO_KPC_INV
    p_cdm_arr = p_cdm_arr_mpc * MPC3_TO_KPC3
    p_eh_arr = p_eh_arr_mpc * MPC3_TO_KPC3

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    mask_dat = k_arr_mpc <= K_CDM_DATA_LIMIT
    mask_eh = k_arr_mpc >= K_CDM_DATA_LIMIT
    ax.plot(k_arr[mask_dat], p_cdm_arr[mask_dat], color="#888888", lw=1.6, label=r"$P_{\rm CDM}(k)$ [nonlinear | linear EH]")
    ax.plot(k_arr[mask_eh], p_eh_arr[mask_eh], color="#888888", lw=1.6)
    ax.axvline(K_CDM_DATA_LIMIT * MPC_INV_TO_KPC_INV, color="#888888", ls=":", lw=1.0)
    ax.text(K_CDM_DATA_LIMIT * MPC_INV_TO_KPC_INV * 1.04, p_eh_arr[mask_eh][0] * 2.0, "CDM data limit", rotation=90, color="#666666", fontsize=8, va="bottom")

    no_row = case_data["noPT"]
    kcut0_mpc = kcut_no_pt_mpc_inv(mphi_ev)
    kcut0 = kcut0_mpc * MPC_INV_TO_KPC_INV
    p_no_level_mpc = WHITE_NOISE_PREFAC * (kcut0_mpc ** -3) * float(no_row["var"]) / DI2_EQ
    p_no_level = p_no_level_mpc * MPC3_TO_KPC3
    p_no = np.where(k_arr <= kcut0, p_no_level, 0.0)
    ax.plot(k_arr, p_no, color="black", ls="--", label=rf"noPT  [$k_{{\rm cut}}={kcut0:.3g}$]")
    ax.axvline(kcut0, color="black", ls=":", lw=1.0)

    rows = []
    hstar = 0.05
    vw = 0.5
    var_no = float(no_row["var"])
    p_dat_limit_mpc = float(np.exp(np.interp(np.log(K_CDM_DATA_LIMIT), np.log(k_dat_mpc), np.log(np.maximum(p_dat_mpc, 1.0e-300)))))
    for beta, color in [(4.0, "#d62728"), (40.0, "#1f77b4")]:
        xi = xi_vec(THETA_GRID, hstar, vw, beta)
        rho = rho_from_xi(THETA_GRID, xi)
        _, var, _ = mean_and_var(THETA_GRID, rho)
        _, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
        kcut_mpc = kcut_ratio * kcut0_mpc
        kcut = kcut_mpc * MPC_INV_TO_KPC_INV
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        p_level_mpc = WHITE_NOISE_PREFAC * (kcut_mpc ** -3) * var / DI2_EQ
        p_level = p_level_mpc * MPC3_TO_KPC3
        p_arr = np.where(k_arr <= kcut, p_level, 0.0)
        ax.plot(k_arr, p_arr, color=color, ls="-", label=rf"$\beta/H_*={beta:g}$  [$k_{{\rm cut}}={kcut:.3g}$, $P_0={p0_ratio:.0f}\times$]")
        ax.axvline(kcut, color=color, ls=":", lw=1.0)
        k_cross_mpc, bracketed = _find_crossing_eh(1.0e-4, kcut_mpc, p_level_mpc)
        rows.append(
            {
                "beta": beta,
                "kcut": kcut,
                "k_cross": k_cross_mpc * MPC_INV_TO_KPC_INV,
                "p_level": p_level,
                "p0_ratio": p0_ratio,
                "ratio_at_kcut_eh": p_level_mpc / float(p_lin_eh(kcut_mpc)),
                "ratio_at_kcut_dat": p_level_mpc / (p_dat_limit_mpc if kcut_mpc > K_CDM_DATA_LIMIT else float(np.exp(np.interp(np.log(kcut_mpc), np.log(k_dat_mpc), np.log(np.maximum(p_dat_mpc, 1.0e-300)))))),
                "bracketed": bracketed,
            }
        )

    k_cross_no_mpc, _ = _find_crossing_eh(1.0e-4, kcut0_mpc, p_no_level_mpc)
    k_cross_no = k_cross_no_mpc * MPC_INV_TO_KPC_INV
    ax.axvline(k_cross_no, color="black", ls="--", lw=0.9, alpha=0.8)
    ax.annotate(
        rf"$P_i = P_{{\rm CDM}}^{{\rm lin}}$" + "\n" + rf"$(k={k_cross_no:.3g}\ \mathrm{{kpc}}^{{-1}})$",
        xy=(k_cross_no, p_no_level),
        xytext=(k_cross_no * 3.0, p_no_level * 5.0),
        arrowprops=dict(arrowstyle="->", lw=0.9),
        fontsize=9,
    )

    k_shade_mpc = np.logspace(math.log10(k_cross_no_mpc), math.log10(kcut0_mpc), 1200)
    k_shade = k_shade_mpc * MPC_INV_TO_KPC_INV
    p_eh_shade = np.asarray(p_lin_eh(k_shade_mpc), dtype=float) * MPC3_TO_KPC3
    p_i_shade = np.full_like(k_shade, p_no_level)
    ax.fill_between(k_shade, p_eh_shade, p_i_shade, color="#d62728", alpha=0.12)

    ax.text(
        0.98,
        0.06,
        r"$M_\phi = 10^{-20}\ \mathrm{eV},\ H_*/M_\phi = 0.05,\ v_w = 0.5$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.text(
        0.03,
        0.98,
        rf"$D_i(z_{{\rm eq}})^{{-2}} \approx {DI2_BOOST:.2e}$ included",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-7, 5.0e-1)
    ax.set_xlabel(r"$k\ [{\rm kpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\ [{\rm kpc}^3]$")
    ax.legend(loc="lower left")
    save_pdf(fig, "Pk_physical_corrected_kpc.pdf")

    print("Corrected physical P(k) benchmark [kpc units]")
    print("Case       | k_cut [kpc^-1] | P_i(Di^2) [kpc^3] | k_cross(EH) [kpc^-1] | P_i/P_EH(k_cut) | P_i/P_dat(k_cut) | P0_ratio")
    print(
        f"noPT       | {kcut0:13.6g} | {p_no_level:16.6g} | {k_cross_no:17.6g} | "
        f"{p_no_level_mpc / float(p_lin_eh(kcut0_mpc)):14.6g} | {p_no_level_mpc / p_dat_limit_mpc:15.6g} | {1.0:8.3f}"
    )
    for row in rows:
        print(
            f"beta={row['beta']:<4g} | {row['kcut']:13.6g} | {row['p_level']:16.6g} | {row['k_cross']:17.6g} | "
            f"{row['ratio_at_kcut_eh']:14.6g} | {row['ratio_at_kcut_dat']:15.6g} | {row['p0_ratio']:8.3f}"
            + ("" if row["bracketed"] else "   WARNING: crossing not bracketed")
        )


def make_delta2_physical_corrected_kpc(case_data: dict[str, dict[str, object]], mphi_ev: float = 1.0e-20) -> None:
    k_dat_mpc, p_dat_mpc = build_pcdm_segments()
    k_arr_mpc = np.logspace(-4, math.log10(5.0e2), 5000)
    p_cdm_arr_mpc = p_cdm_stitched(k_arr_mpc, k_dat_mpc, p_dat_mpc)
    p_eh_arr_mpc = np.asarray(p_lin_eh(k_arr_mpc), dtype=float)
    d2_cdm_arr = delta2_from_p(p_cdm_arr_mpc, k_arr_mpc)
    d2_eh_arr = delta2_from_p(p_eh_arr_mpc, k_arr_mpc)
    k_arr = k_arr_mpc * MPC_INV_TO_KPC_INV

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    mask_dat = k_arr_mpc <= K_CDM_DATA_LIMIT
    mask_eh = k_arr_mpc >= K_CDM_DATA_LIMIT
    ax.plot(k_arr[mask_dat], d2_cdm_arr[mask_dat], color="#888888", lw=1.6, label=r"$\Delta^2_{\rm CDM}(k)$ [nonlinear | linear EH]")
    ax.plot(k_arr[mask_eh], d2_eh_arr[mask_eh], color="#888888", lw=1.6)
    ax.axvline(K_CDM_DATA_LIMIT * MPC_INV_TO_KPC_INV, color="#888888", ls=":", lw=1.0)
    ax.text(K_CDM_DATA_LIMIT * MPC_INV_TO_KPC_INV * 1.04, d2_eh_arr[mask_eh][0] * 1.8, "CDM data limit", rotation=90, color="#666666", fontsize=8, va="bottom")

    no_row = case_data["noPT"]
    kcut0_mpc = kcut_no_pt_mpc_inv(mphi_ev)
    kcut0 = kcut0_mpc * MPC_INV_TO_KPC_INV
    p_no_level_mpc = WHITE_NOISE_PREFAC * (kcut0_mpc ** -3) * float(no_row["var"]) / DI2_EQ
    p_no_mpc = np.where(k_arr_mpc <= kcut0_mpc, p_no_level_mpc, 0.0)
    d2_no = delta2_from_p(p_no_mpc, k_arr_mpc)
    ax.plot(k_arr, d2_no, color="black", ls="--", label=rf"noPT  [$k_{{\rm cut}}={kcut0:.3g}$]")
    ax.axvline(kcut0, color="black", ls=":", lw=1.0)

    hstar = 0.05
    vw = 0.5
    var_no = float(no_row["var"])
    for beta, color in [(4.0, "#d62728"), (40.0, "#1f77b4")]:
        xi = xi_vec(THETA_GRID, hstar, vw, beta)
        rho = rho_from_xi(THETA_GRID, xi)
        _, var, _ = mean_and_var(THETA_GRID, rho)
        _, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
        kcut_mpc = kcut_ratio * kcut0_mpc
        kcut = kcut_mpc * MPC_INV_TO_KPC_INV
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        p_level_mpc = WHITE_NOISE_PREFAC * (kcut_mpc ** -3) * var / DI2_EQ
        p_arr_mpc = np.where(k_arr_mpc <= kcut_mpc, p_level_mpc, 0.0)
        d2_arr = delta2_from_p(p_arr_mpc, k_arr_mpc)
        ax.plot(k_arr, d2_arr, color=color, ls="-", label=rf"$\beta/H_*={beta:g}$  [$k_{{\rm cut}}={kcut:.3g}$, $P_0={p0_ratio:.0f}\times$]")
        ax.axvline(kcut, color=color, ls=":", lw=1.0)

    k_cross_no_mpc, _ = _find_crossing_eh(1.0e-4, kcut0_mpc, p_no_level_mpc)
    k_cross_no = k_cross_no_mpc * MPC_INV_TO_KPC_INV
    d2_cross = delta2_from_p(np.array([p_no_level_mpc]), np.array([k_cross_no_mpc]))[0]
    ax.axvline(k_cross_no, color="black", ls="--", lw=0.9, alpha=0.8)
    ax.annotate(
        rf"$\Delta_i^2 = \Delta_{{\rm CDM}}^2$" + "\n" + rf"$(k={k_cross_no:.3g}\ \mathrm{{kpc}}^{{-1}})$",
        xy=(k_cross_no, d2_cross),
        xytext=(k_cross_no * 2.8, d2_cross * 6.0),
        arrowprops=dict(arrowstyle="->", lw=0.9),
        fontsize=9,
    )

    k_shade_mpc = np.logspace(math.log10(k_cross_no_mpc), math.log10(kcut0_mpc), 1200)
    k_shade = k_shade_mpc * MPC_INV_TO_KPC_INV
    d2_eh_shade = delta2_from_p(np.asarray(p_lin_eh(k_shade_mpc), dtype=float), k_shade_mpc)
    d2_i_shade = delta2_from_p(np.full_like(k_shade_mpc, p_no_level_mpc), k_shade_mpc)
    ax.fill_between(k_shade, d2_eh_shade, d2_i_shade, color="#d62728", alpha=0.12)

    ax.text(
        0.98,
        0.06,
        r"$M_\phi = 10^{-20}\ \mathrm{eV},\ H_*/M_\phi = 0.05,\ v_w = 0.5$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.text(
        0.03,
        0.98,
        rf"$D_i(z_{{\rm eq}})^{{-2}} \approx {DI2_BOOST:.2e}$ included",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-7, 5.0e-1)
    ax.set_xlabel(r"$k\ [{\rm kpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.legend(loc="upper left")
    save_pdf(fig, "Delta2_physical_corrected_kpc.pdf")


def make_delta2_physical_corrected(case_data: dict[str, dict[str, object]], mphi_ev: float = 1.0e-20) -> None:
    k_dat, p_dat = build_pcdm_segments()
    k_arr = np.logspace(-4, math.log10(5.0e2), 5000)
    p_cdm_arr = p_cdm_stitched(k_arr, k_dat, p_dat)
    p_eh_arr = np.asarray(p_lin_eh(k_arr), dtype=float)
    d2_cdm_arr = delta2_from_p(p_cdm_arr, k_arr)
    d2_eh_arr = delta2_from_p(p_eh_arr, k_arr)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    mask_dat = k_arr <= K_CDM_DATA_LIMIT
    mask_eh = k_arr >= K_CDM_DATA_LIMIT
    ax.plot(k_arr[mask_dat], d2_cdm_arr[mask_dat], color="#888888", lw=1.6, label=r"$\Delta^2_{\rm CDM}(k)$ [nonlinear | linear EH]")
    ax.plot(k_arr[mask_eh], d2_eh_arr[mask_eh], color="#888888", lw=1.6)
    ax.axvline(K_CDM_DATA_LIMIT, color="#888888", ls=":", lw=1.0)
    ax.text(K_CDM_DATA_LIMIT * 1.04, d2_eh_arr[mask_eh][0] * 1.8, "CDM data limit", rotation=90, color="#666666", fontsize=8, va="bottom")

    no_row = case_data["noPT"]
    kcut0 = kcut_no_pt_mpc_inv(mphi_ev)
    p_no_level = WHITE_NOISE_PREFAC * (kcut0 ** -3) * float(no_row["var"]) / DI2_EQ
    p_no = np.where(k_arr <= kcut0, p_no_level, 0.0)
    d2_no = delta2_from_p(p_no, k_arr)
    ax.plot(k_arr, d2_no, color="black", ls="--", label=rf"noPT  [$k_{{\rm cut}}={kcut0:.1f}$]")
    ax.axvline(kcut0, color="black", ls=":", lw=1.0)

    rows = []
    hstar = 0.05
    vw = 0.5
    var_no = float(no_row["var"])
    for beta, color in [(4.0, "#d62728"), (40.0, "#1f77b4")]:
        xi = xi_vec(THETA_GRID, hstar, vw, beta)
        rho = rho_from_xi(THETA_GRID, xi)
        _, var, _ = mean_and_var(THETA_GRID, rho)
        _, t_onset, kcut_ratio = t_onset_and_kcut_ratio(hstar, beta, vw)
        kcut = kcut_ratio * kcut0
        p0_ratio = (var / var_no) * ((t_onset / TOSC) ** 1.5)
        p_level = WHITE_NOISE_PREFAC * (kcut ** -3) * var / DI2_EQ
        p_arr = np.where(k_arr <= kcut, p_level, 0.0)
        d2_arr = delta2_from_p(p_arr, k_arr)
        ax.plot(k_arr, d2_arr, color=color, ls="-", label=rf"$\beta/H_*={beta:g}$  [$k_{{\rm cut}}={kcut:.1f}$, $P_0={p0_ratio:.0f}\times$]")
        ax.axvline(kcut, color=color, ls=":", lw=1.0)
        rows.append({"beta": beta, "kcut": kcut, "p0_ratio": p0_ratio})

    k_cross_no, _ = _find_crossing_eh(1.0e-4, kcut0, p_no_level)
    ax.axvline(k_cross_no, color="black", ls="--", lw=0.9, alpha=0.8)
    d2_cross = delta2_from_p(np.array([p_no_level]), np.array([k_cross_no]))[0]
    ax.annotate(
        rf"$\Delta_i^2 = \Delta_{{\rm CDM}}^2$" + "\n" + rf"$(k={k_cross_no:.1f}\ \mathrm{{Mpc}}^{{-1}})$",
        xy=(k_cross_no, d2_cross),
        xytext=(k_cross_no * 2.8, d2_cross * 6.0),
        arrowprops=dict(arrowstyle="->", lw=0.9),
        fontsize=9,
    )

    k_shade = np.logspace(math.log10(k_cross_no), math.log10(kcut0), 1200)
    d2_eh_shade = delta2_from_p(np.asarray(p_lin_eh(k_shade), dtype=float), k_shade)
    d2_i_shade = delta2_from_p(np.full_like(k_shade, p_no_level), k_shade)
    ax.fill_between(k_shade, d2_eh_shade, d2_i_shade, color="#d62728", alpha=0.12)

    ax.text(
        0.98,
        0.06,
        r"$M_\phi = 10^{-20}\ \mathrm{eV},\ H_*/M_\phi = 0.05,\ v_w = 0.5$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.text(
        0.03,
        0.98,
        rf"$D_i(z_{{\rm eq}})^{{-2}} \approx {DI2_BOOST:.2e}$ included",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.95),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-4, 5.0e2)
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.legend(loc="upper left")
    save_pdf(fig, "Delta2_physical_corrected.pdf")


def make_pvsk_per_mphi_batch(
    case_data: dict[str, dict[str, object]],
    masses_ev: list[float],
    hstar_bench: float = 1.0e-4,
    vw_bench: float = 0.5,
) -> None:
    for mass in masses_ev:
        exp = int(round(math.log10(mass)))
        tag = f"1em{abs(exp):02d}"
        make_pk_physical_corrected(
            case_data,
            mphi_ev=mass,
            hstar_bench=hstar_bench,
            vw_bench=vw_bench,
            outdir=PVSK_PER_MPHI_DIR,
            filename=f"Pk_physical_corrected_{tag}.pdf",
        )
        make_pk_physical_nodi(
            case_data,
            mphi_ev=mass,
            hstar_bench=hstar_bench,
            vw_bench=vw_bench,
            outdir=PVSK_PER_MPHI_DIR,
            filename=f"Pk_physical_noDi_{tag}.pdf",
        )


def main() -> None:
    run_unit_sanity_checks()
    case_data, var_no = build_case_data()
    print_summary(case_data, var_no)
    make_proof_figure(case_data)
    make_normalized_figure(case_data)
    make_physical_figure(case_data)
    make_pk_comparison_figure(case_data)
    make_pk_physical_corrected(case_data)
    make_pk_physical_nodi(case_data)
    make_pk_physical_corrected_kpc(case_data)
    make_delta2_physical_corrected(case_data)
    make_delta2_physical_nodi(case_data)
    make_delta2_physical_corrected_kpc(case_data)


if __name__ == "__main__":
    main()
