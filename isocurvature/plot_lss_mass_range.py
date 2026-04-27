#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_isocurvature as iso


OUTDIR = Path(__file__).resolve().parent / "outputs_lss"
OUTDIR.mkdir(parents=True, exist_ok=True)

MASS_LIST = [1.0e-20, 1.0e-21, 1.0e-22]


def save_pdf_to(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    fig.clf()


def main() -> None:
    # Redirect the imported plotting helpers to the dedicated LSS folder.
    iso.OUTDIR = OUTDIR
    case_data, var_no = iso.build_case_data()
    iso.print_summary(case_data, var_no)

    summary: dict[str, object] = {
        "masses_eV": MASS_LIST,
        "kcut_noPT_mpc_inv": {},
        "files": {},
    }

    for mass in MASS_LIST:
        kcut = iso.kcut_no_pt_mpc_inv(mass)
        mass_tag = f"{mass:.0e}".replace("-", "m").replace("+", "")
        summary["kcut_noPT_mpc_inv"][f"{mass:.0e}"] = kcut
        print(f"M_phi={mass:.0e} eV -> k_cut^noPT = {kcut:.6g} Mpc^-1")

        iso.make_physical_figure(case_data, mphi_ev=mass)
        (OUTDIR / "isocurvature_physical.pdf").replace(OUTDIR / f"isocurvature_physical_{mass_tag}.pdf")

        iso.make_pk_comparison_figure(case_data, mphi_ev=mass)
        (OUTDIR / "Pk_comparison.pdf").replace(OUTDIR / f"Pk_comparison_{mass_tag}.pdf")

        summary["files"][f"{mass:.0e}"] = {
            "physical": str((OUTDIR / f"isocurvature_physical_{mass_tag}.pdf").resolve()),
            "pk_comparison": str((OUTDIR / f"Pk_comparison_{mass_tag}.pdf").resolve()),
        }

    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
