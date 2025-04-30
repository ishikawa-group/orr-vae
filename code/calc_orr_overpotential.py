#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORR over-potential workflow (offset-based adsorption version)
============================================================

- Gas-phase optimisation   : O2, H2, H2O, OH, HO2(=OOH), O  (計6種)
- Adsorption optimisation  : O2*, OOH*, O*, OH*  (計4種)
  * Offsets to evaluate
      O2*  : (0,0) (0.5,0) (0.5,0.5)
      OOH* : (1,1) (1.5,1) (1.5,1.5)
      O*   : (2,2) (2.5,2) (2.5,2.5)
      OH*  : (3,3) (3.5,3) (3.5,3.5)

計算回数 : 6 (gas) + 1 (clean slab) + 12 (4 adsorbates × 3 offsets) = 19
最安エネルギーを各吸着種の代表値として採用し ΔE, η を評価する。
"""

from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.io import read

# ----- external helpers ----------------------------------------------------
from calc_orr_energy import (
    optimize_bulk,
    optimize_slab,          # clean slab or slab+adsorbate ともに使う
    optimize_gas,
    calc_adsorption_with_offset,
)
from tool import convert_numpy_types

# ---------------------------------------------------------------------------
# 1. 分子ライブラリ  (gas-phase は全種、adsorbate はサブセット)
# ---------------------------------------------------------------------------
MOLECULES: Dict[str, Atoms] = {
    # adsorbates (gas + adsorption)
    "OH":  Atoms("OH",  positions=[(0, 0, 0), (0, 0, 0.97)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, 0.723, 1.264), (1.666, 0, 1.007)]), #(0, 0, 0), (0, 0, 1.46), (0.939, 0, 1.705)を30度回転
    "O2":  Atoms("OO",  positions=[(0, 0, 0), (0, 0, 1.21)]),
    "O":   Atoms("O",   positions=[(0, 0, 0)]),
    # gas-phase only
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "H2":  Atoms("HH",  positions=[(0, 0, 0), (0, 0, 0.74)]),
}

GAS_ONLY: set[str] = {"H2", "H2O"}      # 吸着計算を行わない分子

ADSORBATES: Dict[str, List[Tuple[float, float]]] = {
    "O2":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "O":   [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "OH":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
}

SLAB_VACUUM = 30.0   # Å
GAS_BOX     = 15.0   # Å
ADS_HEIGHT  = 2.0    # Å

logger = logging.getLogger("orr_workflow")


# ---------------------------------------------------------------------------
# 2. Adsorption-energy utilities (offset ベース)
# ---------------------------------------------------------------------------
def calculate_required_molecules(
    opt_slab: Atoms,
    E_opt_slab: float,
    base_dir: Path,
    force: bool = False,
    calc_type: str = "mattersim",
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}
    base_dir.mkdir(parents=True, exist_ok=True)

    for mol_name, mol in MOLECULES.items():
        logger.info("=== %s ===", mol_name)
        mol_dir  = base_dir / mol_name
        gas_dir  = mol_dir / f"{mol_name}_gas"
        ads_dir  = mol_dir / "adsorption"
        gas_dir.mkdir(parents=True, exist_ok=True)
        ads_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 1. gas optimisation ----------------------------------
        gas_json  = gas_dir / "opt_result.json"
        xyz_gas   = gas_dir / "opt.xyz"
        if gas_json.exists() and xyz_gas.exists() and not force:
            E_gas   = json.load(gas_json.open())["E_opt"]
            opt_mol = read(xyz_gas)
            logger.info("  reuse gas-phase energy = %.3f eV", E_gas)
        else:
            opt_mol, E_gas = optimize_gas(mol_name, GAS_BOX, str(gas_dir), calc_type)
            opt_mol.write(xyz_gas)
            json.dump({"E_opt": float(E_gas)}, gas_json.open("w"))
        results.setdefault(mol_name, {})["E_gas"] = float(E_gas)

        # ---------- 2. adsorption skip for gas-only ----------------------
        if mol_name in GAS_ONLY:
            continue

        offsets = ADSORBATES[mol_name]
        offset_data: Dict[str, Dict[str, float]] = {}

        for ofst in offsets:
            key = f"ofst_{ofst[0]}_{ofst[1]}"
            ofst_json = ads_dir / f"{key}.json"
            # calc sub-dir → …/adsorption/ofst_x_y/
            work_dir   = ads_dir / key

            if ofst_json.exists() and (work_dir / ".done").exists() and not force:
                data    = json.load(ofst_json.open())
                E_total = data["E_total"]
                elapsed = data["elapsed"]
            else:
                E_total, elapsed = calc_adsorption_with_offset(
                    opt_slab, opt_mol, ofst, str(work_dir), calc_type
                )
                json.dump({"E_total": E_total,
                           "elapsed": elapsed}, ofst_json.open("w"))
                (work_dir / ".done").touch()
            offset_data[key] = {"E_total": E_total, "elapsed": elapsed}

        # ---------- 3. pick lowest-E -------------------------------------
        best_key, E_best = min(
            ((k, d["E_total"]) for k, d in offset_data.items()),
            key=lambda x: x[1]
        )
        E_ads_best = E_best - (E_opt_slab + E_gas)
        results[mol_name].update({
            "E_slab":        float(E_opt_slab),
            "E_total_best":  float(E_best),
            "best_offset":   best_key,
            "E_ads_best":    float(E_ads_best),
            "offsets":       offset_data,
        })
        logger.info("  -> best offset: %s   E_ads = %.3f eV",
                    best_key, E_ads_best)

    # ---------- 4. write summary -----------------------------------------
    json.dump(convert_numpy_types(results),
              (base_dir / "all_results.json").open("w"), indent=2)
    return results


# ---------------------------------------------------------------------------
# Reaction/overpotential part
# ---------------------------------------------------------------------------

def compute_reaction_energies(results: Dict[str, Any], E_slab: float) -> Tuple[List[float], Dict[str, float]]:
    """Return (deltaEs_for_overpotential, energies_dict)."""

    def e_gas(mol: str) -> float:  # convenience helpers
        return results[mol]["E_gas"]

    def e_total(mol: str) -> float:
        return results[mol]["E_total_best"]

    # gas‑phase energies ------------------------------------------------------
    E_H2_g   = e_gas("H2")
    E_H2O_g  = e_gas("H2O")
    # O2(g) energy corrected (SI of Bligaard/Nørskov) ------------------------
    E_O2_g   = 2 * (2.46 + E_H2O_g - E_H2_g)

    # slab+adsorbate total energies -----------------------------------------
    E_slab_O2  = e_total("O2")
    E_slab_OOH = e_total("HO2")     # HO2 = OOH*
    E_slab_O   = e_total("O")
    E_slab_OH  = e_total("OH")

    energies = {
        "E_H2_g":    E_H2_g,
        "E_H2O_g":   E_H2O_g,
        "E_O2_g":    E_O2_g,
        "E_slab":    E_slab,
        "E_slab_O2": E_slab_O2,
        "E_slab_OOH":E_slab_OOH,
        "E_slab_O":  E_slab_O,
        "E_slab_OH": E_slab_OH,
    }

    # reaction energies ΔE ----------------------------------------------------
    dE1 = E_slab_O2 - (E_slab + E_O2_g)                       # O2(g) + * → O2*
    dE2 = E_slab_OOH - (E_slab_O2 + 0.5 * E_H2_g)             # O2* + ½H2 → OOH*
    dE3 = (E_slab_O + E_H2O_g) - (E_slab_OOH + 0.5 * E_H2_g)  # OOH* + ½H2 → O* + H2O
    dE4 = E_slab_OH - (E_slab_O + 0.5 * E_H2_g)               # O* + ½H2 → OH*
    dE5 = (E_slab + E_H2O_g) - (E_slab_OH + 0.5 * E_H2_g)     # OH* + ½H2 → * + H2O

    deltaEs = [dE1 + dE2, dE3, dE4, dE5]  #   dE2 is combined with dE1
    energies.update({
        "dE1": dE1, "dE2": dE2, "dE3": dE3, "dE4": dE4, "dE5": dE5
    })
    return deltaEs, energies

# ---------------------------------------------------------------------------
# Overpotential function (unchanged, minor clean‑ups) ------------------------
# ---------------------------------------------------------------------------

def get_overpotential_orr(
    deltaEs: List[float],
    output_dir: Path,
    T: float = 298.15,
    verbose: bool = False,
) -> float:
    """Return ORR overpotential η (V) and save free‑energy diagram."""

    rxn_num = 4  # 4‑electron pathway
    assert len(deltaEs) == rxn_num, "deltaEs must contain 4 elements"

    # ZPE (eV) ---------------------------------------------------------------
    zpe = {
        "H2": 0.27, "H2O": 0.56,
        "O2": 0.05 * 2,
        "O2ads": 0.0, "Oads": 0.07, "OHads": 0.36, "OOHads": 0.40,
    }
    # entropy term T*S -------------------------------------------------------
    S = {
        "H2": 0.41 / T, "H2O": 0.67 / T, "O2": 0.63 / T,
        "Oads": 0.0, "OHads": 0.0, "OOHads": 0.0, "O2ads": 0.0,
    }

    zpe["O2"] = 0 + 2 * (zpe["H2O"] - zpe["H2"])
    S["O2"]   = 0 + 2 * (S["H2O"]  - S["H2"])

    deltaZPE = np.array([
        (zpe["OOHads"] - zpe["O2ads"] - 0.5 * zpe["H2"]) + (zpe["O2ads"] - zpe["O2"]),
        zpe["Oads"] + zpe["H2O"] - zpe["OOHads"] - 0.5 * zpe["H2"],
        zpe["OHads"] - zpe["Oads"] - 0.5 * zpe["H2"],
        zpe["H2O"] - zpe["OHads"] - 0.5 * zpe["H2"],
    ])
    deltaTS = np.array([
        (T * S["OOHads"] - T * S["O2ads"] - 0.5 * T * S["H2"]) + (T * S["O2ads"] - T * S["O2"]),
        T * S["Oads"] + T * S["H2O"] - T * S["OOHads"] - 0.5 * T * S["H2"],
        T * S["OHads"] - T * S["Oads"] - 0.5 * T * S["H2"],
        T * S["H2O"] - T * S["OHads"] - 0.5 * T * S["H2"],
    ])

    deltaEs = np.array(deltaEs)
    deltaG_U0 = deltaEs + deltaZPE - deltaTS  # ΔG at U=0 V

    G_profile_U0 = np.concatenate(([0.0], np.cumsum(deltaG_U0)))
    U_eq = 1.23
    G_profile_Ueq = G_profile_U0 - np.arange(rxn_num + 1) * (-1) * U_eq

    diffG_U0 = np.diff(G_profile_U0)
    diffG_eq = np.diff(G_profile_Ueq)

    G_ORR = np.max(diffG_U0)
    U_L   = abs(G_ORR)
    eta   = U_eq - U_L

    # plot -------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        labels = [
            "O$_2$ + 2H$_2$", "OOH* + 1.5H$_2$", "O* + H$_2$O + H$_2$",
            "OH* + H$_2$O + 0.5H$_2$", "* + 2H$_2$O",
        ]
        steps = np.arange(rxn_num + 1)
        G0_shift = G_profile_U0 - G_profile_U0[-1]
        Geq_shift = G_profile_Ueq - G_profile_Ueq[-1]

        plt.figure(figsize=(7, 6))
        plt.plot(steps, G0_shift, "o-", label="U = 0 V", markersize=6)
        plt.plot(steps, Geq_shift, "o-", label=f"U = {U_eq} V", markersize=6)
        rds = np.argmax(diffG_eq)
        plt.plot(
            [rds, rds + 1],
            [Geq_shift[rds], Geq_shift[rds + 1]],
            "r-",
            linewidth=2.5,
            label=f"RDS (η = {eta:.2f} V)",
        )
        plt.xticks(steps, labels, rotation=15)
        plt.ylabel("ΔG (eV, relative)")
        plt.title("4e⁻ ORR free‑energy diagram")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        fig_path = output_dir / "ORR_free_energy_diagram.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info("Saved diagram → %s", fig_path)
    except Exception as exc:
        logger.warning("Plotting failed: %s", exc)

    if verbose:
        logger.info("ΔG (U=0) = %s", deltaG_U0)
        logger.info("Limiting potential U_L = %.3f V", U_L)
        logger.info("Overpotential η = %.3f V", eta)

    return eta

def calc_orr_overpotential(
    bulk: Atoms,
    base_dir: str = "result/matter_sim",
    force: bool = False,
    log_level: str = "INFO",
    calc_type: str = "mattersim",
) -> float:
    """
    Entry point : bulk → (slab, ads) → ΔE → η
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    # 1. bulk optimisation
    logger.info("Optimising bulk …")
    opt_bulk, E_bulk = optimize_bulk(bulk, str(base_path / "bulk"), calc_type)

    # 2. clean-slab optimisation
    logger.info("Optimising clean slab …")
    opt_slab, E_slab = optimize_slab(opt_bulk, str(base_path / "slab"), calc_type)

    # 3. gas + adsorption calculations (offset scheme)
    logger.info("Running required molecule calculations …")
    results = calculate_required_molecules(
        opt_slab, E_slab, base_path,
        force=force, calc_type=calc_type
    )

    # 4. ΔE & over-potential
    deltaEs, energies = compute_reaction_energies(results, E_slab)
    eta = get_overpotential_orr(deltaEs, base_path, verbose=True)

    # 5. summary
    with (base_path / "ORR_summary.txt").open("w") as f:
        f.write("--- ORR Summary ---\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in deltaEs) + "\n")
        f.write(f"Overpotential η = {eta:.3f} V\n")
    logger.info("Summary written → %s", base_path / "ORR_summary.txt")

    return eta

# ---------------------------------------------------------------------------
# main entry -----------------------------------------------------------------
# ---------------------------------------------------------------------------


# main関数を修正してcalc_orr_overpotential関数を活用する
def main():
    p = argparse.ArgumentParser(description="ORR workflow (offset adsorption)")
    p.add_argument("--base-dir", default="result/matter_sim")
    p.add_argument("--force",  action="store_true")
    p.add_argument("--log",    default="INFO")
    p.add_argument("--calc-type", default="mattersim")
    args = p.parse_args()

    bulk = fcc111("Pt", size=(5, 5, 4), a=4.0, vacuum=None, periodic=True)
    eta = calc_orr_overpotential(
        bulk, args.base_dir, args.force, args.log, args.calc_type
    )
    print(f"η_ORR = {eta:.3f} V")
    return eta


if __name__ == "__main__":
    main()