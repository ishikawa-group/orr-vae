#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified workflow to (1) obtain ORR‑related adsorption energies
and (2) evaluate the 4‑electron ORR overpotential (η).

Steps
-----
1. Pt bulk optimisation → slab construction/optimisation.
2. Gas‑phase optimisation of O‑, H‑containing molecules.
3. Adsorption‑site scan (ontop/bridge/fcc/hcp); stores the most stable site.
4. Reaction‑energy (ΔE) evaluation for the 4e⁻ pathway.
5. Free‑energy diagram + overpotential following the Nørskov scheme.

Run
---
$ python orr_workflow.py  --base-dir result/matter_sim  --force   # re‑run everything
$ python orr_workflow.py                                          # skip finished jobs

The script is intentionally kept as a *single* entry point so that the whole
analysis is reproducible with one command.  Heavy DFT jobs (VASP) are handled
by helper functions in ``calc_orr_energy.py``; those functions must already be
working in your environment.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from ase import Atoms
from ase.build import fcc111

# ---  external helper -------------------------------------------------------
from calc_orr_energy import (
    optimize_bulk,
    optimize_slab,
    optimize_gas,
    calc_adsorption_on_site,
)

from tool import convert_numpy_types

# ---------------------------------------------------------------------------
# Global settings (edit if needed)
# ---------------------------------------------------------------------------
SITES: List[str] = ["ontop", "bridge", "fcc", "hcp"]

MOLECULES: Dict[str, Atoms] = {
    "OH":  Atoms("OH",  positions=[(0, 0, 0), (0, 0, 0.97)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, 0, 1.46), (0.939, 0, 1.705)]),
    "H2":  Atoms("HH",  positions=[(0, 0, 0), (0, 0, 0.74)]),
    "O2":  Atoms("OO",  positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H":   Atoms("H" ,  positions=[(0, 0, 0)]),
    "O":   Atoms("O" ,  positions=[(0, 0, 0)]),
}

SLAB_VACUUM = 30.0  # Å
GAS_BOX     = 15.0  # Å
ADS_HEIGHT  = 2.0   # Å

logger = logging.getLogger("orr_workflow")

# ---------------------------------------------------------------------------
# Adsorption‑energy part
# ---------------------------------------------------------------------------

def calculate_all_molecules(
    opt_slab: Atoms,
    E_opt_slab: float,
    base_dir: Path,
    force: bool = False,
    calc_type: str = "mattersim",
) -> Dict[str, Any]:
    """Optimise all molecules + adsorption on *opt_slab*.

    Returns a nested dict keyed by molecule name.
    The sub‑dict contains *E_gas*, *E_total_best* (slab+adsorbate),
    *best_site*, and *E_ads* (=E_total_best−E_slab−E_gas).
    """
    results: Dict[str, Any] = {}
    base_dir.mkdir(parents=True, exist_ok=True)

    for mol_name, mol in MOLECULES.items():
        logger.info("=== %s ===", mol_name)
        mol_dir   = base_dir / mol_name
        gas_dir   = mol_dir / f"{mol_name}_gas"
        ads_dir   = mol_dir / "adsorption"
        gas_dir.mkdir(parents=True, exist_ok=True)
        ads_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1) gas optimisation ------------------------------------------------
            gas_json = gas_dir / "opt_result.json"
            if gas_json.exists() and not force:
                with gas_json.open() as f:
                    gas_data = json.load(f)
                opt_mol_energy = gas_data["E_opt"]
                opt_mol = mol  # geometry not required afterwards
                logger.info("  Re‑using existing gas optimisation result.")
            else:
                opt_mol, opt_mol_energy = optimize_gas(mol_name, GAS_BOX, str(gas_dir), calc_type)
                with gas_json.open("w") as f:
                    json.dump({"E_opt": float(opt_mol_energy)}, f)

            # 2) adsorption on each site --------------------------------------
            sites_data: Dict[str, Dict[str, float]] = {}
            for site in SITES:
                site_key = f"{site}.json"
                site_json = ads_dir / site_key
                if site_json.exists() and not force:
                    with site_json.open() as f:
                        data = json.load(f)
                    E_total = data["E_total"]
                    elapsed = data.get("elapsed", 0.0)
                else:
                    E_total, elapsed = calc_adsorption_on_site(opt_slab, opt_mol, site, str(ads_dir), calc_type)
                    with site_json.open("w") as f:
                        json.dump({"E_total": float(E_total), "elapsed": elapsed}, f)
                sites_data[site] = {"E_total": E_total, "elapsed": elapsed}

            # 3) pick the most stable site -----------------------------------
            best_site, E_total_best = min(
                ((s, d["E_total"]) for s, d in sites_data.items()),
                key=lambda x: x[1],
            )
            E_ads_best = E_total_best - (E_opt_slab + opt_mol_energy)

            results[mol_name] = {
                "E_gas":            float(opt_mol_energy),
                "E_slab":           float(E_opt_slab),
                "E_total_best":     float(E_total_best),
                "best_site":        best_site,
                "E_ads_best":       float(E_ads_best),
                "sites":            sites_data,
            }
            logger.info("  -> best site: %s  E_ads = %.3f eV", best_site, E_ads_best)
        except Exception as exc:
            logger.exception("  Failed for %s: %s", mol_name, exc)
            results[mol_name] = {"error": str(exc)}

    # write summary -----------------------------------------------------------
    with (base_dir / "all_results.json").open("w") as f:
        results = convert_numpy_types(results)
        json.dump(results, f, indent=2)
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
    base_dir: str = "result/something",
    force: bool = False,
    log_level: str = "INFO",
    calc_type: str = "mattersim"
) -> float:
    """
    ORR過電圧計算を実行するメイン関数
    
    Parameters:
    -----------
    bulk : Atoms
        バルク構造
    base_dir : str
        計算結果を保存するディレクトリ
    force : bool
        Trueの場合、既存の計算結果を上書きする
    log_level : str
        ログレベル（"DEBUG"/"INFO"/"WARNING"/"ERROR"）
    calc_type : str
        計算タイプ（"mattersim"/"vasp"など）
        
    Returns:
    --------
    float
        計算されたORR過電圧（η）
    """
    # ロギングの設定
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    
    # ディレクトリのパス設定
    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 1. バルク最適化
    logger.info("Optimising bulk …")
    opt_bulk, E_bulk = optimize_bulk(bulk, str(base_path / "bulk"), calc_type)
    
    # 2. スラブ最適化
    logger.info("Optimising slab …")
    opt_slab, E_slab = optimize_slab(opt_bulk, str(base_path / "slab"), calc_type)
    
    # 3. 分子吸着計算
    logger.info("Running adsorption scan for all molecules …")
    results = calculate_all_molecules(opt_slab, E_slab, base_path, force=force, calc_type=calc_type)
    
    # 4. 反応エネルギー計算
    deltaEs, energies = compute_reaction_energies(results, E_slab)
    logger.info("ΔE values: %s", [f"{e:+.3f}" for e in deltaEs])
    
    # 5. 過電圧計算
    logger.info("Evaluating ORR overpotential …")
    eta = get_overpotential_orr(deltaEs, base_path, verbose=True)
    logger.info("==> η_ORR = %.3f V ", eta)
    
    # 6. サマリー出力
    with (base_path / "ORR_summary.txt").open("w") as f:
        f.write("--- ORR Summary ---\n\n")
        energies = convert_numpy_types(energies)
        f.write(json.dumps(energies, indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in deltaEs) + "\n")
        f.write(f"Overpotential η = {eta:.3f} V\n")
    
    logger.info("Summary written → %s", base_path / "ORR_summary.txt")
    
    return eta

# ---------------------------------------------------------------------------
# main entry -----------------------------------------------------------------
# ---------------------------------------------------------------------------


# main関数を修正してcalc_orr_overpotential関数を活用する
def main():
    p = argparse.ArgumentParser(description="ORR adsorption + overpotential workflow")
    p.add_argument("--base-dir", default="result/matter_sim", help="top directory for calculations")
    p.add_argument("--force", action="store_true", help="re‑run even if results exist")
    p.add_argument("--log", default="INFO", help="logging level (DEBUG/INFO/WARNING/ERROR)")
    p.add_argument("--calc-type", default="mattersim", help="calculation type (mattersim/vasp/etc)")
    args = p.parse_args()
    
    # バルク構造の作成
    bulk = fcc111("Pt", size=(3, 3, 4), a=4.0, vacuum=None, periodic=True)
    
    # 関数を呼び出して計算を実行
    eta = calc_orr_overpotential(
        bulk=bulk,
        base_dir=args.base_dir,
        force=args.force,
        log_level=args.log,
        calc_type=args.calc_type
    )
    
    return eta