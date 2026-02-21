#!/usr/bin/env python3
"""
Utility helpers for ORR overpotential calculations.

The script provides three entry points in a single file:
    - ``single``: evaluate one structure.
    - ``run-uncalculated``: process the next structure that has not been evaluated yet.
    - ``run-all``: iterate ``run-uncalculated`` until all structures are processed.
"""
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ase.db import connect
from tqdm.auto import tqdm

try:
    from surface.orr_overpotential_calculator import calc_orr_overpotential
except ImportError:  # pragma: no cover - compatibility fallback
    try:
        from orr_overpotential_calculator import calc_orr_overpotential
    except ImportError:
        from orr_overpotential_calculator.calc_orr_overpotential import calc_orr_overpotential

from orr_vae.tool import ALLOY_ELEMENTS, calc_alloy_formation_energy


DEFAULT_SOLVENT_PATH = Path(__file__).parent / "solvent_correction.yaml"

# ``ase.io.extxyz`` emits this warning when ``atoms.info`` contains nested dicts
# such as ``adsorbate_info``. It is harmless in this workflow and clutters logs.
warnings.filterwarnings(
    "ignore",
    message=r"Skipping unhashable information adsorbate_info",
    category=UserWarning,
    module=r"ase\.io\.extxyz",
)


def _load_results(out_json: Path) -> List[Dict[str, Any]]:
    if out_json.exists() and out_json.stat().st_size > 0:
        with open(out_json) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    return []


def calculate_single(
    bulk_db: Path,
    out_json: Path,
    unique_id: str,
    outdir: Path,
    *,
    overwrite: bool = False,
    log_level: str = "INFO",
    calculator: str = "fairchem",
    solvent_correction_yaml_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate ORR overpotential for a single structure and append the result to ``out_json``."""
    if solvent_correction_yaml_path is None:
        solvent_correction_yaml_path = DEFAULT_SOLVENT_PATH

    if not bulk_db.exists():
        raise FileNotFoundError(f"Structure database not found: {bulk_db}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    results = _load_results(out_json)
    ids = [entry.get("unique_id") for entry in results]

    if unique_id not in ids:
        results.append({"unique_id": unique_id, "overpotential": None})
        ids.append(unique_id)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, separators=(",", ": "))
        print(f"Added placeholder entry for unique_id={unique_id}")

    db = connect(str(bulk_db))
    row = None
    search_methods = (
        lambda: list(db.select(f'unique_id="{unique_id}"')),
        lambda: list(db.select(unique_id=unique_id)),
        lambda: [
            r
            for r in db.select()
            if "unique_id" in r.key_value_pairs and r.key_value_pairs["unique_id"] == unique_id
        ],
        lambda: [db.get(int(unique_id))],
    )
    for method in search_methods:
        try:
            rows = method()
            if rows:
                row = rows[0]
                break
        except (ValueError, KeyError):
            continue

    if row is None:
        raise ValueError(f"unique_id '{unique_id}' was not found in the database")

    bulk_atoms = row.toatoms(add_additional_information=True)
    info = bulk_atoms.info.pop("data", {})
    bulk_atoms.info["adsorbate_info"] = info.get("adsorbate_info", {})

    calc_dir = outdir / unique_id
    calc_dir.mkdir(parents=True, exist_ok=True)

    symbols = bulk_atoms.get_chemical_symbols()
    total_atoms = len(symbols)
    raw_counts = Counter(symbols)
    element_counts = {element: int(raw_counts.get(element, 0)) for element in ALLOY_ELEMENTS}
    if total_atoms == 0:
        composition = {element: 0.0 for element in ALLOY_ELEMENTS}
    else:
        composition = {element: element_counts[element] / total_atoms for element in ALLOY_ELEMENTS}

    orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
        "HO2": [(1.0, 1.0), (1.5, 1.0), (1.33, 1.33), (1.66, 1.66)],
        "O": [(1.0, 1.0), (1.5, 1.0), (1.33, 1.33), (1.66, 1.66)],
        "OH": [(1.0, 1.0), (1.5, 1.0), (1.33, 1.33), (1.66, 1.66)],
    }

    print(f"Starting ORR calculation for unique_id={unique_id} ...")
    result = calc_orr_overpotential(
        bulk=bulk_atoms,
        outdir=str(calc_dir),
        overwrite=overwrite,
        log_level=log_level,
        calculator=calculator,
        adsorbates=orr_adsorbates,
        solvent_correction_yaml_path=solvent_correction_yaml_path,
    )

    eta = result["eta"]
    limiting_potential = 1.23 - eta
    diffg_u0 = result["diffG_U0"]
    diffg_eq = result["diffG_eq"]
    e_bulk_alloy = result.get("E_bulk")

    opt_bulk_property = None
    opt_bulk_path = calc_dir / "bulk" / "optimized_bulk.extxyz"
    if opt_bulk_path.exists():
        from ase.io import read

        try:
            opt_bulk = read(str(opt_bulk_path))
            opt_bulk_property = {
                "positions": opt_bulk.get_positions().tolist(),
                "cell": opt_bulk.get_cell().array.tolist(),
                "chemical_symbols": opt_bulk.get_chemical_symbols(),
                "pbc": opt_bulk.get_pbc().tolist(),
                "atomic_numbers": opt_bulk.get_atomic_numbers().tolist(),
            }
            print(f"Loaded optimized bulk structure: {opt_bulk_path}")
        except Exception as exc:  # pragma: no cover - informational warning only
            print(f"Warning: failed to load optimized bulk structure ({exc})")

    e_alloy_formation = None
    if e_bulk_alloy is not None:
        try:
            e_alloy_formation = calc_alloy_formation_energy(
                bulk_atoms,
                e_bulk_alloy,
                calculator=calculator,
                cache_dir=bulk_db.parent,
            )
        except Exception as exc:  # pragma: no cover - informational warning only
            print(f"Warning: alloy formation energy calculation failed: {exc}")

    entry = {
        "unique_id": unique_id,
        "overpotential": eta,
        "limiting_potential": limiting_potential,
        "diffG_U0": diffg_u0,
        "diffG_eq": diffg_eq,
        "composition": {element: float(composition[element]) for element in ALLOY_ELEMENTS},
        "element_counts": element_counts,
        "chemical_formula": bulk_atoms.get_chemical_formula(),
        "E_bulk_alloy": e_bulk_alloy,
        "E_alloy_formation": e_alloy_formation,
        "opt_bulk_property": opt_bulk_property,
    }
    for element in ALLOY_ELEMENTS:
        entry[f"{element.lower()}_fraction"] = float(composition[element])

    # Keep legacy keys for downstream compatibility.
    entry["ni_fraction"] = float(composition.get("Ni", 0.0))
    entry["pt_fraction"] = float(composition.get("Pt", 0.0))

    replaced = False
    for idx, existing in enumerate(results):
        if existing.get("unique_id") == unique_id:
            results[idx] = entry
            replaced = True
            break
    if not replaced:
        results.append(entry)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, separators=(",", ": "))

    print(f"ORR overpotential: {eta:.3f} V")
    print(f"ΔG(U=0 V): {diffg_u0}")
    print(f"ΔG(U=1.23 V): {diffg_eq}")
    return entry


def run_uncalculated(
    *,
    iter_idx: int,
    base_data_dir: Path,
    temp_base_dir: Path,
    calculator: str,
    log_level: str,
    overwrite: bool,
    keep_temp: bool,
    solvent_correction_yaml_path: Optional[Path] = None,
) -> bool:
    """Process a single uncalculated structure. Returns True if a calculation was performed."""
    bulk_file = base_data_dir / f"iter{iter_idx}_structures.json"
    result_file = base_data_dir / f"iter{iter_idx}_calculation_result.json"

    print(f"=== ORR calculation for pending structures (iter{iter_idx}) ===")
    print(f"Structure DB: {bulk_file}")
    print(f"Result JSON: {result_file}")

    if not bulk_file.exists():
        raise FileNotFoundError(f"Structure file not found: {bulk_file}")

    result_file.parent.mkdir(parents=True, exist_ok=True)
    if not result_file.exists() or result_file.stat().st_size == 0:
        with open(result_file, "w") as f:
            json.dump([], f)

    db = connect(str(bulk_file))
    bulk_ids = []
    for row in db.select():
        if getattr(row, "unique_id", None) is not None:
            bulk_ids.append(str(row.unique_id))
        else:
            bulk_ids.append(str(row.id))

    with open(result_file) as f:
        reaction_data = json.load(f)
    reaction_ids = [str(entry.get("unique_id", entry.get("id"))) for entry in reaction_data]

    uncalculated = [uid for uid in bulk_ids if uid not in reaction_ids]
    if not uncalculated:
        print("All structures are already processed.")
        return False

    uid = uncalculated[0]
    print(f"Remaining structures: {len(uncalculated)}")
    print(f"Next structure to process: unique_id={uid}")

    temp_dir = temp_base_dir / f"iter{iter_idx}_{uid}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {temp_dir}")

    try:
        calculate_single(
            bulk_db=bulk_file,
            out_json=result_file,
            unique_id=uid,
            outdir=temp_dir,
            overwrite=overwrite,
            log_level=log_level,
            calculator=calculator,
            solvent_correction_yaml_path=solvent_correction_yaml_path,
        )
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Temporary directory removed: {temp_dir}")
        else:
            print(f"Temporary directory kept: {temp_dir}")

    with open(result_file) as f:
        updated = json.load(f)

    print(f"Completed calculations: {len(updated)}/{len(bulk_ids)}")
    remaining = len(bulk_ids) - len(updated)
    print(f"Remaining structures: {remaining}")
    return True


def run_all(
    *,
    iter_idx: int,
    base_dir: Path,
    base_data_dir: Optional[Path],
    temp_base_dir: Optional[Path],
    calculator: str,
    log_level: str,
    overwrite: bool,
    keep_temp: bool,
    max_count: int,
    wait_time: int,
    solvent_correction_yaml_path: Optional[Path] = None,
) -> None:
    """Iteratively call ``run_uncalculated`` until no structures remain."""
    if base_data_dir is None:
        base_data_dir = base_dir / "data"
    if temp_base_dir is None:
        temp_base_dir = base_dir / "result" / "test"

    print(f"=== Batch ORR calculation (iter{iter_idx}) ===")
    print(f"Data directory: {base_data_dir}")
    print(f"Temporary root: {temp_base_dir}")
    print(f"Maximum attempts: {max_count}")
    print(f"Wait time between attempts: {wait_time} s")
    print(f"Calculator: {calculator}")
    print(f"Log level: {log_level}")
    print(f"Force overwrite: {overwrite}")
    print(f"Keep temporary outputs: {keep_temp}")

    success_count = 0
    error_count = 0

    progress = tqdm(
        range(1, max_count + 1),
        desc=f"iter{iter_idx} ORR",
        unit="attempt",
        dynamic_ncols=True,
    )
    for attempt in progress:
        print("-------------------------------------")
        print(f"Attempt {attempt}/{max_count} (success: {success_count}, error: {error_count})")
        try:
            did_work = run_uncalculated(
                iter_idx=iter_idx,
                base_data_dir=base_data_dir,
                temp_base_dir=temp_base_dir,
                calculator=calculator,
                log_level=log_level,
                overwrite=overwrite,
                keep_temp=keep_temp,
                solvent_correction_yaml_path=solvent_correction_yaml_path,
            )
        except Exception as exc:
            error_count += 1
            progress.set_postfix(success=success_count, error=error_count, status="error")
            print(f"Error encountered: {exc}")
            if attempt < max_count:
                print(f"Retrying in {wait_time} s ...")
                time.sleep(wait_time)
            continue

        if not did_work:
            progress.set_postfix(success=success_count, error=error_count, status="done")
            print("No remaining structures to process. Exiting loop.")
            break

        success_count += 1
        progress.set_postfix(success=success_count, error=error_count, status="worked")
        if attempt < max_count:
            print(f"Waiting {wait_time} s before the next attempt ...")
            time.sleep(wait_time)

    print("-------------------------------------")
    print(f"=== Batch calculation finished (iter{iter_idx}) ===")
    print(f"Success count: {success_count}")
    print(f"Error count: {error_count}")
    total = success_count + error_count
    if total:
        print(f"Success rate: {success_count / total * 100:.1f}%")


def parse_args(argv: Optional[List[str]] = None) -> Tuple[str, argparse.Namespace]:
    if argv is None:
        argv = sys.argv[1:]

    commands = {"single", "run-uncalculated", "run-all"}
    if argv and argv[0] in commands:
        command = argv[0]
        command_args = argv[1:]
    else:
        command = "single"
        command_args = argv

    if command == "single":
        parser = argparse.ArgumentParser(description="Compute ORR overpotential for a single structure")
        parser.add_argument("--bulk_db", required=True, help="Bulk structure database in ASE JSON format")
        parser.add_argument("--out_json", required=True, help="Output JSON file for accumulating results")
        parser.add_argument("--unique_id", required=True, help="unique_id of the target structure")
        parser.add_argument("--outdir", default="./result", help="Working directory for intermediate files")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Re-run even if the target already has stored results",
        )
        parser.add_argument("--log_level", default="INFO", help="Log level (default: INFO)")
        parser.add_argument(
            "--calculator",
            default="fairchem",
            choices=["mace", "vasp", "emt", "fairchem", "orb-v3", "7net"],
            help="Calculator backend (default: fairchem)",
        )
        parser.add_argument(
            "--solvent_correction_yaml_path",
            type=Path,
            default=DEFAULT_SOLVENT_PATH,
            help="Path to solvent correction YAML file",
        )
        args = parser.parse_args(command_args)
        return command, args

    if command == "run-uncalculated":
        parser = argparse.ArgumentParser(description="Process the next uncalculated structure")
        parser.add_argument("--iter", type=int, default=0, help="Iteration index (default: 0)")
        parser.add_argument(
            "--base_data_dir",
            type=Path,
            default=Path(__file__).parent / "data",
            help="Directory containing structure JSON files",
        )
        parser.add_argument(
            "--temp_base_dir",
            type=Path,
            default=Path(__file__).parent / "result" / "test",
            help="Base directory for temporary calculation artefacts",
        )
        parser.add_argument(
            "--calculator",
            default="fairchem",
            choices=["mace", "vasp", "emt", "fairchem", "orb-v3", "7net"],
            help="Calculator backend (default: fairchem)",
        )
        parser.add_argument("--log_level", default="INFO", help="Log level (default: INFO)")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Re-run even when results exist",
        )
        parser.add_argument(
            "--keep_temp",
            action="store_true",
            help="Do not remove the temporary directory after completion",
        )
        parser.add_argument(
            "--solvent_correction_yaml_path",
            type=Path,
            default=DEFAULT_SOLVENT_PATH,
            help="Path to solvent correction YAML file",
        )
        args = parser.parse_args(command_args)
        return command, args

    parser = argparse.ArgumentParser(description="Loop ORR calculations until all structures are processed")
    parser.add_argument("--iter", type=int, default=0, help="Iteration index (default: 0)")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).parent,
        help="Base directory containing data and result folders",
    )
    parser.add_argument(
        "--base_data_dir",
        type=Path,
        default=None,
        help="Optional override for the structure directory (default: base_dir/data)",
    )
    parser.add_argument(
        "--temp_base_dir",
        type=Path,
        default=None,
        help="Optional override for the temporary directory root (default: base_dir/result/test)",
    )
    parser.add_argument("--max_count", type=int, default=256, help="Maximum number of attempts")
    parser.add_argument("--wait_time", type=int, default=2, help="Wait time between attempts in seconds")
    parser.add_argument(
        "--calculator",
        default="fairchem",
        choices=["mace", "vasp", "emt", "fairchem", "orb-v3", "7net"],
        help="Calculator backend (default: fairchem)",
    )
    parser.add_argument("--log_level", default="INFO", help="Log level (default: INFO)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even when results exist",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Do not remove temporary directories after each run",
    )
    parser.add_argument(
        "--solvent_correction_yaml_path",
        type=Path,
        default=DEFAULT_SOLVENT_PATH,
        help="Path to solvent correction YAML file",
    )
    args = parser.parse_args(command_args)
    return command, args


def main(argv: Optional[List[str]] = None) -> int:
    command, args = parse_args(argv)

    try:
        if command == "single":
            calculate_single(
                bulk_db=Path(args.bulk_db),
                out_json=Path(args.out_json),
                unique_id=str(args.unique_id),
                outdir=Path(args.outdir),
                overwrite=args.overwrite,
                log_level=args.log_level,
                calculator=args.calculator,
                solvent_correction_yaml_path=args.solvent_correction_yaml_path,
            )
            return 0

        if command == "run-uncalculated":
            did_work = run_uncalculated(
                iter_idx=args.iter,
                base_data_dir=args.base_data_dir,
                temp_base_dir=args.temp_base_dir,
                calculator=args.calculator,
                log_level=args.log_level,
                overwrite=args.overwrite,
                keep_temp=args.keep_temp,
                solvent_correction_yaml_path=args.solvent_correction_yaml_path,
            )
            return 0 if did_work else 2

        run_all(
            iter_idx=args.iter,
            base_dir=args.base_dir,
            base_data_dir=args.base_data_dir,
            temp_base_dir=args.temp_base_dir,
            calculator=args.calculator,
            log_level=args.log_level,
            overwrite=args.overwrite,
            keep_temp=args.keep_temp,
            max_count=args.max_count,
            wait_time=args.wait_time,
            solvent_correction_yaml_path=args.solvent_correction_yaml_path,
        )
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
