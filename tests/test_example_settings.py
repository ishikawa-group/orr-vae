from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_example_structure_counts():
    root = Path(__file__).resolve().parents[1]

    ptni = _load_module(root / "examples" / "Pt-Ni" / "code" / "run_workflow.py", "ptni_workflow")
    ptni_args = ptni.build_parser().parse_args([])
    assert ptni_args.initial_num_structures == 128
    assert ptni_args.generated_num_structures == 128

    mixed = _load_module(
        root / "examples" / "Pt-Ni_Pt-Ti_Pt-Y" / "code" / "run_workflow.py",
        "mixed_workflow",
    )
    mixed_args = mixed.build_parser().parse_args([])
    assert mixed_args.initial_num_structures == 255
    assert mixed_args.generated_num_structures == 255
