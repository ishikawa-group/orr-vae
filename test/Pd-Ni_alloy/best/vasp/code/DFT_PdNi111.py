#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASEのインポート
from ase.build import fcc111
import numpy as np

# ORR過電圧計算関数をインポート
from orr_overpotential_calculator import calc_orr_overpotential

#---------------------
# 引数の設定
outdir = str(Path(__file__).parent.parent / "result/PdNi111/vasp")
force = True
log_level = "INFO"
calc_type = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

# まずNiのfcc111表面を作成
bulk = fcc111("Ni", size=(4, 4, 4), a=3.7, vacuum=None, periodic=True)

# 指定された原子番号の順番に基づいて合金を作成
# 28: Ni, 46: Pd
atomic_numbers = [28, 46, 28, 28, 28, 46, 28, 28, 28, 46, 28, 46, 28, 28, 46, 28, 
                  28, 28, 46, 28, 28, 28, 46, 46, 28, 28, 28, 46, 46, 28, 28, 28, 
                  28, 46, 46, 28, 28, 46, 28, 28, 28, 46, 28, 46, 46, 28, 28, 28, 
                  46, 46, 46, 28, 46, 46, 28, 28, 28, 28, 28, 28, 28, 46, 28, 28]

# 原子番号に基づいて元素を設定
for i, atomic_number in enumerate(atomic_numbers):
    if atomic_number == 28:
        bulk[i].symbol = 'Ni'
    elif atomic_number == 46:
        bulk[i].symbol = 'Pd'

orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.5, 0.0), (0.33, 0.33), (0.66, 0.66)], #bridge, fcc, hcp
    "O":   [(0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    "OH":  [(0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
}

# 関数呼び出しの変更：辞書として結果を受け取る
result = calc_orr_overpotential(
    bulk=bulk,
    outdir=outdir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    adsorbates=orr_adsorbates,
    yaml_path=yaml_path
)

# 必要な値を辞書から取得
eta = result["eta"]
diffG_U0 = result["diffG_U0"]
diffG_eq = result["diffG_eq"]

print(f"ORR overpotential: {eta:.3f} V")
print(f"Reaction Free Energy Change at U=0V: {diffG_U0}")
print(f"Reaction Free Energy Change at U=1.23V: {diffG_eq}")