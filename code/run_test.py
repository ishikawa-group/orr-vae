#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# ASEのインポート
from ase.build import fcc111, fcc100

# 自作モジュールからORR過電圧計算関数をインポート
sys.path.append(str(Path(__file__).parent))
from calc_orr_overpotential import calc_orr_overpotential

#---------------------
# 引数の設定
base_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/RPBE/Pt111"
force = True
log_level = "INFO"
calc_type = "mattersim"
#----------------

bulk = fcc100("Pt", size=(5, 5, 4), a=4.0, vacuum=None, periodic=True)

eta = calc_orr_overpotential(
    bulk=bulk,
    base_dir=base_dir,
    force=force,
    log_level=log_level,
    calc_type=calc_type
)

print(f"ORR overpotential: {eta:.3f} V")
