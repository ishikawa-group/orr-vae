#!/usr/bin/env python3
"""
Ni–Pd 64-site ORR overpotential minimisation
via Gaussian-Process Bayesian optimisation (skopt).

・試行回数: 640 (= 128 * 5) 
・探索履歴:  result/bo_history.csv
・最終結果:  result/best_result.json
"""

# --- 0. 依存ライブラリ ----------------------
import json
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.build import fcc111
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver   # 任意
from skopt.plots import plot_convergence      # 任意

from orr_overpotential_calculator import calc_orr_overpotential  # ユーザ実装

# --- 1. 定数・パス --------------------------
N_SITES       = 64              # 4×4×4 fcc111 slab の表面サイト数
N_CALLS       = 128 * 5              
INIT_POINTS   = 128
BASE_DIR      = Path(__file__).parent.parent
OUTDIR_ROOT   = BASE_DIR / "result"
FIGURE_DIR   = OUTDIR_ROOT / "figures"
OUTDIR_ROOT.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

CSV_PATH      = FIGURE_DIR / "bo_history.csv"
JSON_PATH     = FIGURE_DIR / "best_result.json"
CKPT_PATH     = FIGURE_DIR / "checkpoint.pkl"

# Pd 基板を生成（コピーして使い回す）
bulk_template = fcc111("Pd", size=(4, 4, 4), a=3.9, vacuum=None, periodic=True)

# 検索空間: 0→Pd, 1→Ni
search_space  = [Integer(0, 1, name=f"site{i}") for i in range(N_SITES)]

# --- 2. CSV ロガー --------------------------
class CSVLogger:
    """skopt OptimizeResult を受け取り CSV を追記保存するコールバック"""
    def __init__(self, path: Path):
        self.path = path
        # 既存ファイル有無でヘッダ出力を切り替え
        self.header_written = self.path.exists()

    def __call__(self, res):
        iter_idx = len(res.func_vals) - 1       # 現イテレーション番号
        eta      = res.func_vals[-1]            # 過電圧
        comp     = res.x_iters[-1]              # 0/1 配列 (長さ 64)

        with self.path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not self.header_written:
                header = (["iteration", "eta"]
                          + [f"site{i}" for i in range(N_SITES)])
                writer.writerow(header)
                self.header_written = True
            writer.writerow([iter_idx, eta, *comp])

# --- 3. 目的関数 ----------------------------
@use_named_args(search_space)
def objective(**kwargs):
    """
    kwargs == {"site0":0/1, ..., "site63":0/1}
    """
    bulk = deepcopy(bulk_template)
    for idx, val in enumerate(kwargs.values()):
        bulk[idx].symbol = 'Ni' if val == 1 else 'Pd'

    res = calc_orr_overpotential(
        bulk=bulk,
        outdir=str(OUTDIR_ROOT / f"iter{objective.calls}"),
        force=True,
        log_level="INFO",
        calc_type="mace",
        adsorbates={
            "HO2": [(0, 0), (0.5, 0), (0.33, 0.33), (0.66, 0.66)],
            "O"  : [(0, 0), (0.5, 0), (0.33, 0.33), (0.66, 0.66)],
            "OH" : [(0, 0), (0.5, 0), (0.33, 0.33), (0.66, 0.66)],
        },
        yaml_path=str(BASE_DIR / "vasp.yaml"),
    )
    eta = res["eta"]
    objective.calls += 1
    return eta

objective.calls = 0  # カウンタ初期化

# --- 4. コールバック設定 --------------------
csv_logger = CSVLogger(CSV_PATH)
ckpt_saver = CheckpointSaver(str(CKPT_PATH), compress=9)   # Pickle 保存（任意）

callbacks  = [csv_logger, ckpt_saver]

# --- 5. 最適化実行 --------------------------
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=N_CALLS,
    n_initial_points=INIT_POINTS,
    acq_func="EI",
    acq_optimizer="sampling",
    noise=1e-6,
    random_state=42,
    callback=callbacks,
    verbose=True,
)

# --- 6. 最終結果を JSON 保存 -----------------
best_data = {
    "best_eta" : float(result.fun),
    "best_comp": [int(x) for x in result.x],      # list 0/1 × 64
}
with JSON_PATH.open("w") as f:
    json.dump(best_data, f, indent=2)

# --- 7. コンソール出力 ----------------------
print(f"\n=== 最小過電圧: {result.fun:.3f} V ===")
print("最適組成 (0=Pd, 1=Ni) → 表面層 4×4:")
print(np.array(result.x).reshape(4, 4, 4)[0])

# --- 8. 収束プロット (任意) -----------------
try:
    import matplotlib.pyplot as plt
    plot_convergence(result)
    plt.savefig(FIGURE_DIR / "convergence.png", dpi=150, bbox_inches="tight")
except Exception as e:
    print(f"[WARN] 収束プロット生成に失敗: {e}")
