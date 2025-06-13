#!/usr/bin/env python
"""
学習済みVAEから新しい触媒構造を生成するスクリプト
"""
import os
import argparse
import torch
import numpy as np
import importlib.util
import sys
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.db import connect
from ase.data import atomic_numbers
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from tool import vegard_lattice_constant, tensor_to_structure, sort_atoms

import torch.nn.functional as F

# グローバル変数でiter番号を設定
ITER = 2  # ここで現在のiter番号を設定

def load_vae_class():
    """03_conditional_vae.pyからConditionalVAEクラスを動的にインポート"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vae_script_path = os.path.join(current_dir, "03_conditional_vae.py")
    
    if not os.path.exists(vae_script_path):
        raise FileNotFoundError(f"VAEスクリプトが見つかりません: {vae_script_path}")
    
    print(f"VAEスクリプトを読み込み中: {vae_script_path}")
    
    spec = importlib.util.spec_from_file_location("conditional_vae", vae_script_path)
    conditional_vae_module = importlib.util.module_from_spec(spec)
    sys.modules["conditional_vae"] = conditional_vae_module
    spec.loader.exec_module(conditional_vae_module)
    return conditional_vae_module.ConditionalVAE

def convert_tensor_to_atomic_numbers(tensor):
    """
    生成されたテンソル (12, 8, 8) を原子番号に変換
    12チャンネル = 4層 × 3クラス（空/Ni/Pt）
    """
    discrete_tensor = torch.zeros(4, 8, 8, dtype=torch.long)
    
    for layer in range(4):
        # 各層の3チャンネル（3クラス分類）
        layer_logits = tensor[layer*3:(layer+1)*3]  # [3, 8, 8]
        # ソフトマックス適用後、最大値のクラスを選択
        layer_probs = F.softmax(layer_logits, dim=0)
        layer_discrete = torch.argmax(layer_probs, dim=0)  # [8, 8]
        discrete_tensor[layer] = layer_discrete
    
    # 原子番号へのマッピング
    atomic_numbers_tensor = torch.zeros_like(discrete_tensor, dtype=torch.int64)
    atomic_numbers_tensor[discrete_tensor == 1] = 28  # Ni
    atomic_numbers_tensor[discrete_tensor == 2] = 78  # Pt
    # discrete_tensor == 0 の場合は 0 のまま (空)
    
    return atomic_numbers_tensor  # [4, 8, 8]

def calculate_composition(atomic_numbers_tensor):
    """
    原子番号テンソルからNi/Pt比率を計算
    """
    flat_tensor = atomic_numbers_tensor.flatten()
    ni_count = torch.sum(flat_tensor == 28).item()
    pt_count = torch.sum(flat_tensor == 78).item()
    total_atoms = ni_count + pt_count
    
    if total_atoms == 0:
        return 0.5, 0.5  # デフォルト値
    
    ni_fraction = ni_count / total_atoms
    pt_fraction = pt_count / total_atoms
    
    return ni_fraction, pt_fraction

def create_template_structure(ni_fraction, pt_fraction, size, vacuum):
    """
    テンプレート構造を作成
    """
    alloy_elements = ["Pt", "Ni"]
    fractions = [pt_fraction, ni_fraction]
    
    # Vegard法で格子定数を計算
    lattice_const = vegard_lattice_constant(alloy_elements, fractions)
    
    # fcc111構造の作成
    bulk = fcc111(symbol="Pt", 
                  size=size, 
                  a=lattice_const,
                  vacuum=vacuum, 
                  periodic=True)
    
    # 原子をソート
    bulk_sorted = sort_atoms(bulk, axes=("z", "y", "x"))
    
    return bulk_sorted, lattice_const

def load_existing_structures(data_dir, iter_names):
    """
    既存のiterの構造をASE Atomsオブジェクトとして読み込み
    """
    existing_structures = []
    
    for iter_name in iter_names:
        db_path = os.path.join(data_dir, f"{iter_name}_structures.json")
        if os.path.exists(db_path):
            print(f"既存構造を読み込み中: {db_path}")
            try:
                db = connect(db_path)
                count = 0
                for row in db.select():
                    atoms = row.toatoms(add_additional_information=True)
                    d = atoms.info.pop("data", {})
                    atoms.info["adsorbate_info"] = d["adsorbate_info"]
                    existing_structures.append(atoms)
                    count += 1
                print(f"  {count}個の構造を読み込みました")
            except Exception as e:
                print(f"警告: {db_path} の読み込みに失敗しました: {e}")
        else:
            print(f"警告: {db_path} が見つかりません")
    
    print(f"合計 {len(existing_structures)} 個の既存構造を読み込みました")
    return existing_structures

def generate_structures():
    """
    学習済みVAEから新しい構造を生成
    """
    # デフォルト値を動的に生成
    default_vae_model_path = f"/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae/result/iter{ITER}/final_cvae_iter{ITER}.pt"
    default_existing_iters = [f"iter{i}" for i in range(ITER + 1)]  # iter0からiter{ITER}まで
    
    parser = argparse.ArgumentParser(description="学習済みVAEを使用してfcc111表面の合金を生成")
    parser.add_argument("--num", type=int, default=128,
                        help="生成する構造数 (default: 128)")
    parser.add_argument("--output_dir", type=str, 
                        default="/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae/data",
                        help="出力ディレクトリ")
    parser.add_argument("--vae_model_path", type=str, 
                        default=default_vae_model_path,
                        help=f"学習済みVAEモデルのパス (default: iter{ITER}のモデル)")
    parser.add_argument("--target_condition", type=float, default=1.0,
                        help="Target condition for generation (default: 1.0)")
    parser.add_argument("--latent_size", type=int, default=128,
                        help="潜在変数の次元")
    parser.add_argument("--existing_iters", type=str, nargs='+', 
                        default=default_existing_iters,
                        help=f"重複チェックに含める既存のiter名のリスト (default: {default_existing_iters})")
    args = parser.parse_args()

    # パラメータ設定
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    size = [4, 4, 4]
    vacuum = None
    
    print("=== 学習済みVAEからの触媒構造生成 ===")
    print(f"現在のITER: {ITER}")
    print(f"使用デバイス: {DEVICE}")
    print(f"VAEモデルパス: {args.vae_model_path}")
    print(f"目標ni比率: {args.target_condition}")
    print(f"生成数: {args.num}")
    print(f"重複チェック対象iter: {args.existing_iters}")
    
    # ConditionalVAEクラスを読み込み
    ConditionalVAE = load_vae_class()
    
    # VAEモデルの初期化と重みの読み込み
    vae_model = ConditionalVAE(latent_size=args.latent_size, condition_dim=1).to(DEVICE)
    vae_model.load_state_dict(torch.load(args.vae_model_path, map_location=DEVICE))
    vae_model.eval()
    
    print("VAEモデルの読み込み完了")
    
    # 対称性等価性チェッカーの初期化
    symmetry_checker = SymmetryEquivalenceCheck(
        angle_tol=1.0,
        ltol=0.05,
        stol=0.05,
        vol_tol=0.1,
        scale_volume=True,
        to_primitive=True
    )
    
    # 出力先ディレクトリの設定
    data_dir = args.output_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 既存構造の読み込み
    existing_structures = load_existing_structures(data_dir, args.existing_iters)

    # 出力ファイルパスを動的に生成
    next_iter = ITER + 1
    db_path = os.path.join(data_dir, f"iter{next_iter}_structures.json")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = connect(db_path)
    
    print(f"出力先: iter{next_iter}_structures.json")
    print(f"{args.num}個の重複しない構造を生成中...")
    print(f"重複チェック対象: {len(existing_structures)}個の既存構造")
    
    # 生成済み構造のリスト（既存構造を含む）
    unique_structures = existing_structures.copy()
    successful_generations = 0
    max_attempts = args.num * 1000
    attempt = 0
    duplicate_with_existing = 0  # 既存構造との重複カウント
    duplicate_with_new = 0       # 新規生成同士の重複カウント
    
    with torch.no_grad():
        while successful_generations < args.num and attempt < max_attempts:
            try:
                attempt += 1
                
                # ランダムな潜在変数と条件の生成
                z = torch.randn(1, args.latent_size).to(DEVICE)
                condition = torch.tensor([[args.target_condition]], dtype=torch.float32).to(DEVICE)
                
                # VAEのデコーダ部分のみを使用
                generated_tensor = vae_model.decode(z, condition)  # [1, 12, 8, 8]
                
                # バッチ次元を除去
                if generated_tensor.dim() == 4:
                    generated_tensor = generated_tensor.squeeze(0)  # [12, 8, 8]
                
                # テンソルを原子番号に変換 ([12, 8, 8] -> [4, 8, 8])
                atomic_numbers_tensor = convert_tensor_to_atomic_numbers(generated_tensor)
                
                # 組成を計算
                ni_fraction, pt_fraction = calculate_composition(atomic_numbers_tensor)
                
                # 全て空の場合はスキップ
                if ni_fraction == 0 and pt_fraction == 0:
                    continue

                # テンプレート構造の作成
                template_structure, lattice_const = create_template_structure(
                    ni_fraction, pt_fraction, size, vacuum)
                
                # ASE Atomsオブジェクトに変換
                final_structure = tensor_to_structure(atomic_numbers_tensor, template_structure)

                # 化学式にXが含まれる構造を除外
                chemical_formula = final_structure.get_chemical_formula()
                if 'X' in chemical_formula:
                    if attempt % 100 == 0:
                        print(f"試行 {attempt}: 未知元素(X)を含む構造をスキップ (成功: {successful_generations}/{args.num})")
                    continue

                # 対称性等価性チェック（既存構造も含めて）
                is_duplicate = False
                if unique_structures:
                    # 既存構造との重複チェック
                    for i, existing_structure in enumerate(unique_structures):
                        if symmetry_checker.compare(final_structure, [existing_structure]):
                            is_duplicate = True
                            if i < len(existing_structures):
                                duplicate_with_existing += 1
                            else:
                                duplicate_with_new += 1
                            break
                
                if is_duplicate:
                    if attempt % 100 == 0:
                        print(f"試行 {attempt}: 重複構造をスキップ (成功: {successful_generations}/{args.num}, "
                              f"既存重複: {duplicate_with_existing}, 新規重複: {duplicate_with_new})")
                    continue
                
                # 重複していない場合は追加
                unique_structures.append(final_structure.copy())
                
                # EMT計算器の設定
                final_structure.calc = EMT()

                # 表面情報の取得
                ads_info = final_structure.info.get("adsorbate_info", {})
                
                # データベースへの書き込み
                data = {
                    "chemical_formula": final_structure.get_chemical_formula(),
                    "ni_fraction": float(ni_fraction),
                    "pt_fraction": float(pt_fraction),
                    "lattice_constant": float(lattice_const),
                    "run": successful_generations + 1,
                    "generation_method": "conditional_vae",
                    "target_condition": float(args.target_condition),
                    "adsorbate_info": ads_info,
                    "total_attempts": attempt
                }
                
                db.write(final_structure, data=data)
                successful_generations += 1
                
                print(f"生成完了: {successful_generations}/{args.num} (試行回数: {attempt})")
                    
            except Exception as e:
                print(f"試行 {attempt} でエラーが発生: {e}")
                continue
    
    print(f"\n=== 生成結果 ===")
    print(f"成功した生成数: {successful_generations}/{args.num}")
    print(f"総試行回数: {attempt}")
    print(f"成功率: {successful_generations/attempt*100:.2f}%")
    print(f"既存構造との重複: {duplicate_with_existing}回")
    print(f"新規生成同士の重複: {duplicate_with_new}回")
    print(f"総重複回数: {duplicate_with_existing + duplicate_with_new}回")
    print(f"生成された構造を保存: {db_path}")
    
    if successful_generations < args.num:
        print(f"警告: 目標数に達しませんでした。パラメータ調整を検討してください。")
    
    print("処理完了")

def main():
    generate_structures()

if __name__ == "__main__":
    main()