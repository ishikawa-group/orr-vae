#!/usr/bin/env python
"""
GANの生成器から新しい触媒構造を生成するスクリプト
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

def load_generator_class():
    """03_conditional_gan.pyからGeneratorクラスを動的にインポート"""
    # 現在のスクリプトのディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gan_script_path = os.path.join(current_dir, "03_conditional_gan.py")
    
    # ファイルの存在確認
    if not os.path.exists(gan_script_path):
        raise FileNotFoundError(f"GANスクリプトが見つかりません: {gan_script_path}")
    
    print(f"GANスクリプトを読み込み中: {gan_script_path}")
    
    spec = importlib.util.spec_from_file_location("conditional_gan", gan_script_path)
    conditional_gan_module = importlib.util.module_from_spec(spec)
    sys.modules["conditional_gan"] = conditional_gan_module
    spec.loader.exec_module(conditional_gan_module)
    return conditional_gan_module.Generator

def convert_tensor_to_atomic_numbers(tensor):
    """
    生成されたテンソル (4, 8, 8) を原子番号に変換
    1. torch.round(tensor)で各要素を 0,1,2 にする
    2. 1 -> 28 (Ni), 2 -> 78 (Pt), 0 -> 0 (空) にマッピング
    """
    # Step 1: 四捨五入してクランプ
    discrete_tensor = torch.round(tensor).clamp(0, 2).long()
    
    # Step 2: 原子番号へのマッピング
    atomic_numbers_tensor = torch.zeros_like(discrete_tensor, dtype=torch.int64)
    atomic_numbers_tensor[discrete_tensor == 1] = 28  # Ni
    atomic_numbers_tensor[discrete_tensor == 2] = 78  # Pt
    # discrete_tensor == 0 の場合は 0 のまま (空)
    
    return atomic_numbers_tensor

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
    
    # 原子をソート（tensor_to_structureで必要）
    bulk_sorted = sort_atoms(bulk, axes=("z", "y", "x"))
    
    return bulk_sorted, lattice_const

def generate_structures():
    """
    GANの生成器から新しい構造を生成（重複チェック付き）
    """
    # --- コマンドライン引数の定義 ---
    parser = argparse.ArgumentParser(description="GANを使用してfcc111表面の合金を生成")
    parser.add_argument("--num", type=int, default=128,
                        help="Number of structures to generate (default: 128)")
    parser.add_argument("--output_dir", type=str, default="/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data",
                        help="Output directory (default: /gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data)")
    parser.add_argument("--generator_path", type=str, 
                        default="/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/result/iter1/final_generator_iter1.pt",
                        help="Path to the generator model")
    parser.add_argument("--target_condition", type=float, default=1.0,
                        help="Target condition for generation (default: 1.0)")
    args = parser.parse_args()

    # --- パラメータ設定 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    size = [4, 4, 4]
    vacuum = None
    
    print("=== GAN生成器からの触媒構造生成 ===")
    print(f"使用デバイス: {DEVICE}")
    print(f"生成器パス: {args.generator_path}")
    print(f"生成条件: 高性能触媒（ラベル={args.target_condition}）")
    print(f"生成数: {args.num}")
    
    # Generatorクラスを読み込み
    Generator = load_generator_class()
    
    # 生成器の初期化と重みの読み込み
    generator = Generator(latent_size=128, condition_dim=1).to(DEVICE)
    generator.load_state_dict(torch.load(args.generator_path, map_location=DEVICE))
    generator.eval()
    
    print("生成器の読み込み完了")
    
    # 対称性等価性チェッカーの初期化
    symmetry_checker = SymmetryEquivalenceCheck(
        angle_tol=1.0,      # 角度許容度（度）
        ltol=0.05,          # 格子ベクトル長の相対許容度
        stol=0.05,          # サイト位置の許容度
        vol_tol=0.1,        # 体積許容度
        scale_volume=True,  # 体積をスケールして比較
        to_primitive=True   # プリミティブセルに変換してから比較
    )
    
    # --- 出力先ディレクトリの設定 ---
    data_dir = args.output_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 出力ファイルのパス
    db_path = os.path.join(data_dir, "iter2_structures.json")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = connect(db_path)
    
    print(f"{args.num}個の重複しない高性能触媒構造を生成中...")
    
    # 生成済み構造のリスト
    unique_structures = []
    successful_generations = 0
    max_attempts = args.num * 100  # 最大試行回数（numの10倍）
    attempt = 0
    
    with torch.no_grad():
        while successful_generations < args.num and attempt < max_attempts:
            try:
                attempt += 1
                
                # ランダムノイズと条件の生成
                z = torch.randn(1, generator.latent_size).to(DEVICE)
                condition = torch.tensor([[args.target_condition]], dtype=torch.float32).to(DEVICE)
                
                # 構造テンソルの生成 (期待される形状: [1, 4, 8, 8])
                generated_tensor = generator(z, condition)
                
                # バッチ次元を除去して (4, 8, 8) にする
                if generated_tensor.dim() == 4:
                    generated_tensor = generated_tensor.squeeze(0)  # [4, 8, 8]
                
                # テンソルを原子番号に変換
                atomic_numbers_tensor = convert_tensor_to_atomic_numbers(generated_tensor)
                
                # 組成を計算
                ni_fraction, pt_fraction = calculate_composition(atomic_numbers_tensor)
                
                # 全て空の場合はスキップ
                if ni_fraction == 0 and pt_fraction == 0:
                    continue
                
                # テンプレート構造の作成
                template_structure, lattice_const = create_template_structure(
                    ni_fraction, pt_fraction, size, vacuum)
                
                # tensor_to_structure関数を使用してASE Atomsオブジェクトに変換
                final_structure = tensor_to_structure(atomic_numbers_tensor, template_structure)

                # chemical_formulaにXが含まれる構造を除外
                chemical_formula = final_structure.get_chemical_formula()
                if 'X' in chemical_formula:
                    if attempt % 100 == 0:  # 100回に1回進捗を表示
                        print(f"試行 {attempt}: 未知元素(X)を含む構造を検出、スキップ中... (成功: {successful_generations}/{args.num})")
                    continue

                # 対称性等価性チェック
                is_duplicate = False
                if unique_structures:  # 既に生成済みの構造がある場合
                    is_duplicate = symmetry_checker.compare(final_structure, unique_structures)
                
                if is_duplicate:
                    if attempt % 100 == 0:  # 100回に1回進捗を表示
                        print(f"試行 {attempt}: 重複構造を検出、スキップ中... (成功: {successful_generations}/{args.num})")
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
                    "run": successful_generations + 1,  # 成功した生成数をrunに設定
                    "generation_method": "conditional_gan",
                    "target_condition": float(args.target_condition),
                    "adsorbate_info": ads_info,
                    "total_attempts": attempt  # 総試行回数も記録
                }
                
                db.write(final_structure, data=data)
                successful_generations += 1
                
                print(f"生成完了: {successful_generations}/{args.num} (試行回数: {attempt})")
                    
            except Exception as e:
                print(f"試行 {attempt} でエラーが発生しました: {e}")
                continue
    
    print(f"\n=== 生成結果 ===")
    print(f"成功した生成数: {successful_generations}/{args.num}")
    print(f"総試行回数: {attempt}")
    print(f"成功率: {successful_generations/attempt*100:.2f}%")
    print(f"生成された構造を保存しました: {db_path}")
    
    if successful_generations < args.num:
        print(f"警告: 目標数 ({args.num}) に達しませんでした。最大試行回数 ({max_attempts}) に到達しました。")
        print("パラメータを調整するか、最大試行回数を増やすことを検討してください。")
    
    print("処理完了")

def main():
    """メイン関数"""
    generate_structures()

if __name__ == "__main__":
    main()