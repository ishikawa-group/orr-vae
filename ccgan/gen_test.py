#!/usr/bin/env python
"""
GANの生成器から出力されるテンソルの形状を確認するスクリプト
"""
import os
import torch
import numpy as np
import importlib.util
import sys

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
    生成されたテンソルを原子番号に変換（形状確認付き）
    1. torch.round(tensor).clamp(0, 2) で各要素を 0,1,2 にする
    2. 1 -> 28 (Ni), 2 -> 78 (Pt), 0 -> 0 (空) にマッピング
    """
    print(f"  変換前テンソル形状: {tensor.shape}")
    print(f"  変換前テンソル値の範囲: [{tensor.min():.4f}, {tensor.max():.4f}]")
    
    # Step 1: 四捨五入してクランプ
    discrete_tensor = torch.round(tensor).clamp(0, 2)
    print(f"  四捨五入後テンソル形状: {discrete_tensor.shape}")
    print(f"  四捨五入後テンソル値の範囲: [{discrete_tensor.min():.0f}, {discrete_tensor.max():.0f}]")
    
    # 各値の分布を確認
    unique_values, counts = torch.unique(discrete_tensor, return_counts=True)
    print(f"  四捨五入後の値分布:")
    for val, count in zip(unique_values, counts):
        print(f"    値 {val:.0f}: {count} 個")
    
    # Step 2: 原子番号へのマッピング
    atomic_numbers_tensor = torch.zeros_like(discrete_tensor, dtype=torch.int64)
    atomic_numbers_tensor[discrete_tensor == 1] = 28  # Ni
    atomic_numbers_tensor[discrete_tensor == 2] = 78  # Pt
    # discrete_tensor == 0 の場合は 0 のまま (空)
    
    print(f"  原子番号テンソル形状: {atomic_numbers_tensor.shape}")
    print(f"  原子番号テンソル値の範囲: [{atomic_numbers_tensor.min()}, {atomic_numbers_tensor.max()}]")
    
    # 原子番号の分布を確認
    unique_atoms, atom_counts = torch.unique(atomic_numbers_tensor, return_counts=True)
    print(f"  原子番号分布:")
    for atom_num, count in zip(unique_atoms, atom_counts):
        if atom_num == 0:
            print(f"    空 (0): {count} 個")
        elif atom_num == 28:
            print(f"    Ni (28): {count} 個")
        elif atom_num == 78:
            print(f"    Pt (78): {count} 個")
    
    return atomic_numbers_tensor

def inspect_tensor_shapes():
    """
    Generatorの出力テンソルと四捨五入後のテンソル形状を確認
    """
    # パラメータ設定
    GENERATOR_PATH = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/iter0/final_generator.pt"
    OUTPUT_DIR = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/tensor_inspection"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=== Generator出力テンソル形状確認 ===")
    print(f"使用デバイス: {DEVICE}")
    print(f"生成器パス: {GENERATOR_PATH}")
    
    # Generatorクラスを読み込み
    Generator = load_generator_class()
    
    # 生成器の初期化と重みの読み込み
    generator = Generator(latent_size=128, condition_dim=1).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()
    
    print("生成器の読み込み完了")
    
    # 複数の条件で生成して調査
    conditions = [0.0, 1.0]  # 低性能、高性能
    num_samples = 3  # 各条件で3つのサンプル
    
    with torch.no_grad():
        for cond_idx, condition_value in enumerate(conditions):
            print(f"\n=== 条件値 {condition_value} での生成 ===")
            
            for sample_idx in range(num_samples):
                print(f"\nサンプル {sample_idx + 1}:")
                
                # ランダムノイズと条件の生成
                z = torch.randn(1, generator.latent_size).to(DEVICE)
                condition = torch.tensor([[condition_value]], dtype=torch.float32).to(DEVICE)
                
                # 構造テンソルの生成
                generated_tensor = generator(z, condition)
                print(f"  Generator出力形状: {generated_tensor.shape}")
                print(f"  Generator出力データ型: {generated_tensor.dtype}")
                
                # バッチ次元を除去
                if generated_tensor.dim() == 4:
                    squeezed_tensor = generated_tensor.squeeze(0)
                    print(f"  squeeze後の形状: {squeezed_tensor.shape}")
                else:
                    squeezed_tensor = generated_tensor
                    print(f"  squeeze不要（既に3次元）: {squeezed_tensor.shape}")
                
                # 期待される形状(4, 8, 8)かチェック
                expected_shape = (4, 8, 8)
                if squeezed_tensor.shape == expected_shape:
                    print(f"  ✓ 期待される形状 {expected_shape} と一致")
                else:
                    print(f"  ✗ 期待される形状 {expected_shape} と異なる: {squeezed_tensor.shape}")
                
                # テンソルを原子番号に変換（詳細な形状確認付き）
                atomic_numbers_tensor = convert_tensor_to_atomic_numbers(squeezed_tensor)
                
                # 簡単な統計情報
                print(f"  総要素数: {atomic_numbers_tensor.numel()}")
                print(f"  期待される要素数 (4×8×8): {4*8*8}")
                
                # テンソル内容をファイルに保存
                filename = f"tensor_shape_analysis_cond{condition_value}_sample{sample_idx + 1}.txt"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, 'w') as f:
                    f.write(f"=== Generator出力テンソル形状解析 ===\n")
                    f.write(f"条件値: {condition_value}\n")
                    f.write(f"サンプル: {sample_idx + 1}\n\n")
                    
                    f.write(f"Generator出力:\n")
                    f.write(f"  形状: {generated_tensor.shape}\n")
                    f.write(f"  データ型: {generated_tensor.dtype}\n")
                    f.write(f"  デバイス: {generated_tensor.device}\n\n")
                    
                    f.write(f"squeeze後:\n")
                    f.write(f"  形状: {squeezed_tensor.shape}\n")
                    f.write(f"  期待形状との一致: {squeezed_tensor.shape == expected_shape}\n\n")
                    
                    f.write(f"四捨五入・クランプ後:\n")
                    discrete_tensor = torch.round(squeezed_tensor).clamp(0, 2)
                    f.write(f"  形状: {discrete_tensor.shape}\n")
                    f.write(f"  値の範囲: [{discrete_tensor.min():.0f}, {discrete_tensor.max():.0f}]\n\n")
                    
                    f.write(f"原子番号変換後:\n")
                    f.write(f"  形状: {atomic_numbers_tensor.shape}\n")
                    f.write(f"  データ型: {atomic_numbers_tensor.dtype}\n")
                    f.write(f"  総要素数: {atomic_numbers_tensor.numel()}\n\n")
                    
                    # 各層の詳細
                    if atomic_numbers_tensor.shape[0] == 4:  # z方向が4層の場合
                        f.write(f"各層の統計:\n")
                        for layer in range(4):
                            layer_data = atomic_numbers_tensor[layer]
                            unique_vals, counts = torch.unique(layer_data, return_counts=True)
                            f.write(f"  層 {layer}: 形状 {layer_data.shape}\n")
                            for val, count in zip(unique_vals, counts):
                                if val == 0:
                                    f.write(f"    空: {count}\n")
                                elif val == 28:
                                    f.write(f"    Ni: {count}\n")
                                elif val == 78:
                                    f.write(f"    Pt: {count}\n")
                
                print(f"  詳細解析を保存: {filename}")
    
    print(f"\n=== 形状確認完了 ===")
    print(f"詳細結果は {OUTPUT_DIR} に保存されました")

if __name__ == "__main__":
    inspect_tensor_shapes()