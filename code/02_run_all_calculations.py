#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_uncalculated.pyを連続実行するスクリプト
"""
import subprocess
import time
import sys
import re
import argparse
import os
from pathlib import Path

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='02_run_uncalculated.pyの連続実行')
    parser.add_argument('--iter', type=int, default=0,
                       help='イテレーション番号 (default: 0)')
    parser.add_argument('--max_count', type=int, default=128*2,
                       help='最大実行回数 (default: 128*2)')
    parser.add_argument('--wait_time', type=int, default=2,
                       help='各実行間の待機時間（秒） (default: 2)')
    parser.add_argument('--script_path', type=str,
                       default=str(Path(__file__).parent / "02_run_uncalculated.py"),
                       help='02_run_uncalculated.pyのパス')
    parser.add_argument('--base_dir', type=str,
                       default=str(Path(__file__).parent),
                       help='実行ディレクトリ')
    parser.add_argument('--calc_type', type=str, default='mace',
                       choices=['mace', 'vasp', 'emt'],
                       help='計算タイプ (default: mace)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='ログレベル (default: INFO)')
    parser.add_argument('--force', action='store_true',
                       help='強制実行フラグ')
    parser.add_argument('--keep_temp', action='store_true',
                       help='一時ディレクトリを保持')
    return parser.parse_args()

def main():
    # コマンドライン引数を取得
    args = parse_args()
    
    # パラメータ設定
    ITER = args.iter
    SCRIPT_PATH = args.script_path
    MAX_COUNT = args.max_count
    WAIT_TIME = args.wait_time
    BASE_DIR = args.base_dir
    
    print(f"=== 02_run_uncalculated.py 連続実行 (iter{ITER}) ===")
    print(f"スクリプトパス: {SCRIPT_PATH}")
    print(f"最大実行回数: {MAX_COUNT}")
    print(f"待機時間: {WAIT_TIME}秒")
    print(f"実行ディレクトリ: {BASE_DIR}")
    print(f"計算タイプ: {args.calc_type}")
    print(f"ログレベル: {args.log_level}")
    print(f"強制実行: {args.force}")
    print(f"一時ディレクトリ保持: {args.keep_temp}")
    
    # ファイル存在チェック
    if not os.path.exists(SCRIPT_PATH):
        print(f"エラー: スクリプトが見つかりません: {SCRIPT_PATH}")
        sys.exit(1)
    
    if not os.path.exists(BASE_DIR):
        print(f"エラー: 実行ディレクトリが見つかりません: {BASE_DIR}")
        sys.exit(1)
    
    success_count = 0
    error_count = 0
    
    print(f"\n02_run_uncalculated.pyを最大{MAX_COUNT}回実行します")
    
    for i in range(1, MAX_COUNT + 1):
        print(f"-------------------------------------")
        print(f"実行 {i} / {MAX_COUNT} (成功: {success_count}, エラー: {error_count})")
        
        # 02_run_uncalculated.pyの実行コマンドを構築
        cmd = [
            "python3", 
            SCRIPT_PATH,
            "--iter", str(ITER),
            "--calc_type", args.calc_type,
            "--log_level", args.log_level
        ]
        
        # オプションフラグの追加
        if args.force:
            cmd.append("--force")
        if args.keep_temp:
            cmd.append("--keep_temp")
        
        print(f"実行コマンド: {' '.join(cmd)}")
        
        # スクリプト実行と出力キャプチャ
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=BASE_DIR
        )
        stdout, stderr = process.communicate()
        
        # 出力を表示
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        
        # 「未計算の構造はありません」というメッセージを検出
        if "未計算の構造はありません" in stdout:
            print("すべての構造が計算済みのため、処理を終了します")
            break
        
        # 成功したかを確認
        if process.returncode == 0:
            if re.search(r"の計算が完了しました", stdout):
                success_count += 1
                print(f"✓ 成功: {success_count}")
            else:
                print("⚠ 警告: 計算完了メッセージが見つかりませんでした")
        else:
            error_count += 1
            print(f"✗ エラーが発生しました (終了コード: {process.returncode})")
            print(f"✗ エラー回数: {error_count}")
        
        # 統計情報の表示
        total_attempts = success_count + error_count
        if total_attempts > 0:
            success_rate = (success_count / total_attempts) * 100
            print(f"現在の成功率: {success_rate:.1f}% ({success_count}/{total_attempts})")
        
        # 次の実行まで待機
        if i < MAX_COUNT:
            print(f"{WAIT_TIME}秒待機中...")
            time.sleep(WAIT_TIME)
    
    print(f"-------------------------------------")
    print(f"=== 処理完了 (iter{ITER}) ===")
    print(f"成功: {success_count}件")
    print(f"エラー: {error_count}件")
    print(f"総実行回数: {success_count + error_count}件")
    if success_count + error_count > 0:
        final_success_rate = (success_count / (success_count + error_count)) * 100
        print(f"最終成功率: {final_success_rate:.1f}%")
    
    if success_count > 0:
        print(f"✓ {success_count}件の計算が正常に完了しました")
    if error_count > 0:
        print(f"⚠ {error_count}件の計算でエラーが発生しました")

if __name__ == "__main__":
    main()