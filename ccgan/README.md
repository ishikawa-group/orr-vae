# ORR Catalyst Generator with Conditional GAN

条件付きGANを用いたORR（酸素還元反応）触媒の反復的設計システム

## 概要

このシステムは、Pt-Ni合金触媒のORR過電圧を最小化する構造を、条件付きGANを用いて反復的に探索。

### 主な特徴

- **iter0**: ランダム構造生成 → ORR過電圧計算 → 条件付きGAN学習
- **iter1以降**: GAN生成構造 → ORR過電圧計算 → GAN再学習（累積データ使用）
- **目標**: ORR過電圧 < 0.5V かつ Pt含有量 < 50% の構造生成

## ファイル構成

```
code_2/
├── 01_generate_random_structures.py  # ランダム構造生成（iter0のみ）
├── 02_calculate_overpotentials.py    # ORR過電圧計算（モック実装）
├── 03_conditional_gan.py             # 条件付きGAN学習
├── 04_generate_new_structures.py     # GAN による新構造生成
├── run_iter0.sh                      # iter0実行スクリプト
├── run_iter_next.sh                  # iter1以降実行スクリプト
├── tool.py                           # ユーティリティ関数
└── README.md                         # このファイル
```

## データ表現

### 構造データ
- **入力**: Pt4×4×4構造（64原子）
- **テンソル変換**: 4チャンネル×8×8（各層を1チャンネルとして表現）
- **元素マッピング**: 0（空サイト）, 1（Ni）, 2（Pt）

### 条件ラベル
- **ORR過電圧ラベル**: 0.5V未満なら1、以上なら0
- **Pt含有量ラベル**: 50%未満なら1、以上なら0

## GANアーキテクチャ

### 生成器 (Generator)
- **入力**: 128次元ノイズ + 2次元条件ラベル
- **出力**: 4チャンネル×8×8テンソル（確率分布）
- **構造**: 線形層 → 転置畳み込み層

### 識別器 (Discriminator)
- **入力**: 4チャンネル×8×8テンソル
- **出力**: 3ノード（真偽判定, ORR過電圧予測, Pt含有量予測）
- **構造**: 畳み込み層 → 線形層

### 損失関数
- **生成器**: 真データ + 目標条件（ORR過電圧=1, Pt含有量=1）を目指す
- **識別器**: 真偽判定 + 条件ラベル予測

## 使用方法

### 1. 環境構築

```bash
# 必要なライブラリのインストール
pip install ~

### 2. iter0の実行

```bash
# ランダム構造生成 → 過電圧計算 → GAN学習
./run_iter0.sh
```

これにより以下が実行されます：
- 100個のランダムPt-Ni合金構造生成
- ORR過電圧計算（モック）
- 条件付きGAN学習（100エポック）

### 3. iter1以降の実行

```bash
# GAN生成構造 → 過電圧計算 → GAN再学習
./run_iter_next.sh 1
./run_iter_next.sh 2
./run_iter_next.sh 3
# ... 必要な回数だけ繰り返し
```

### 4. 個別実行

各ステップを個別に実行することも可能：

```bash
# ランダム構造生成
python 01_generate_random_structures.py --num 100 --output_dir ./data

# 過電圧計算
python 02_calculate_overpotentials.py \
    --structures_file ./data/iter0_structures.json \
    --output_file ./data/iter0_overpotentials.json

# GAN学習
python 03_conditional_gan.py \
    --structures_file ./data/iter0_structures.json \
    --overpotentials_file ./data/iter0_overpotentials.json \
    --epochs 100

# 新構造生成
python 04_generate_new_structures.py \
    --model_path ./models/generator.pth \
    --num_structures 100 \
    --iter_num 1
```

## 出力ファイル

### データファイル
- `data/iter{N}_structures.json`: 構造データ
- `data/iter{N}_overpotentials.json`: 過電圧データ
- `data/all_structures.json`: 全イテレーションの構造データ統合
- `data/all_overpotentials.json`: 全イテレーションの過電圧データ統合

### モデルファイル
- `models/generator.pth`: 学習済み生成器
- `models/discriminator.pth`: 学習済み識別器
- `models/training_loss.png`: 学習曲線

## パラメータ調整

主要なパラメータは各スクリプトのコマンドライン引数で調整可能：

- `--num`: 生成構造数
- `--epochs`: 学習エポック数
- `--batch_size`: バッチサイズ
- `--lr`: 学習率
- `--seed`: ランダムシード

## 注意事項

1. **ORR過電圧計算**: 現在はモック実装である。実際の使用では`orr_overpotential_calculator`ライブラリに置き換える必要がある。

2. **計算資源**: GANの学習には時間がかかる。GPU使用を推奨する。

3. **テンソル変換**: `tool.py`の`slab_to_tensor`と`tensor_to_slab`関数が正しく動作することを確認する必要がある。



## トラブルシューティング

### よくある問題

1. **テンソル変換エラー**: 
   - 構造の原子数が4×4×4=64と一致しているか確認
   - `tool.py`の関数が正しくインポートされているか確認

2. **GPU メモリ不足**:
   - バッチサイズを小さくする (`--batch_size 8`)
   - CPUモードで実行

3. **学習が収束しない**:
   - 学習率を調整 (`--lr 0.0001`)
   - エポック数を増やす (`--epochs 200`)

## カスタマイズ

- **元素の変更**: `atomic_numbers`の参照を変更
- **構造サイズの変更**: `grid_size`パラメータを調整
- **条件ラベルの変更**: 閾値（0.5V, 50%）を変更
- **GANアーキテクチャの変更**: `Generator`と`Discriminator`クラスを修正