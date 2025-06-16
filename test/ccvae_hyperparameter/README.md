# Conditional VAE of ORR Catalyst Generator のハイパーパラメータチューニング

## Overview (概要)

ORR（酸素還元反応）触媒に向けた条件付きVAEのハイパーパラメータチューニングを行う。

## Data Representation (データ表現)

### Structure Data (構造データ)
- **入力**: Pt4×4×4構造（64原子）
- **テンソル変換**: 4チャンネル×8×8（各層を1チャンネルとして表現）
- **元素マッピング**: 0（空サイト）, 1（Ni）, 2（Pt）

### Condition Labels (条件ラベル)
- **ORR過電圧ラベル**: データセットの内、過電圧が低いもの64個を1、残りを0とする

## Conditional VAE Architecture (条件付きVAEアーキテクチャ)

### Encoder (エンコーダ)
- **入力**: 4チャンネル×8×8テンソル + 1次元条件ラベル
- **条件埋め込み**: 線形層（1→16→16→8次元）で条件を変換
- **結合**: 条件を空間的に拡張して入力テンソルと結合（12チャンネル）
- **出力**: 潜在変数の平均μと分散logvar（各n次元）
- **構造**: 畳み込み層（256→512→1024） → 全結合層

### Decoder (デコーダ)
- **入力**: n次元潜在変数 + 1次元条件ラベル
- **条件埋め込み**: 線形層（1→16→16→8次元）で条件を変換
- **結合**: 潜在変数と条件埋め込みを結合（136次元）
- **出力**: 12チャンネル×8×8テンソル（各層3クラス分類用logits）
- **構造**: 全結合層 → 転置畳み込み層（64→n→64→32→12）

### Loss Function (損失関数)
- **再構成損失**: 各層でクラス重み付きクロスエントロピー
- **KL発散**: 潜在変数の正規化項
- **総損失**: 再構成損失 + β × KL発散

## 探索ワークフロー

### VAEによる触媒生成・散策の反復的ワークフロー

本研究では、条件付きVAEを用いた高性能ORR触媒の探索を反復的に実行する。各イテレーションにおいて以下の5つのステップを順次実行する：

#### ステップ1: 初期構造生成（Iteration 0）
```bash
python3 01_generate_random_structures.py --num 128
```
- Pt(111)面上にNiとPtをランダム配置した初期構造を128個生成
- Pt4×4×4のスーパーセル構造（64原子）を使用

#### ステップ2: DFT計算による物性評価
```bash
python3 02_run_all_calculations.py --iter [n]
```
- 第一原理DFT計算（VASP）による酸素還元反応過電圧の計算
- ORR反応中間体（O*, OH*, OOH*）の吸着エネルギー評価
- 過電圧に基づく条件ラベル生成（低過電圧: 1, 高過電圧: 0）

#### ステップ3: 条件付きVAE学習
```bash
python3 03_conditional_vae.py --iter [n] --max_epoch 200 --beta [β] --latent_size [n_dim]
```
- 構造データ（4×8×8テンソル）と過電圧条件ラベルを用いたVAE訓練
- 損失関数: 再構成損失 + β × KL発散
- ハイパーパラメータ: β値（KL項の重み）、潜在変数次元数

#### ステップ4: 新規構造生成
```bash
python3 04_generate_new_structures.py --iter [n] --num 64 --target_condition 1 --latent_size [n_dim]
python3 04_generate_new_structures.py --iter [n] --num 64 --target_condition 0 --latent_size [n_dim]
```
- 訓練済みVAEデコーダによる新規触媒構造生成
- **高性能条件（target_condition=1）および低性能条件（target_condition=0）で各64構造**
- 潜在空間からのサンプリングと条件付き生成

#### ステップ5: 潜在空間可視化
```bash
python3 05_visualize_latent_space.py --iter [n] --latent_size [n_dim]
```
- t-SNEによる潜在空間の2次元可視化
- 高性能・低性能サンプルの潜在空間における分布確認

### 反復プロセス
上記ステップ2-5を複数イテレーション（通常5回）繰り返し、段階的に高性能触媒を探索する。各イテレーションで新たに生成された構造を既存データセットに追加し、VAEの学習データを拡張することで、より精度の高い構造生成を実現する。

### グリッドサーチによるハイパーパラメータ探索

#### 探索対象パラメータ

1. **β値（KL発散項の重み）**
   - 探索範囲: 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 50.0
   - 効果: 潜在空間の正則化強度を制御
   - 小さい値: 再構成精度重視、大きい値: 潜在空間の滑らかさ重視

2. **潜在変数次元数**
   - 探索範囲: 8, 16, 32, 64, 128, 256
   - 効果: 潜在表現の表現力と計算効率のトレードオフ
   - 小さい次元: 計算効率、大きい次元: 表現力

#### グリッドサーチ実行方法

各パラメータ組み合わせに対して独立したジョブスクリプト（[`run_tsubame_vae_*.sh`](ORR_catalyst_generator/ccvae_test/)）を作成し、TSUBAME計算環境で並列実行する。

例：β=0.1, 潜在次元=64の場合
```bash
# beta_size_0-1/run_tsubame_vae_0-1.sh
python3 03_conditional_vae.py --iter 0 --max_epoch 200 --beta 0.1 --latent_size 64
```

例：β=1.0, 潜在次元=8の場合
```bash
# latent_size_8/run_tsubame_vae_8.sh
python3 03_conditional_vae.py --iter 0 --max_epoch 200 --beta 1 --latent_size 8
```

#### 評価指標
- **触媒性能**: 生成構造の過電圧分布、高性能触媒発見率
- **学習効率**: 学習曲線（再構成損失、KL損失の収束性）
- **潜在空間品質**: t-SNE可視化による潜在表現の分離性能


## Results (結果)

### 潜在変数次元の影響 (beta=1.0)

<table>
<tr>
<th>潜在変数次元</th>
<th>Learning Curve (iter)</th>
<th>次元圧縮後の潜在変数空間</th>
</tr>
<tr>
<td>8</td>
<td><img src="result/latent_size/latent_size_8/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/latent_size/latent_size_8/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>16</td>
<td><img src="result/latent_size/latent_size_16/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/latent_size/latent_size_16/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>32</td>
<td><img src="result/latent_size/latent_size_32/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/latent_size/latent_size_32/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>64</td>
<td><img src="result/latent_size/latent_size_64/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/latent_size/latent_size_64/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>128</td>
<td><img src="result/latent_size/latent_size_128/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/latent_size/latent_size_128/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>256</td>
<td><img src="result/latent_size/latent_size_256/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/latent_size/latent_size_256/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
</table>

<table>
<tr>
<th>潜在変数次元</th>
<th>触媒探索結果</th>
<th>iter毎の推移</th>
</tr>
<tr>
<td>8</td>
<td><img src="result/latent_size/latent_size_8/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/latent_size/latent_size_8/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>16</td>
<td><img src="result/latent_size/latent_size_16/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/latent_size/latent_size_16/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>32</td>
<td><img src="result/latent_size/latent_size_32/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/latent_size/latent_size_32/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>64</td>
<td><img src="result/latent_size/latent_size_64/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/latent_size/latent_size_64/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>128</td>
<td><img src="result/latent_size/latent_size_128/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/latent_size/latent_size_128/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>256</td>
<td><img src="result/latent_size/latent_size_256/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/latent_size/latent_size_256/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
</table>

### iter毎の最小過電圧の推移


<img src="result/analysis/min_overpotential_comparison_latent_size.png" width="500">


| Latent | iter0 | iter1 | iter2 | iter3 | iter4 | iter5 | Δ(iter5 - iter0) |
|--------|-------|-------|-------|-------|-------|-------|-------------------|
| 8      | 0.297 | 0.358 | 0.280 | 0.274 | 0.271 | 0.273 | -0.024 |
| 16     | 0.297 | 0.332 | 0.287 | 0.282 | 0.283 | 0.253 | -0.044 |
| 32     | 0.297 | 0.334 | 0.285 | 0.285 | 0.285 | 0.278 | -0.019 |
| 64     | 0.297 | 0.332 | 0.283 | 0.270 | 0.283 | 0.283 | -0.014 |
| 128    | 0.297 | 0.320 | 0.289 | 0.284 | 0.283 | 0.287 | -0.010 |
| 256    | 0.297 | 0.323 | 0.287 | 0.282 | 0.285 | 0.284 | -0.013 |




### betaサイズの影響 (latent_size=64)

<table>
<tr>
<th>β値</th>
<th>Learning Curve (iter4)</th>
<th>次元圧縮後の潜在変数空間</th>
</tr>
<tr>
<td>0.1</td>
<td><img src="result/beta_size/beta_size_0-1/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/beta_size/beta_size_0-1/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>0.5</td>
<td><img src="result/beta_size/beta_size_0-5/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/beta_size/beta_size_0-5/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>1</td>
<td><img src="result/beta_size/beta_size_1/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/beta_size/beta_size_1/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>2.5</td>
<td><img src="result/beta_size/beta_size_2-5/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/beta_size/beta_size_2-5/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>5</td>
<td><img src="result/beta_size/beta_size_5/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/beta_size/beta_size_5/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
<tr>
<td>10</td>
<td><img src="result/beta_size/beta_size_10/result/iter4/learning_curves.png" width="800"></td>
<td><img src="result/beta_size/beta_size_10/result/visualization/iter4/tsne_latent_space_iter4_mean_all_data.png" width="300"></td>
</tr>
</table>

<table>
<tr>
<th>β値</th>
<th>触媒探索結果</th>
<th>iter毎の推移</th>
</tr>
<tr>
<td>0.1</td>
<td><img src="result/beta_size/beta_size_0-1/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/beta_size/beta_size_0-1/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>0.5</td>
<td><img src="result/beta_size/beta_size_0-5/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/beta_size/beta_size_0-5/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>1</td>
<td><img src="result/beta_size/beta_size_1/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/beta_size/beta_size_1/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>2.5</td>
<td><img src="result/beta_size/beta_size_2-5/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/beta_size/beta_size_2-5/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>5</td>
<td><img src="result/beta_size/beta_size_5/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/beta_size/beta_size_5/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
<tr>
<td>10</td>
<td><img src="result/beta_size/beta_size_10/result/figure/overpotential_vs_ni_fraction_iter0-4.png" width="400"></td>
<td><img src="result/beta_size/beta_size_10/result/figure/boxplots_comparison_iter0-4.png" width="600"></td>
</tr>
</table>

### iter毎の最小過電圧の推移

<img src="result/analysis/min_overpotential_comparison_beta.png" width="500">

| β値 | iter0 | iter1 | iter2 | iter3 | iter4 | iter5 | Δ(iter5 - iter0) |
|-----|-------|-------|-------|-------|-------|-------|-------------------|
| 0.1 | 0.297 | 0.290 | 0.297 | 0.255 | 0.283 | 0.286 | -0.011 |
| 0.5 | 0.297 | 0.285 | 0.286 | 0.289 | 0.283 | 0.283 | -0.014 |
| 1   | 0.297 | 0.332 | 0.285 | 0.287 | 0.293 | 0.282 | -0.015 |
| 2.5 | 0.297 | 0.285 | 0.297 | 0.282 | 0.282 | 0.290 | -0.007 |
| 5   | 0.297 | 0.282 | 0.285 | 0.285 | 0.276 | 0.282 | -0.015 |
| 10  | 0.297 | 0.366 | 0.335 | 0.339 | 0.323 | 0.278 | -0.019 |


## 考察
- 定量的な評価ではなく目視となってしまうが、latent_sizeは32~64以下、beta_sizeは1.0以下が、潜在空間をにデータが均一に分布している。

- 単体のPtが元々過電圧が低いため、VAEによる触媒生成の効果が見にくいが、どのモデルもおおよそ過電圧が0.02V前後改善していることが確認できる。

→そのため、指摘にもあった通り、問題設定としては、Pd-Ni系やAu-Ni系の方が、過電圧改善の効果が見やすく適している可能性はある。(Pd系やAu系のFC触媒開発のreviewを調べます)
