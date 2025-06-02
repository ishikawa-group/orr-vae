# ORR_catalyst_generator

## 概要

ORR（酸素還元反応）用の合金触媒を生成するAIを開発する

- 触媒は過電圧で評価
- 詳細は今後追加予定

## 参考

- [DFT-GAN](https://github.com/atsushi-ishikawa/dft_gan/tree/master)


## ゴール

Pt(111) 4 × 4 × 4 スラブをベースにNi（他の金属の可能性もある）の置換率 0–100% をランダム／学習的に変化させ、4電子ORR過電圧ηができるだけ小さい合金構造を反復的に見つける。

## ディレクトリ構成

```
ORR_catalyst_generator/
├─ code/          # 4本のステップ別スクリプト（01–04）
├─ data/          # 構造・計算結果JSONをイテレーションごとに保存
├─ models/        # 学習済みCVAE重み・学習曲線
├─ tools/         # 共通ユーティリティ（ランダムシード・構造↔️テンソル変換等）
└─ requirements.txt # 使用ライブラリ一覧
```

## 1イテレーション（手動）での4ステップ

| 順序 | スクリプト | 内容 | iter0 | iter1以降 |
|------|------------|------|-------|-----------|
| ① | 01_generate_initial.py | Ptスラブに金属を一様ランダム置換して100–200構造生成 | ○ | × |
| ② | 02_calc_overpotential.py | 各構造をDFTで計算しORR過電圧ηを取得・保存 | ○ | ○ |
| ③ | 03_train_cvae.py | （既存＋新規データ）で条件付きVAEを学習 | ○ | ○ |
| ④ | 04_generate_new.py | 学習済みCVAEから目標η条件で新構造を100–200作成 | - | ○ |

※イテレーションは3–5回を目安に手動で繰り返す。  
iter1以降は①をスキップし、④で作った構造を②→③に回す。

## データフロー

```
構造JSON ─┐
          │  (②) calc_orr_overpotential → η
ηJSON ────┤
          ▼
     学習データ              (③)
          │
          ▼
  条件付きCVAE (models/)
          │  (④) 目標η指定
          ▼
  新・構造JSON → 次イテレーションへ
```

### ファイル命名規則

- 構造保存: `data/iter{k}_structure_*.json`
- 計算結果: `data/iter{k}_results.json`
- モデル: `models/cvae_iter{k}.pth`

## スクリプトの責務

| スクリプト | 主なオプション | 出力 |
|------------|---------------|------|
| 01_generate_initial.py | --nstruct --seed | 初期構造JSON群 |
| 02_calc_overpotential.py | --iter k | η結果JSON |
| 03_train_cvae.py | --iter k --epochs | CVAE重み・学習曲線 |
| 04_generate_new.py | --iter k --target_eta --nsample --seed | 次世代構造JSON |

## データ表現ルール

- **テンソル格子**: 4層 × 8 × 8
- **サイト表現**: 0（空サイト）, 1（Ni）, 2（Pt）
- **金属マッピング**: 旧Pd(46) → Ni(28)に置換済み
- **格子定数**: Ni濃度に応じVegard法で再設定

## 再現性・設定

- 乱数シード: NumPy/random/PyTorch全てを`tools.set_random_seed`で固定
- ファイル管理: イテレーション番号とunique_idで一意管理
- DFT計算: 既存ライブラリ`calc_orr_overpotential(Atoms)`に委任
- 終了条件: 手動判断（目標η範囲達成時など）

## 実行例

### 環境構築
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### iter0
```bash
python code/01_generate_initial.py --nstruct 150 --seed 42
python code/02_calc_overpotential.py --iter 0
python code/03_train_cvae.py --iter 0 --epochs 40
python code/04_generate_new.py --iter 0 --target_eta 0.10 --nsample 150
```

### iter1（初期構造生成は不要）
```bash
python code/02_calc_overpotential.py --iter 1
python code/03_train_cvae.py --iter 1 --epochs 40
python code/04_generate_new.py --iter 1 --target_eta 0.10 --nsample 150
```

## 今後の予定

1. requirements.txtへ依存ライブラリ確定
2. NNPを用いてVAEまたはGAN用のデータセットの作成