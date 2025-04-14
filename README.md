# ORR_catalyst_generator

## 概要

ORR（酸素還元反応）用の合金触媒を生成する解釈性の高いAIを開発する

- 触媒は過電圧で評価
- Conditional Convolutional VAE (CCVAE) モデルを使用
- GradCAMでデコーダーから生成される表面構造のattention mapを取得
- 潜在空間の解析により生成構造と過電圧の関係を可視化

## 参考

- [DFT-GAN](https://github.com/atsushi-ishikawa/dft_gan/tree/master)
- [kinetics](https://github.com/ishikawa-group/kinetics/main)

## 今後

- 入力データの形式を確認
- エネルギー計算時の分子配置・配向を確認
- DFTでのエネルギー計算の実装
- CCVAEの学習収束の確認
- 潜在空間の可視化と過電圧評価の実施
