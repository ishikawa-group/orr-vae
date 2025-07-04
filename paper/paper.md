---
title: "Theoretical Generation of Catalysts for Oxygen Reduction Reaction with Machine Learning Potential and Variational Auto-Encoder"
author: "Name"
bibliography: [orr-vae.bib]
csl: american-chemical-society.csl
---

## 1. Introduction

- 固体高分子型燃料電池（PEMFC）は、エネルギー問題に対処するための効果的な手段である。
- しかし、カソードにおける酸素還元反応（ORR）が遅いことが問題となっている。
- そのため、カソード電極材料にはPt触媒が使用されているが、その高いコストがPEMFCの普及を妨げている。
- したがって、電極活性と材料コストのバランスを考慮した触媒の開発が不可欠である。

- 最近の研究では、Pt系二元合金（Pt-M）触媒の開発が進められている。
- 例えば、Pt-NiおよびPt-Co合金はPt単体よりも高いORR活性を示し、触媒性能を向上させながらPt使用量を削減できる可能性を示唆している。

- しかし、合金触媒の設計には元素組成と原子配置の最適化が必要であり、これらを実験的または理論的アプローチで網羅的に検証することは困難である。
- そのため、触媒組成と配置を効率的に探索するために機械学習によるアプローチが提案されている。
- 特に、生成型人工知能を用いることで限られた初期データセットでも合金触媒の組成と配置の外挿的な生成と最適化が可能であることが報告されている
- さらに、機械学習のための触媒特性のデータセット生成には、ニューラルネットワークポテンシャル(NNP)を用いた計算検討されており、計算コストを大幅に削減しながら第一原理計算に近い精度を提供することが可能である。

- 本研究では、データセット生成のためのNNPと生成AIアプローチとしてVariational Auto-Encoder（VAE）を用いて、効率的に合金触媒を設計する手法を提案する。
- Pt-Ni合金を対象システムとして、条件付きVAEを用いて、ランダムに生成された初期データセットから、ORR活性とPt使用量の両方を考慮した触媒の探索と生成を行う。

## 2. Methods

## 2.1 Oxygen Reduction Reaction

- ORR触媒活性の評価には、Nørskovらによって提案された計算水素電極（CHE）モデルを用いて計算された理論過電圧を利用する。
- 理論過電圧は、以下の4電子反応経路を仮定して計算した。
- (1) O2 + H+ + e− + * → OOH*
- (2) OOH* + H+ + e− → O* + H2O
- (3) O* + H+ + e− → OH*
- (4) OH* + H+ + e− → H2O + *
- そして、各ステップのギブスエネルギー変化ΔG1, ΔG2, ΔG3, ΔG4を計算し、以下の式でORR理論過電圧ηを求める。
- η = 1.23 + max(ΔG1, ΔG2, ΔG3, ΔG4) [V]

## 2.2 NNP and DFT calculation

- VAEのトレーニングのためのデータセット生成には、PBEレベルで計算されたmatpesデータセットでfine-turningされたMACE（omat-0）モデルを使用した。

- さらに、VAEによって出力された構造の一部は、Vienna Ab Initio Simulation Package（VASP）を用いたDFT計算を実行し、ORR活性を検証した。DFT計算には、交換相関汎函数としてGGA-RPBEを使用した。電子は、PAW法を用いて表現され、平面波のカットオフは000 eVとした。全ての計算には、ファンデルワールス相互作用を考慮するためにBecke-Johnson-D3補正を適用した。slabに対する計算では、分極補正とスピン分極を考慮した。


- 計算に用いたスラブモデルは、PtまたはNiの4×4のfcc(111)面によって構成される４層のスラブであり、真空層は15 Åとした。
- また、全てのスラブモデルは、真空層を追加する前に、セルサイズと原子位置の構造最適化が行われた。

- OOH*、O*、OH*におけるエネルギー計算は、スラブのontop, bridge, fcc-hollow, hcp-hollowの4つの吸着サイトにおいて実施し、最安定なサイトを選択した。
- 

## 2.3 Variational Auto-Encoder

- テンソル形式に変換された結晶構造情報を入力として、スラブモデルのORR過電圧とPt含有量を条件ラベルとする条件付き畳み込みVAEを実装した。

- テンソルへの変換は、図xのようにz軸方向のレイヤーをチャンネルとして、行列の要素間に交互に0を配置することで、fccの結晶構造に対応するように行った。また、原子位置に対応する要素には、Ni=1、Pt=2の値を設定した。

- VAEは、畳み込み層と転置畳み込み層から構成される条件付き畳み込みVAEを用いた。条件ラベルには、ORR過電圧とPt含有量を使用した。それぞれ、データセット中の中央値以上と未満に対応する0と1の値を設定した。

- 損失関数は、再構成損失とKLダイバージェンスの和であり、以下のように定義される。
- ここで、再構成誤差は、テンソルの要素ごとのクロスエントロピー損失を用いて計算される。
- KLダイバージェンスは、潜在変数の平均と分散を用いて計算される。

$$\begin{align}
\mathcal{L}_{\rm recon}
&=
\sum_{z=1}^{4}
\sum_{b=1}^{B}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
\mathrm{CE}\bigl(x_{b,z,h,w},\,\hat x_{b,z,:,h,w}\bigr),
\\
\mathcal{L}_{\rm KL}
&=
-\frac12
\sum_{b=1}^{B}
\sum_{j=1}^{D}
\Bigl(1 + \log\sigma_{b,j}^2 - \mu_{b,j}^2 - \sigma_{b,j}^2\Bigr),
\\
\mathcal{L}_{\text{VAE}}
&=
\mathcal{L}_{\rm recon}
\;+\;
\mathcal{L}_{\rm KL}
\end{align}$$



## 3. Results and Discussion

## 3.1 Generated Catalyst Surfaces

- まず、ランダムな配置によって生成された128個のPt-Ni合金の構造に対して、NNPを用いてORR過電圧を計算し、それをiter0のデータセットとして条件付きVAEのトレーニングを行った。
- VAEのトレーニング後、デコーダーを用いて、ORR過電圧とPt含有量の条件ラベルを中央値未満に設定して、128個の触媒構造を生成し、iter1のデータセットを得た。

- 過電圧 vs Pt含有量(凡例iterの図)
- iterごとの過電圧の分布、Pt含有量の分布がわかる箱ひげ図
- 縦軸iter、横軸 atom indexの図

## 3.2 Comparison with DFT
- VAEから得られた構造の一部を選択し、NNPとDFTでの自由エネルギーダイアグラムの比較。

## 3.3 Calculated Theoretical Overpotentials

- （特定の表面における過電圧の詳細）
  
## 4. Conclusions

- （結論を記述）

## References

- pandoc paper.md -o paper.pdf --bibliography=orr-vae.bib --csl=american-chemical-society.csl --citeproc --standalone

## Supplementary Information
- NNPとDFTで計算したNi, Ir, Ru, Rh, Pd, Pt, AuのORR過電圧のボルケーノプロットの比較
- NNPとDFTで計算したPtとNiの自由エネルギーダイアグラムの図