---
title: "Theoretical Generation of Catalysts for Oxygen Reduction Reaction with Machine Learning Potential and Variational Auto-Encoder"
author: "Name"
bibliography: [orr-vae.bib]
csl: american-chemical-society.csl
---

## 1. Introduction

- Proton Exchange Membrane Fuel Cells (PEMFCs) are effective for addressing energy challenges.
- However, the oxygen reduction reaction (ORR) at the cathode is sluggish.
- Consequently, Pt catalysts are employed at the cathode electrode, but their high cost impedes widespread PEMFC adoption.
- Therefore, development of catalysts that balance electrode activity and material cost is essential.

- Recent research has focused on developing Pt-based binary alloy (Pt-M) catalysts.
- For instance, Pt-Ni and Pt-Co alloys have demonstrated higher ORR activity than pure Pt catalysts, suggesting the possibility of reducing Pt usage while enhancing catalytic performance.

- However, designing alloy catalysts requires optimization of elemental composition and atomic arrangement, which is challenging to investigate comprehensively through either experimental or theoretical approaches.
- Thus, machine learning approaches have been proposed to efficiently explore catalyst compositions and configurations, with generative artificial intelligence enabling extrapolative generation and optimization of heterogeneous catalyst compositions and arrangements even with limited initial datasets.
- Additionally, neural network potentials are being considered for dataset generation for machine learning applications, offering accuracy comparable to first-principles calculations while substantially reducing computational costs.

- In this research, we propose a methodology for designing alloy catalysts using neural network potentials for dataset generation and Variational Auto-Encoder (VAE) as a generative AI approach.
- Using Pt-Ni alloy as a target system, we employ conditional VAE to explore and generate catalysts that consider both ORR activity and Pt usage, beyond what is available in the randomly generated initial dataset.


## 2. Methods

## 2.1 Oxygen Reduction Reaction

- For evaluation of ORR catalytic activity, we utilize theoretical overpotential calculated using the computational hydrogen electrode (CHE) model proposed by Nørskov and colleagues.
- The theoretical overpotential was calculated assuming a 4-electron reaction pathway.

## 2.2 Variational Auto-Encoder

- We implement a conditional convolutional VAE that takes crystal structure information converted to tensor format as input, with ORR overpotential and Pt content in the slab model as conditional labels.

## 2.3 NNP/DFT calculation

- For dataset generation for VAE training, we employed MACE (omat-0) models finetuned on matpes dataset at PBE level of theory.

- Additionally, DFT calculations using the Vienna Ab Initio Simulation Package (VASP) were performed to verify the ORR activity of selected structures output by the VAE.


## 3. Results and Discussion

## 3.1 Calculated Theoretical Overpotentials

- (Details of the overpotential for some specific surface)

## 3.2 Generated Catalyst Surfaces

- (Details of the VAE-generated surface)
  
## 4. Conclusions

- (Describe some conclusion)

## References

- pandoc paper.md -o paper.pdf --bibliography=orr-vae.bib --csl=american-chemical-society.csl --citeproc --standalone