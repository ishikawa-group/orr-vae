---
title: "Optimizing Activity and Stability of Alloy ORR Catalysts Using Conditional Variational Autoencoder and Machine Learning Interatomic Potential" 
author: "Name"
bibliography: [orr-vae.bib]
csl: american-chemical-society.csl
---

## Abstract

In this study, we have developed a method to simultaneously optimize the activity and stability of alloy catalysts using a conditional variational autoencoder (cVAE) and universal neural network potential.

This method was applied to the design of Pt–Ni alloy catalysts for the oxygen reduction reaction (ORR).

cVAE generated new structures by learning both overpotential (η) based on the computational hydrogen electrode (CHE) and alloy formation energy (E_form) as conditional labels.

The distribution of η and E_form shifted toward lower potentials and more stable values by generating and evaluating 768 structures across six iterations.

And visualizing data obtained in each iteration using the learned VAE encoder suggested that structures were generated in data spaces not included in the initial dataset, and that data with different structural properties from the initial dataset were generated.

This method is able to efficiently optimize alloy catalysts by extrapolatively generating structures that satisfy both activity and stability from limited initial data.

---


## 1. Introduction
Proton exchange membrane fuel cells (PEMFCs) are known as a clean power generation technology that can use hydrogen from renewable energy sources.  
[@debeElectrocatalystApproachesChallenges2012]

However, the rate of the oxygen reduction reaction (ORR) at the cathode limits practical application, and the development of highly active electrode catalysts is a challenge.  
[@gittlemanMaterialsResearchDevelopment2019][@cullenNewRoadsChallenges2021]

Currently, platinum (Pt) is the main commercial catalyst, but due to cost and rarity, alloying Pt with inexpensive elements is an effective for balancing performance and resource constraints.  
[@huangAdvancedPlatinumBasedOxygen2021][@greeleyAlloysPlatinumEarly2009][@zhangRecentAdvancesPtbased2021]

Especially, Pt–Ni is one of the representative ORR alloy catalysts that has been widely studied.  
[@wangFundamentalComprehensionRecent2021][@tianEngineeringBunchedPtNi2019]

And it is known that the Pt-skin structure which has a high Pt content on the catalyst surface is effective for improving the catalytic activity of Pt-Ni to optimize the d-band center.  
[@stamenkovicImprovedOxygenReduction2007][@shinDensityFunctionalTheory2021][@kumedaInterfacialStructurePtNi2017][@limRoleTransitionMetals2023]

However, there are differences between reports about the optimal composition and arrangement for improving activity and stability, and it is not yet fully understood.  
[@yangMethanolTolerantOxygen2005][@carpenterSolvothermalSynthesisPlatinum2012][@xiaUnveilingCompositionDependentCatalytic2024]

Such catalyst design for the optimization of alloy catalysts is effective using screening with computational methods like a density functional theory (DFT) calculations.  

[@shambhawiDesignOptimizationHeterogeneous2024]

Also, descriptor based machine learning screening is effective for determining important features and predicting properties.  
[@shamekhiHighthroughputScreeningDFT2025][@hartMachineLearningAlloys2021][@sharmaMachineLearningGuidedDiscovery2025][@yinMachinelearningacceleratedDesignHighperformance2024][@lucchettiRevolutionizingORRCatalyst2024]


But model training requires a sufficient amount of data for each task, and exploring new structures needs a lot of inferences using the trained model.

therefore, the use of generative models has been proposed for inverse design from material properties to structures, and is expected to be an effective approach for generating structures from unknown chemical data spaces.  
[@hellmanBriefOverviewDeep2025][@parkHasGenerativeArtificial2024]

In particular, the iterative process of structure generation by generative models and evaluation by DFT calculations has shown to enable the proposal of extrapolative candidates of alloy catalyst structure not included in initial data sets for alloy catalyst design.  
[@ishikawaHeterogeneousCatalystDesign2022]

But the workflow of evaluating proposed structures from generative models by DFT calculations each time becomes rapidly inefficient with increasing amount of data.

To solve this problem, using a universal neural network potential (NNP) trained by first-principles calculation data can accelerate the iteration of generation and evaluation while maintaining accuracy close to DFT.  
[@hisamaTheoreticalCatalystScreening2024]

These studies have used generative adversarial networks (GANs) as generative models, although variational autoencoders (VAEs), which are also general generative models, have been used for the design of bulk materials.  

The typical Generative adversarial networks (GANs) can sometimes be unstable during training when the amount of training data is small.  
[@liComprehensiveSurveyDataEfficient2022][@zhaoDifferentiableAugmentationDataEfficient2020][@karrasTrainingGenerativeAdversarial2020]

This suggests that GAN use may be limited when a sufficient amount of data cannot be prepared computationally or experimentally.  

While, VAEs are relatively easy to train and stabilize compared to GANs, and they can easily generate candidates from unobserved data spaces because they have a regularized continuous latent space.
[@bajpaiScalableCrystalRepresentation2023]

Additionally, it has been reported that conditional learning allows for high-precision generation of specified label (class) data.  
[@turkAssessingDeepGenerative2022]

Thus, this study proposes a workflow that integrates fast evaluation using a universal NNP and structure generation using VAE for the optimization of Pt–Ni alloy catalysts for ORR.

Specifically, new structures are generated using a conditional variational autoencoder (cVAE) conditioned by ORR overpotential based on the computational hydrogen electrode (CHE) and alloy formation energy. 

The generated structures are evaluated using NNP.

By repeating this process, we efficiently explore Pt–Ni catalysts that have high activity and stability.

---






## 2. Methods

We used the uma-s-1p1 model as a universal neural network potential (NNP) for structure optimization and energy calculations.  
[@woodUMAFamilyUniversal2025]

uma-s-1p1 is pre-trained by about 500M DFT calculation data and includes Mixture of Linear Experts (MoLE) into the equivariant Smooth Energy Network (eSEN).

And that is reported accuracy of MAE 24.9 meV/atom for the evaluation test of formation energy of high-entropy alloys (HEAs) and MAE 68.8 meV for the evaluation test of adsorption energy using the OC20 dataset.

All spin-polarized DFT calculations were performed using the Vienna Ab initio Simulation Package (VASP 6.5.1) with the GGA-RPBE exchange-correlation functional and the projector augmented-wave (PAW) method.  
[@kresseEfficientIterativeSchemes1996][@kresseUltrasoftPseudopotentialsProjector1999]

The valence electrons were expanded with a plane-wave cutoff of 450 eV, and a Methfessel–Paxton smearing of width 0.20 eV was applied.

The convergence threshold for self-consistent calculations was set to 1×10^{-5} eV, and the Brillouin zone sampling was performed using Monkhorst–Pack grids with a bulk 2×2×2, slab 2×2×1, and the Γ point for gas-phase molecules.

Each slab consists of (4×4) 4-layer structures made from fcc(111) and the bottom 2 layers fixed and the top 2 layers and adsorbates relaxed with full degrees of freedom.

z-axis is applied with a vacuum of 15 Å, and a dipole correction was applied for DFT calculations.

And the gas-phase references H2 and H2O were optimized in a 15 Å cubic cell.

The convergence threshold for structure optimization with NNP and DFT calculations was set to 0.05 eV/Å.

For adsorption systems, OOH, O, and OH were initially placed on ontop/bridge/fcc-hollow/hcp-hollow sites searched for the most stable sites.

Cell size was optimized before applying the vacuum layer to slab.

Also we consistently used NNP (uma-s-1p1) for the evaluation of η and E_form in the iterative loop, while DFT was used for the validation of NNP and the analysis of represent structures.

---




## 2.2 Oxygen Reduction Reaction

In this study, we evaluated the ORR activity based on the theoretical limiting potential U_L and overpotential η using the computational hydrogen electrode (CHE) proposed by J. K. Nørskov et al.   
[@norskovOriginOverpotentialOxygen2004]

We used a four-electron reaction mechanism under acidic conditions, represented by the following four reactions:

1. O2 + H+ + e− + * → OOH*
2. OOH* + H+ + e− → O* + H2O
3. O* + H+ + e− → OH*
4. OH* + H+ + e− → H2O + *


$$\Delta G_i(U)=\Delta G_i(0)+eU$$


$$U_L=-\max_i\left[\frac{\Delta G_i(0)}{e}\right],\quad \eta=1.23-U_L\;\mathrm{[V]}$$



Then ΔG_i(0) is calculated by adding zero-point energy (ZPE) and vibrational entropy (TS, T = 298.15 K) based on literature values to DFT calculated energies.  
[@norskovOriginOverpotentialOxygen2004]

And we calculated the gas-phase O2 energy using the energy of H2, H2O and the free energy change for water formation that is 2.46 eV to avoid known errors.

Also, we introduced solvent effects as constant corrections based on literature values, applying 0.00, 0.28, and 0.57 eV to O*, OOH*, and OH*.  
[@zhangSolvationEffectsDFT2019][@heImportanceSolvationAccurate2017]

---


1. O2 + H+ + e− + * → OOH*
2. OOH* + H+ + e− → O* + H2O
3. O* + H+ + e− → OH*
4. OH* + H+ + e− → H2O + *


$$\Delta G_i(U)=\Delta G_i(0)-eU$$


$$U_L=\min_i\left[\frac{\Delta G_i(0)}{e}\right],\quad \eta=1.23-U_L\;\mathrm{[V]}$$



$$
E_{\mathrm{form}} = E_{\mathrm{bulk}}^{\mathrm{alloy}} - \sum_i N_i \; \frac{E_{\mathrm{bulk}}(i)}{N_{\mathrm{bulk}}(i)}.
$$


## 2.4 Variational Auto-Encoder

### 2.4.1 Structure Representation
Catalyst structures were represented as a (4, 8, 8) tensor mapping each of the four layers of the fcc(111) surface to a grid.

Each layer was converted into a two-dimensional matrix encoded the occupancy of blank, Ni, and Pt for each element.


Thus we can treat layer stacking and in-plane alloy arrangements.

<img src="fig/structure_tensor_2.svg" alt="Structure tensor representation" style="background-color: white; width: 50%;">

Figure 1. Structure representation of the catalyst slab as a (4, 8, 8) tensor map each fcc(111) layer to a 2D grid.

### 2.4.2 VAE Architecture
We used conditional VAE for learning and generating catalyst structures.  
[@kingmaAutoEncodingVariationalBayes2022][@kingmaSemiSupervisedLearningDeep2014][@sohnLearningStructuredOutput2015]

And we augmented the (4, 8, 8) structure tensor with binary labels that summarize the overpotential and alloy formation energy performance.

At iteration k we aggregated all structures available up to that point (iter0–k), ranked them by overpotential (ascending; lower values are better) and by formation energy (ascending; more negative values indicate higher stability), and reassigned the labels. The top 30% in each ranking received a label value of 1, whereas the remaining 70% received 0. This percentile-based assignment was recomputed at every iteration so that the conditional labels reflected the latest performance distribution.

Consequently each structure carries one of four condition vectors—(1,1), (1,0), (0,1), or (0,0)—depending on whether it met the targets for low overpotential and/or high stability.

The VAE encoder and decoder were based on convolutional blocks, and has 32-dimension latent space.

The conditional label were combined to the encoder and decoder after the embedding from fully connected layers.

The shape of decoder output is to (12, 8, 8) for 4 layers and 3 classes. 

During structure generation, it is reshaped to (4, 8, 8) class labels using the softmax function.

<img src="fig/vae_simple_2.svg" alt="VAE Architecture" style="background-color: white; width: 50%;">

Figure 2. Conditional convolutional VAE architecture: label embeddings are combined into encoder and decoder; the decoder outputs logits of shape (12, 8, 8) corresponding to 4 layers × 3 classes.

### 2.4.3 Training Process
The VAE training is conducted using the integrated data until the iteration performed, and was divided into training and evaluation sets in a 9:1 ratio.

The training optimizer was AdamW with a learning rate of 2×10^{-4} and a weight decay of 1×10^{-4}.

And batch size was 16, and training was performed for 200 epochs.

The loss function combined the weighted multi-class cross-entropy for the 3 classes (blank, Ni, Pt) for each pixel in the 4 layers with the KL divergence of the latent distribution.


$$p_{bz h w, c} \,=\, \mathrm{softmax}(\hat x_{bzchw})_c$$


$$\begin{align}
\mathcal{L}_{\rm recon}
&= -\sum_{b=1}^{B}\sum_{z=1}^{4}\sum_{h=1}^{H}\sum_{w=1}^{W}
\, w_{x_{bzhw}} \, \log p_{bz h w,\, x_{bzhw}},\\
\mathcal{L}_{\rm KL}
&= -\tfrac12\sum_{b=1}^{B}\sum_{j=1}^{D}\Bigl(1+\log\sigma_{bj}^2-\mu_{bj}^2-\sigma_{bj}^2\Bigr),\\
\mathcal{L}_{\text{total}}
&= \mathcal{L}_{\rm recon} + \beta\,\mathcal{L}_{\rm KL} \qquad (\beta=2.0).
\end{align}$$


Also, we use β-VAE technique and multiply the KL divergence by a weight of β=2.0 for regularization.  
[@higginsVVAELEARNINGBASIC2017]


### 2.4.4 Structure Generation
After VAE training, we selected the condition [1,1] (low overpotential and low formation energy) and output the tensor from the trained VAE decoder.

The new catalyst structure was generated by applying the inverse structure representation.

During each iteration we generated structures from the VAE decoder until we collected 128 unique slabs, discarding any candidate whose atomic arrangement exactly matched a structure already present in the current iteration or in the accumulated dataset.

## 2.5 Iterative Loop
In this study, we conducted six iterations (iter0–5): iter0 generated 128 random Pt–Ni slabs as the seed dataset, and each of the subsequent five iterations (iter1–5) generated and evaluated 128 structures with the cVAE-guided sampling, yielding 768 structures in total.

The structure of iter 0 was generated with a random atomic arrangement, and the ORR overpotential η and alloy formation energy E_form were evaluated using NNP.

Obtained dataset (structure, η, E_form) was used to train the conditional VAE.  

After training, the VAE decoder was set to the condition of "low overpotential and low formation energy" to generate new structures.

For the generated structures, we again evaluated (η, E_form) using NNP, added them to the dataset, defined the conditional labels, and repeated the "generation → evaluation → addition → training" loop from iter1 to 5.

For VAE training, we used the Python library Pytorch.

And for structure generation and management, we used the Python library ASE.  
[@hjorthlarsenAtomicSimulationEnvironment2017]

<img src="fig/workflow.svg" alt="VAE training and structure generation workflow" style="background-color: white; width: 50%;">

Figure 3. Iterative workflow integrating structure generation with cVAE and property evaluation with NNP.

## 3. Results and Discussion

## 3.1 Accuracy of NNP

We checked using universal NNP (uma‑s‑1p1) for the Pt-Ni alloy system by comparing it with DFT calculations.

For this validation we newly generated 50 Pt–Ni slabs with the same random sampling scheme as iter0 (composition drawn from a uniform distribution and atomic positions shuffled within the (4×4) cell), ensuring a broad coverage of surface configurations, and we evaluated each structure with both NNP and spin-polarized DFT under the settings described in Section 2.3.

Figure 4 shows plots of (i) ORR overpotential η (V) and (ii) alloy formation energy E_form (eV/atom) for the Pt–Ni alloy structure, with DFT on the x-axis and NNP on the y-axis.

Each plot shows a high Spearman rank correlation coefficient with ρ ≈ 0.985 for overpotential and ρ ≈ 0.982 for formation energy. That confirms the maintain of the order relation between the data.

And the absolute errors were MAE ≈ 0.060 V and MAE ≈ 0.007 eV/atom.

The η plots are almost match in the wide range (about 0.5–1.8 V).

E_form has a tendency to underestimate on the unstable side in the region of less than -0.03 eV/atom, but the error is about 0.01 eV/atom which is able to use for screening purposes.

Therefore, we judged that the property evaluation by NNP is valid in this study's workflow.

---




$$
E_{\mathrm{form}} = E_{\mathrm{bulk}}^{\mathrm{alloy}} - \sum_i N_i \; \frac{E_{\mathrm{bulk}}(i)}{N_{\mathrm{bulk}}(i)}.
$$



Figure 4. Parity plot of overpotential (V) and alloy formation energy (eV/atom) between DFT (x) and NNP (y).

<img src="fig/overpotential_and_formation.png" alt="Parity of overpotential and formation energy: DFT vs NNP" style="width: 80%; background-color: white;">


## 3.2 Iterative Improvement and Data Distributions

We confirmed that the distributions of overpotential and alloy formation energy shifted toward the high-performance side across six iterations of the generation and evaluation process, where iter0 is the random initialization and iter1–5 are cVAE-guided updates.

Violin plots in Figure 5 show that the overpotential distribution shifts toward the high-activity side, and the formation energy distribution moves toward a more stable side with iteration progresses, 

The mean of overpotential decreased from 1.126 V to 0.520 V, and the mean of alloy formation energy decreased from -0.027 eV/atom to -0.047 eV/atom.

And Figure 6 shows the change in distribution, and the alloy formation energy also moves to a more stable region as the overpotential decreases.

Then we can confirm that data is generated toward the coexistence of activity and stability.

--- 



Figure 5. Distributions of overpotential and alloy formation energy (iter0–5).

<img src="fig/violin_combined_iter0-5.png" alt="Combined violin: formation energy and overpotential" style="width: 50%; background-color: white;">

Figure 6. Overpotential vs alloy formation energy (iter0–5).

<img src="fig/overpotential_vs_alloy_formation_iter0-5.png" alt="Scatter: overpotential vs alloy formation energy" style="width: 50%; background-color: white;">

## 3.3 Evolution of Catalytic Properties


Figure 7 shows volcano plots based on the CHE and linear scaling relations for the ORR, with the structures generated from each iteration.   
[@kulkarniUnderstandingCatalyticActivity2018]

Figure 8 shows a phase diagram drawn by the composition ratio and the alloy formation energy.

Figure 7 shows the x-axis as ΔG_OH and the y-axis as the limiting potential U_L, with two boundary lines obtained from CHE and scaling relations and an ideal line (U_L = 1.23 V).

The cross point is located at ΔG_OH ≈ 0.86 eV, which corresponds to the top of the volcano plot (U_L maximum).

We can confirm that the data points shift to the top as the iterations progress, and the activity approaches the theoretically optimal area. 


And Figure 8 shows a phase diagram plotted with the Ni concentration and the alloy formation energy.

The distribution tends to shift towards the more negative E_form side, with samples concentrating around region of x_Ni ≈ 0.5.

These results indicate that the high-activity region (near the vertex of Figure 7) and the thermodynamically stable region (more negative E_form) are improved by the iterative exploration.

---



Figure 7. Volcano plot: ΔG_OH vs limiting potential (iter0–5). [@kulkarniUnderstandingCatalyticActivity2018]

<img src="fig/volcano_dG_OH_vs_limiting_potential_iter0-5.png" alt="Volcano plot" style="width: 50%; background-color: white;">


Figure 8. Phase diagram: Ni fraction vs formation energy colored by iter.

  <img src="fig/phase_diagram_stability_analysis_iter0-5.png" alt="Phase diagram stability analysis" style="width: 50%; background-color: white;">

 
## 3.4 DFT Validation

We check the 16 structures obtained in iteration 5 that satisfy η < 0.60 V and E_form < −0.05 eV/atom by DFT evaluation.

The trend of overpotential was well matched between NNP and DFT, with a mean absolute error of MAE ≈ 0.018 V.

While the alloy formation energy is underestimated by NNP compared to DFT by about 0.01 eV/atom.

Also we checked Pt35Ni29 which is one of the high-activity and high-stability structures obtained in iteration 5 and visualized and compared the most stable site and adsorption reaction energies for the 3 adsorbates OH*, O*, and OOH* using NNP and DFT.

The 3 adsorbates all have the same most stable site.

Furthermore, comparing the ORR free energy diagrams from NNP and DFT in Figure 11, the limiting potential U_L values are 0.731 V and 0.756 V, an error of 0.025 V, and the rate determining steps are consistent.

---


Figure 9. Parity and property plots for DFT vs NNP: (a) overpotential, (b) alloy formation energy, and (c) overpotential vs formation energy.

<img src="fig/scatter_overpotential_DFTx_NNPy.png" alt="Parity: overpotential (DFT vs NNP)" style="width: 31%; background-color: white;"> <img src="fig/scatter_formation_energy_DFTx_NNPy.png" alt="Parity: formation energy (DFT vs NNP)" style="width: 31%; background-color: white;"> <img src="fig/overpotential_vs_formation_energy_DFT_NNP.png" alt="Overpotential vs formation energy: DFT (red) and NNP (green)" style="width: 31%; background-color: white;">


Figure 10. Adsorption structures and adsorption reaction energies.

<img src="fig/adsorption_matrix.png" alt="Adsorption structures and energies: NNP vs DFT (OH*, O*, OOH*)" style="width: 50%; background-color: white;">

Figure 11. ORR free‑energy diagrams.

<img src="fig/ORR_free_energy_diagram_NNP.png" alt="ORR free energy diagram (NNP)" style="width: 46%; background-color: white;"> <img src="fig/ORR_free_energy_diagram_DFT.png" alt="ORR free energy diagram (DFT)" style="width: 46%; background-color: white;">



$$
E_{\mathrm{ads}}(\mathrm{OH}^{\ast}) = E(\mathrm{OH}^{\ast}) - \left[ E(\ast) + E(\mathrm{H_2O}) - \tfrac{1}{2}E(\mathrm{H_2}) \right]
$$

$$
E_{\mathrm{ads}}(\mathrm{O}^{\ast}) = E(\mathrm{O}^{\ast}) - \left[ E(\ast) + E(\mathrm{H_2O}) - E(\mathrm{H_2}) \right]
$$

$$
E_{\mathrm{ads}}(\mathrm{OOH}^{\ast}) = E(\mathrm{OOH}^{\ast}) - \left[ E(\ast) + E(\mathrm{H_2O}) + \tfrac{1}{2}E(\mathrm{H_2}) \right]
$$

## 3.5 Latent-Space Visualization and Property-Colored Distributions

We visualized all data points to 2D space using t-SNE to analyze the data distribution characteristics, after reducing the dimensionality to the 32-dimensional post mean (μ) of latent space (z) learned cVAE iteration 5.

Figure 12 shows the results colored by iteration.

The initial data (iter0) is mainly distributed in the left area of the figure, while iter1 and later iterations show that the distribution spreads widely to the center and right side.

This indicates that the structural exploration and generation have expanded into areas of the data space that did not exist in iter0.

In addition, we colored each point by a parameter showing how many Pt atoms exist in the top layer compared to the lower layers, in order to investigate the influence of the surface structure on catalytic activity.

As a result, most points are close to 0 in iter0, so the number of Pt atoms in the top layer is almost equal to that in the lower layers.

While, in iter1 and later iterations, the number of points with positive values (red) increases significantly, and the top layer has significantly more Pt atoms than the lower layers (average of layers 2-4).

that suggests that structures close to the so-called Pt-skin are selectively generated.  
[@stamenkovicImprovedOxygenReduction2007][@shinDensityFunctionalTheory2021][@kumedaInterfacialStructurePtNi2017][@limRoleTransitionMetals2023]

And this is consist with the improvement of the overpotential distribution shown in Section 3.2 (shift to the low η side).

Therefore, iterative structure generation and evaluation can extend the data space that did not exist in the initial dataset.

And it was confirmed that the feature as a catalytic structure is appearing in the mapping of the data.

---



$$
\mathrm{Surface\ Pt\ excess}
\;=\;
\frac{1}{3}\sum_{k=2}^{4}\bigl[\,N_{\mathrm{Pt}}^{(\mathrm{top})}-N_{\mathrm{Pt}}^{(k)}\,\bigr],
$$



Figure 12. Latent-space visualization by t‑SNE (top: colored by iteration; bottom: colored by surface Pt excess).

<img src="fig/tsne_latent_space_iter0-5_mean.png" alt="t-SNE latent space (mean) iter0-5" style="width: 50%; background-color: white;">

<img src="fig/tsne_latent_space_pt_layer_diff_iter0-5_mean.png" alt="t-SNE latent space (mean) iter0-5" style="width: 50%; background-color: white;">

## 4. Conclusions


For the oxygen reduction reaction (ORR) on the Pt–Ni alloy surface, the cVAE was trained with conditional labels based on the computational hydrogen electrode (CHE) overpotential and alloy formation energy, and the NNP accelerated the evaluation of the generated structures.

The values of overpotential and alloy formation energy obtained by NNP were confirmed by DFT calculations to maintain the order of size relationships with high accuracy.

The distribution of η and E_form shifted toward lower potentials and more stable values by generating and evaluating 768 structures over six iterations.

The data obtained in each iteration converged toward the vicinity of the reported volcano peak and the thermodynamically stable regions of the Ni–Pt phase diagram, confirming the physicochemical plausibility of the generated structures.

Also, using the encoder of the trained cVAE, each structure was mapped to two dimensions, and it was confirmed that the data expanded into a data space that did not exist in the initial data by iterations progressed.

And the structure with a high concentration of Pt at the surface was confirmed as a meaningful feature of the catalyst structure.

This workflow demonstrates a general computational screening method that can extrapolate alloy surface structures satisfying both activity and stability from limited initial data.

---

## Data and Code Availability

- github

## References

- Debe, Mark K. Electrocatalyst Approaches and Challenges for Automotive Fuel Cells. Nature (2012) 486(7401), 43--51. DOI: 10.1038/nature11115.

- Gittleman, Craig S.; Kongkanand, Anusorn; Masten, David; Gu, Wenbin. Materials Research and Development Focus Areas for Low Cost Automotive Proton-Exchange Membrane Fuel Cells. Current Opinion in Electrochemistry (2019) 18, 81--89. DOI: 10.1016/j.coelec.2019.10.009.

- Cullen, David A.; Neyerlin, K. C.; Ahluwalia, Rajesh K.; Mukundan, Rangachary; More, Karren L.; Borup, Rodney L.; Weber, Adam Z.; Myers, Deborah J.; Kusoglu, Ahmet. New Roads and Challenges for Fuel Cells in Heavy-Duty Transportation. Nature Energy (2021) 6(5), 462--474. DOI: 10.1038/s41560-021-00775-z.

- Huang, Lei; Zaman, Shahid; Tian, Xinlong; Wang, Zhitong; Fang, Wensheng; Xia, Bao Yu. Advanced Platinum-Based Oxygen Reduction Electrocatalysts for Fuel Cells. Accounts of Chemical Research (2021) 54(2), 311--322. DOI: 10.1021/acs.accounts.0c00488.

- Greeley, J.; Stephens, I. E. L.; Bondarenko, A. S.; Johansson, T. P.; Hansen, H. A.; Jaramillo, T. F.; Rossmeisl, J.; Chorkendorff, I.; Nørskov, J. K. Alloys of Platinum and Early Transition Metals as Oxygen Reduction Electrocatalysts. Nature Chemistry (2009) 1(7), 552--556. DOI: 10.1038/nchem.367.

- Zhang, Xuewei; Li, Haiou; Yang, Jian; Lei, Yijie; Wang, Cheng; Wang, Jianlong; Tang, Yaping; Mao, Zongqiang. Recent Advances in Pt-based Electrocatalysts for PEMFCs. RSC Advances (2021) 11(22), 13316--13328. DOI: 10.1039/D0RA05468B.

- Wang, Yao; Wang, Dingsheng; Li, Yadong. A Fundamental Comprehension and Recent Progress in Advanced Pt-based ORR Nanocatalysts. SmartMat (2021) 2(1), 56--75. DOI: 10.1002/smm2.1023.

- Tian, Xinlong; Zhao, Xiao; Su, Ya-Qiong; Wang, Lijuan; Wang, Hongming; Dang, Dai; Chi, Bin; Liu, Hongfang; Hensen, Emiel J.M.; Lou, Xiong Wen (David); Xia, Bao Yu. Engineering Bunched Pt-Ni Alloy Nanocages for Efficient Oxygen Reduction in Practical Fuel Cells. Science (2019) 366(6467), 850--856. DOI: 10.1126/science.aaw7493.

- Stamenkovic, Vojislav R.; Fowler, Ben; Mun, Bongjin Simon; Wang, Guofeng; Ross, Philip N.; Lucas, Christopher A.; Marković, Nenad M. Improved Oxygen Reduction Activity on Pt3Ni(111) via Increased Surface Site Availability. Science (2007) 315(5811), 493--497. DOI: 10.1126/science.1135941.

- Shin, Dong Yun; Shin, Yeon-Jeong; Kim, Min-Su; Kwon, Jeong An; Lim, Dong-Hee. Density Functional Theory-Based Design of a Pt-skinned PtNi Catalyst for the Oxygen Reduction Reaction in Fuel Cells. Applied Surface Science (2021) 565, 150518. DOI: 10.1016/j.apsusc.2021.150518.

- Kumeda, Tomoaki; Otsuka, Naoto; Tajiri, Hiroo; Sakata, Osami; Hoshi, Nagahiro; Nakamura, Masashi. Interfacial Structure of PtNi Surface Alloy on Pt(111) Electrode for Oxygen Reduction Reaction. ACS Omega (2017) 2(5), 1858--1863. DOI: 10.1021/acsomega.7b00301.

- Lim, Chaewon; Fairhurst, Alasdair R.; Ransom, Benjamin J.; Haering, Dominik; Stamenkovic, Vojislav R. Role of Transition Metals in Pt Alloy Catalysts for the Oxygen Reduction Reaction. ACS Catalysis (2023) 13(22), 14874--14893. DOI: 10.1021/acscatal.3c03321.

- Yang, Hui; Coutanceau, Christophe; Léger, Jean-Michel; Alonso-Vante, Nicolas; Lamy, Claude. Methanol Tolerant Oxygen Reduction on Carbon-Supported Pt-Ni Alloy Nanoparticles. Journal of Electroanalytical Chemistry (2005) 576(2), 305--313. DOI: 10.1016/j.jelechem.2004.10.026.

- Carpenter, Michael K.; Moylan, Thomas E.; Kukreja, Ratandeep Singh; Atwan, Mohammed H.; Tessema, Misle M. Solvothermal Synthesis of Platinum Alloy Nanoparticles for Oxygen Reduction Electrocatalysis. Journal of the American Chemical Society (2012) 134(20), 8535--8542. DOI: 10.1021/ja300756y.

- Xia, Zhiyuan; Zhu, Xiangyu; Chen, Xing; Zhou, Yanan; Luo, Qiquan; Yang, Jinlong. Unveiling the Composition-Dependent Catalytic Mechanism of Pt-Ni Alloys for Oxygen Reduction: A First-Principles Study. The Journal of Physical Chemistry Letters (2024) 15(38), 9566--9574. DOI: 10.1021/acs.jpclett.4c02164.

- Shambhawi; Mohan, Ojus; Choksi, Tej S.; Lapkin, Alexei A. The Design and Optimization of Heterogeneous Catalysts Using Computational Methods. Catalysis Science & Technology (2024) 14(3), 515--532. DOI: 10.1039/D3CY01160G.

- Shamekhi, Mehri; Toghraei, Arsham; Guay, Daniel; Peslherbe, Gilles H. High-Throughput Screening and DFT Characterization of Bimetallic Alloy Catalysts for the Nitrogen Reduction Reaction. Physical Chemistry Chemical Physics (2025) Advance Article. DOI: 10.1039/D5CP01094B.

- Hart, Gus L. W.; Mueller, Tim; Toher, Cormac; Curtarolo, Stefano. Machine Learning for Alloys. Nature Reviews Materials (2021) 6(8), 730--755. DOI: 10.1038/s41578-021-00340-w.

- Sharma, Rahul Kumar; Minhas, Harpriya; Pathak, Biswarup. Machine Learning-Guided Discovery of Alloy Nanoclusters: Steering Morphology-Based Activity and Selectivity Relationships in Bifunctional Electrocatalysts. ACS Applied Materials & Interfaces (2025) 17(28), 40488--40498. DOI: 10.1021/acsami.5c07198.

- Yin, Peng; Niu, Xiangfu; Li, Shuo-Bin; Chen, Kai; Zhang, Xi; Zuo, Ming; Zhang, Liang; Liang, Hai-Wei. Machine-Learning-Accelerated Design of High-Performance Platinum Intermetallic Nanoparticle Fuel Cell Catalysts. Nature Communications (2024) 15(1), 415. DOI: 10.1038/s41467-023-44674-1.

- Lucchetti, Lanna E. B.; De Almeida, James M.; Siahrostami, Samira. Revolutionizing ORR Catalyst Design through Computational Methodologies and Materials Informatics. EES Catalysis (2024) 2(5), 1037--1058. DOI: 10.1039/d4ey00104d.

- Hellman, Anders. A Brief Overview of Deep Generative Models and How They Can Be Used to Discover New Electrode Materials. Current Opinion in Electrochemistry (2025) 49, 101629. DOI: 10.1016/j.coelec.2024.101629.

- Park, Hyunsoo; Li, Zhenzhu; Walsh, Aron. Has Generative Artificial Intelligence Solved Inverse Materials Design?. Matter (2024) 7(7), 2355--2367. DOI: 10.1016/j.matt.2024.05.017.

- Ishikawa, Atsushi. Heterogeneous Catalyst Design by Generative Adversarial Network and First-Principles Based Microkinetics. Scientific Reports (2022) 12(1), 11657. DOI: 10.1038/s41598-022-15586-9.

- Hisama, Kaoru; Ishikawa, Atsushi; Aspera, Susan Menez; Koyama, Michihisa. Theoretical Catalyst Screening of Multielement Alloy Catalysts for Ammonia Synthesis Using Machine Learning Potential and Generative Artificial Intelligence. The Journal of Physical Chemistry C (2024) 128(44), 18750--18758. DOI: 10.1021/acs.jpcc.4c04018.

- Li, Ziqiang; Xia, Beihao; Zhang, Jing; Wang, Chaoyue; Li, Bin. A Comprehensive Survey on Data-Efficient GANs in Image Generation. arXiv 2204.08329 (2022). DOI: 10.48550/arXiv.2204.08329.

- Zhao, Shengyu; Liu, Zhijian; Lin, Ji; Zhu, Jun-Yan; Han, Song. Differentiable Augmentation for Data-Efficient GAN Training. arXiv 2006.10738 (2020). DOI: 10.48550/arXiv.2006.10738.

- Karras, Tero; Aittala, Miika; Hellsten, Janne; Laine, Samuli; Lehtinen, Jaakko; Aila, Timo. Training Generative Adversarial Networks with Limited Data. arXiv 2006.06676 (2020). DOI: 10.48550/arXiv.2006.06676.

- Bajpai, Rochan; Shukla, Atharva; Kumar, Janish; Tewari, Abhishek. A Scalable Crystal Representation for Reverse Engineering of Novel Inorganic Materials Using Deep Generative Models. Computational Materials Science (2023) 230, 112525. DOI: 10.1016/j.commatsci.2023.112525.

- Türk, Hanna; Landini, Elisabetta; Kunkel, Christian; Margraf, Johannes T.; Reuter, Karsten. Assessing Deep Generative Models in Chemical Composition Space. Chemistry of Materials (2022) 34(21), 9455--9467. DOI: 10.1021/acs.chemmater.2c01860.

- Wood, Brandon M.; Dzamba, Misko; Fu, Xiang; Gao, Meng; Shuaibi, Muhammed; Barroso-Luque, Luis; Abdelmaqsoud, Kareem; Gharakhanyan, Vahe; Kitchin, John R.; Levine, Daniel S.; Michel, Kyle; Sriram, Anuroop; Cohen, Taco; Das, Abhishek; Rizvi, Ammar; Sahoo, Sushree Jagriti; Ulissi, Zachary W.; Zitnick, C. Lawrence. UMA: A Family of Universal Models for Atoms. arXiv 2506.23971 (2025). DOI: 10.48550/arXiv.2506.23971.

- Kresse, G.; Furthmüller, J. Efficient Iterative Schemes for Ab Initio Total-Energy Calculations Using a Plane-Wave Basis Set. Physical Review B (1996) 54(16), 11169--11186. DOI: 10.1103/PhysRevB.54.11169.

- Kresse, G.; Joubert, D. From Ultrasoft Pseudopotentials to the Projector Augmented-Wave Method. Physical Review B (1999) 59(3), 1758--1775. DOI: 10.1103/PhysRevB.59.1758.

- Nørskov, J. K.; Rossmeisl, J.; Logadottir, A.; Lindqvist, L.; Kitchin, J. R.; Bligaard, T.; Jónsson, H. Origin of the Overpotential for Oxygen Reduction at a Fuel-Cell Cathode. The Journal of Physical Chemistry B (2004) 108(46), 17886--17892. DOI: 10.1021/jp047349j.

- Zhang, Qiang; Asthagiri, Aravind. Solvation Effects on DFT Predictions of ORR Activity on Metal Surfaces. Catalysis Today (2019) 323, 35--43. DOI: 10.1016/j.cattod.2018.07.036.

- He, Zheng-Da; Hanselman, Selwyn; Chen, Yan-Xia; Koper, Marc T. M.; Calle-Vallejo, Federico. Importance of Solvation for the Accurate Prediction of Oxygen Reduction Activities of Pt-Based Electrocatalysts. The Journal of Physical Chemistry Letters (2017) 8(10), 2243--2246. DOI: 10.1021/acs.jpclett.7b01018.

- Kingma, Diederik P.; Welling, Max. Auto-Encoding Variational Bayes. arXiv 1312.6114 (2022). DOI: 10.48550/arXiv.1312.6114.

- Kingma, Diederik P.; Rezende, Danilo J.; Mohamed, Shakir; Welling, Max. Semi-Supervised Learning with Deep Generative Models. arXiv 1406.5298 (2014). DOI: 10.48550/arXiv.1406.5298.

- Sohn, Kihyuk; Yan, Xinchen; Lee, Honglak. Learning Structured Output Representation Using Deep Conditional Generative Models. MIT Press (2015).

- Higgins, Irina; Matthey, Loic; Pal, Arka; Burgess, Christopher; Glorot, Xavier; Botvinick, Matthew; Mohamed, Shakir; Lerchner, Alexander. β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK. 2017.

- Hjorth Larsen, Ask; Jørgen Mortensen, Jens; Blomqvist, Jakob; Castelli, Ivano E; Christensen, Rune; Dułak, Marcin; Friis, Jesper; Groves, Michael N; Hammer, Bjørk; Hargus, Cory; Hermes, Eric D; Jennings, Paul C; Bjerre Jensen, Peter; Kermode, James; Kitchin, John R; Leonhard Kolsbjerg, Esben; Kubal, Joseph; Kaasbjerg, Kristen; Lysgaard, Steen; Bergmann Maronsson, Jón; Maxson, Tristan; Olsen, Thomas; Pastewka, Lars; Peterson, Andrew; Rostgaard, Carsten; Schiøtz, Jakob; Schütt, Ole; Strange, Mikkel; Thygesen, Kristian S; Vegge, Tejs; Vilhelmsen, Lasse; Walter, Michael; Zeng, Zhenhua; Jacobsen, Karsten W. The Atomic Simulation Environment-a Python Library for Working with Atoms. Journal of Physics: Condensed Matter (2017) 29(27), 273002. DOI: 10.1088/1361-648X/aa680e.

- Kulkarni, Ambarish; Siahrostami, Samira; Patel, Anjli; Nørskov, Jens K. Understanding Catalytic Activity Trends in the Oxygen Reduction Reaction. Chemical Reviews (2018) 118(5), 2302--2312. DOI: 10.1021/acs.chemrev.7b00488.


## Supplementary Information


<img src="fig/overpotential_vs_alloy_formation_ni_fraction_heatmap_iter0-5.png" alt="Heatmap: overpotential vs alloy formation with Ni fraction" style="width: 50%; background-color: white;">

  
  - Overpotential histogram (iter0–5)
  
  <img src="fig/overpotential_histogram_iter0-5.png" alt="Histogram of overpotential" style="width: 50%; background-color: white;">


  - Alloy-formation histogram (iter0–5)
  
  <img src="fig/alloy_formation_histogram_iter0-5.png" alt="Histogram of alloy formation energy" style="width: 50%; background-color: white;">

  
  
  <img src="fig/volcano_dG_OH_vs_limiting_potential_ni_heatmap_iter0-5.png" alt="Volcano Ni heatmap" style="width: 50%; background-color: white;">


  
  <img src="fig/volcano_dG_OH_vs_limiting_potential_pt_heatmap_iter0-5.png" alt="Volcano Pt heatmap" style="width: 50%; background-color: white;">

  
  - Structure lineup (t-SNE interpolation)
  
  <img src="fig/tsne_latent_space_iter_mean_LERP_LERP.png" alt="Latent interpolation structure lineup" style="width: 90%; background-color: white;">
  
  - Start/end highlight on t-SNE map
  
  <img src="fig/tsne_latent_space_iter_mean_LERP_LERP_arrow.png" alt="t-SNE interpolation endpoints" style="width: 70%; background-color: white;">
