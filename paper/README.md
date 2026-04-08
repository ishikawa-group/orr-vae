# cvae-nnp-orr-alloy-design

This repository bundles the Pt–Ni alloy ORR catalyst design workflow used in the paper
“Optimizing Activity and Stability of Alloy ORR Catalysts Using Conditional Variational
Autoencoder and Machine Learning Interatomic Potential.”
DOI: https://doi.org/10.1038/s41524-026-02075-0

## Requirements

- Python 3.11 or newer
- PyTorch 2.7.x
- ASE, NumPy, pandas, matplotlib, PyYAML
- `orr-overpotential-calculator` repository (tested with the `develop-waka` branch)
- ORR calculator backend (e.g. `mace-torch` model or another supported engine such as `fairchem`)

## Usage Overview

1. Activate your virtual environment and install the dependencies listed above.
2. Define the environment variables `DATA_DIR`, `RESULT_DIR`, `OUTPUT_DIR`, `SEED`,
   `LABEL_THRESHOLD`, `BATCH_SIZE`, `MAX_EPOCH`, and `LATENT_SIZE` as needed for your run.
3. Execute `code/run.sh` to iterate through structure generation, ORR overpotential evaluation,
   conditional VAE training, and new structure generation. You can also run each Python script
   in `code/` individually for finer control.
