### **Note:** This repository is currently under development and refactoring. The code and data in the paper [https://doi.org/10.1038/s41524-026-02075-0](https://doi.org/10.1038/s41524-026-02075-0) are located under `paper/`

# orr-vae

`orr-vae` is a refactored ORR catalyst screening workflow package.
Core package (`src/orr_vae`) focuses on:

- ORR overpotential + alloy formation-energy evaluation
- conditional VAE training
- structure generation from trained VAE
- latent-space visualization and analysis

Random initial dataset generation is intentionally example-specific and implemented under `examples/*/code`.

## Repository layout

- `src/orr_vae/`: reusable workflow modules (evaluation, training, generation, analysis)
- `reference/`: external references (submodule)
- `examples/`: runnable Pt-Ni and Pt-Ni_Pt-Ti_Pt-Y workflows
- `tests/`: lightweight regression checks
- `code/`: legacy wrappers / migration guidance

## Installation

```bash
git clone git@github.com:wakamiya0315/orr-vae.git
cd orr-vae
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI (src package)

```bash
python -m orr_vae --help
```

Main subcommands:

- `calc-orr`
- `train-cvae`
- `generate-structures`
- `visualize-latent`
- `analyze`

## Example-driven execution

Use example scripts for end-to-end loops (includes initial dataset generation):

- `examples/Pt-Ni/script/run_iterative_screening.sh`
- `examples/Pt-Ni_Pt-Ti_Pt-Y/script/run_iterative_screening.sh`

During ORR evaluation, alloy formation-energy references are cached per run under
each `DATA_DIR` as `{calculator}_bulk_data.json` (for example `fairchem_bulk_data.json`).
If the cache is missing, bulk references are generated automatically and then reused.

## Reference dependency

- `reference/orr-overpotential-calculator` (git submodule, `develop` branch)
- `pyproject.toml` pins:
  - `orr-overpotential-calculator @ git+https://github.com/ishikawa-group/orr-overpotential-calculator.git@develop`
  - `fairchem-core`

## Testing

```bash
source .venv/bin/activate
PYTHONPATH=src pytest
```
