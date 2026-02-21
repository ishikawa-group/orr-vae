# orr-vae

`orr-vae` is a refactored ORR catalyst screening workflow package.
The workflow keeps the original loop semantics:

1. random slab generation
2. ORR overpotential + alloy formation-energy calculation
3. conditional VAE training
4. new structure generation
5. latent-space visualization / data analysis (optional)

## Repository layout

- `src/orr_vae/`: library and workflow modules
- `reference/`: external references (submodule)
- `examples/`: runnable Pt-Ni and Pt-Ni_Pt-Ti_Pt-Y examples
- `tests/`: lightweight regression and I/O compatibility tests
- `code/`: legacy command wrappers (`01_*.py` etc.)

## Installation

```bash
git clone git@github.com:wakamiya0315/orr-vae.git
cd orr-vae
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## New CLI

```bash
python -m orr_vae --help
python -m orr_vae run-pipeline --data_dir ./data --result_dir ./result --output_dir . --max_iter 5
```

Main subcommands:

- `generate-random`
- `calc-orr`
- `train-cvae`
- `generate-structures`
- `visualize-latent`
- `analyze`
- `run-pipeline`

## Legacy CLI compatibility

The historical scripts are preserved as wrappers under `code/`.
Existing operations such as `python code/03_conditional_vae.py ...` continue to work.

## Reference dependency

- `reference/orr-overpotential-calculator` (git submodule, `develop` branch)
- `pyproject.toml` also pins:
  - `orr-overpotential-calculator @ git+https://github.com/ishikawa-group/orr-overpotential-calculator.git@develop`
  - `fairchem-core`

## Examples

- `examples/Pt-Ni`
- `examples/Pt-Ni_Pt-Ti_Pt-Y`

Both examples include `code/`, `script/`, `results/`, and execution README.

## Testing

```bash
source .venv/bin/activate
pytest
```

Tests are lightweight and focus on compatibility and I/O invariants.
