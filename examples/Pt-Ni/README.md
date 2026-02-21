# Pt-Ni example

Pt-Ni example is now driven by Python config files under `code/`.

## Main flow

1. `code/initial_generation.py` creates initial `iter0_structures.json` (128 structures)
2. loop for `iter0..iter5`:
   - ORR evaluation (`calc-orr run-all`)
   - CVAE training (`train-cvae`)
   - structure generation (`generate-structures`, 128 structures/iter)

Default slab size for both initial and generated structures is `4x4x6`
(`GRID_X=4`, `GRID_Y=4`, `GRID_Z=6`).

During ORR evaluation, alloy formation-energy reference cache is written to
`${DATA_DIR}/{calculator}_bulk_data.json` (typically `fairchem_bulk_data.json`).
The first run creates this cache automatically.

## Run locally

```bash
cd examples/Pt-Ni/code
python3 run_iterative_screening.py
```

## Batch submission

```bash
cd examples/Pt-Ni/script
python3 submit_all_jobs.py --seed 0 --dry-run
python3 submit_all_jobs.py --seed 0
```

`submit_all_jobs.py` reads `../code/condition_list.csv` and submits
`run_tsubame_vae_parametric.sh`, which executes `../code/run_iterative_screening.py`.
