# Pt-Ni_Pt-Ti_Pt-Y example

Mixed-alloy example is now driven by Python config files under `code/`.

## Main flow

1. `code/initial_generation.py` creates initial `iter0_structures.json` for Pt-Ni / Pt-Ti / Pt-Y (255 structures)
2. loop for `iter0..iter5`:
   - ORR evaluation (`calc-orr run-all`)
   - CVAE training (`train-cvae`)
   - structure generation (`generate-structures`, 255 structures/iter)

During ORR evaluation, alloy formation-energy reference cache is written to
`${DATA_DIR}/{calculator}_bulk_data.json` (typically `fairchem_bulk_data.json`).
The first run creates this cache automatically.

## Run locally

```bash
cd examples/Pt-Ni_Pt-Ti_Pt-Y/script
bash run_iterative_screening.sh
```

## Batch submission

```bash
cd examples/Pt-Ni_Pt-Ti_Pt-Y/script
python3 submit_all_jobs.py --seed 0 --dry-run
python3 submit_all_jobs.py --seed 0
```

`submit_all_jobs.py` reads `../code/condition_list.csv`.
