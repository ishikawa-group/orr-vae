# Pt-Ni code settings

This directory owns the Pt-Ni workflow configuration and initial data generation logic.

- `settings.py`: typed runtime configuration (seed, CVAE params, slab shape, paths)
- `build_plan.py`: emits shell exports consumed by `script/*.sh`
- `initial_generation.py`: creates `iter0_structures.json` for Pt-Ni
- `solvent_correction.yaml`: solvent correction table used in ORR evaluation
- `condition_list.csv`: parameter table for batch submissions
- `generate_conditions.py`: regenerates `condition_list.csv`

Default structure counts:
- initial: 128
- each generation step: 128
