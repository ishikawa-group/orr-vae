# Pt-Ni_Pt-Ti_Pt-Y code settings

This directory owns mixed-alloy workflow configuration and initial data generation logic.

- `settings.py`: typed runtime configuration
- `build_plan.py`: emits shell exports consumed by `script/*.sh`
- `initial_generation.py`: creates `iter0_structures.json` across Pt-Ni / Pt-Ti / Pt-Y
- `solvent_correction.yaml`: solvent correction table used in ORR evaluation
- `condition_list.csv`: parameter table for batch submissions
- `generate_conditions.py`: regenerates `condition_list.csv`

Default structure counts:
- initial: 255
- each generation step: 255
