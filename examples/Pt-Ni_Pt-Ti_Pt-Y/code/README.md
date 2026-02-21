# Pt-Ni_Pt-Ti_Pt-Y code settings

This directory owns mixed-alloy example logic and data templates.

- `run_workflow.py`: single entry point (initial generation + evaluate/train/generate loop)
- `solvent_correction.yaml`: solvent correction table used in ORR evaluation
- `condition_list.csv`: parameter table for batch submissions
- `generate_conditions.py`: regenerates `condition_list.csv`

Default structure counts:
- initial: 255
- each generation step: 255
