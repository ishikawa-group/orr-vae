# Pt-Ni example

This example runs iterative ORR screening for Pt-Ni up to `iter5`.

## Structure

- `code/`: configuration templates
- `script/`: runnable shell scripts
- `results/`: output directory (gitignored content)

## Quick start

```bash
cd examples/Pt-Ni/script
bash run_iterative_screening.sh
```

The script executes:

- random generation
- `(overpotential + formation-energy -> CVAE training -> structure generation)`
- loop for `iter0..iter5`
