#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
code/run.sh is deprecated.

The full loop is now example-specific and starts with example-local initial generation:
  - examples/Pt-Ni/script/run_iterative_screening.sh
  - examples/Pt-Ni_Pt-Ti_Pt-Y/script/run_iterative_screening.sh

Each example has Python settings under examples/*/code/settings.py.
MSG

exit 2
