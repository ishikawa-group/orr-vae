#!/usr/bin/env python
"""Legacy entrypoint retained for guidance."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "This entrypoint is deprecated.\n"
        "Random initial dataset generation is now example-specific.\n"
        "Use one of:\n"
        "  - examples/Pt-Ni/code/initial_generation.py\n"
        "  - examples/Pt-Ni_Pt-Ti_Pt-Y/code/initial_generation.py\n"
        "Or run the full loop from example scripts:\n"
        "  - examples/Pt-Ni/script/run_iterative_screening.sh\n"
        "  - examples/Pt-Ni_Pt-Ti_Pt-Y/script/run_iterative_screening.sh"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
