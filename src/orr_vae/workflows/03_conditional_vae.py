#!/usr/bin/env python
"""Compatibility shim for modules that load 03_conditional_vae.py by file path."""

from orr_vae.workflows.conditional_vae import *  # noqa: F401,F403


if __name__ == "__main__":
    from orr_vae.workflows.conditional_vae import main

    main()
