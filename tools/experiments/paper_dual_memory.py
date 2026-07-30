#!/usr/bin/env python3
"""Compatibility entrypoint for the modular PC-DLCMNet runner."""

from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pc_dlcmnet.training.runner import main


if __name__ == "__main__":
    main()
