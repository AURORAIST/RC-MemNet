#!/usr/bin/env python3
"""Compatibility entrypoint for the 0722 RC-MemNet experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tools.experiments.run_rc_dmnet import main


if __name__ == "__main__":
    main()
