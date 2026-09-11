"""Archive the active v4 cache namespace before the v5 rerun."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11e import archive_active_stack_v4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(archive_active_stack_v4(ROOT, args.gene))
