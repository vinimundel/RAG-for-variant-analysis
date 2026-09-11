"""Archive active V5 caches before emitting the V6 cache namespace."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11e import archive_active_stack_v5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(archive_active_stack_v5(ROOT, args.gene))
