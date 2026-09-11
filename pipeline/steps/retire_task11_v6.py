"""Archive the unannotated V6 benchmark before Task 11F.3."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11e import retire_v6_confirmatory_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(retire_v6_confirmatory_benchmark(ROOT, args.gene))
