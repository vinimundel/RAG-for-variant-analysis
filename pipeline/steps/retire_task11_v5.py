"""Retire the unannotated V5 benchmark before context separation."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11e import retire_v5_confirmatory_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(retire_v5_confirmatory_benchmark(ROOT, args.gene))
