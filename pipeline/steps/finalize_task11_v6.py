"""Single-use finalizer for the V6 mechanistic-only benchmark."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11f2 import finalize_v6


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(finalize_v6(ROOT, args.gene))
