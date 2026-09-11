"""Single-use finalizer for the V5 confirmatory benchmark."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11f import finalize_v5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(finalize_v5(ROOT, args.gene))
