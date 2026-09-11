"""Read-only preflight for the sealed V7 benchmark."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11f2 import preflight_v7


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(preflight_v7(ROOT, args.gene))
