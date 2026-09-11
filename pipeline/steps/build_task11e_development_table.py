"""Build the development-only table from the first 120 historical labels."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11e import build_development_training_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(build_development_training_table(ROOT, args.gene))
