"""Move existing Task 11 RAG outputs into the provisional namespace."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11 import provisionalize_task11_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(provisionalize_task11_outputs(ROOT, args.gene))
