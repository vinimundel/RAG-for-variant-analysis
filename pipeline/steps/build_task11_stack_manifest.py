"""Emit the active retrieval stack manifest."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11 import build_stack_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(build_stack_manifest(ROOT, args.gene))
