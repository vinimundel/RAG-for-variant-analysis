"""Freeze the historical filled Task 11 sheets as development-only inputs."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11_annotations import freeze_exploratory_annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(freeze_exploratory_annotations(ROOT, args.gene))
