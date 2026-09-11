"""Run read-only controls for the V6 separated evidence caches."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11e_control import audit_v6_controls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(audit_v6_controls(ROOT, args.gene))
