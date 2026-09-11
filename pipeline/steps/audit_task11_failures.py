"""Produce the Task 11D development-only RAG failure audit."""

from __future__ import annotations

import argparse

from pipeline.config import ROOT
from src.literature.task11_audit import audit_task11_failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(audit_task11_failures(ROOT, args.gene))
