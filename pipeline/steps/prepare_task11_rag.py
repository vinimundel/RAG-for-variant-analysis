"""Prepare the immutable inputs and manifests for the Task 11 rerun."""

from __future__ import annotations

import argparse
import json

from pipeline.config import ROOT
from src.literature.task11 import (
    archive_pre_curation_caches, build_stack_manifest, close_grobid_manifest,
    snapshot_pre_curation_benchmark, snapshot_pre_curation_rankings,
)


def run(gene: str = "BRAF") -> dict:
    outputs = {
        "grobid": close_grobid_manifest(ROOT, gene),
        "caches": archive_pre_curation_caches(ROOT, gene),
        "benchmark_archive": snapshot_pre_curation_benchmark(ROOT, gene),
        "ranking_archive": snapshot_pre_curation_rankings(ROOT, gene),
    }
    # The stack manifest is emitted after the caller re-emits the RAG index
    # manifest, so it is deliberately not built in this preparation step.
    return {key: str(value.relative_to(ROOT)) for key, value in outputs.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(json.dumps(run(args.gene), indent=2))
