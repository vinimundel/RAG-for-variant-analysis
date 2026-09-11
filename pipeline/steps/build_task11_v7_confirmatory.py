"""Freeze Task 11F.3 V7 from the existing V6 caches."""

from __future__ import annotations

import argparse

from pipeline.steps import build_task11_v6_confirmatory as builder


builder.CURRENT_TAG = "v7"
builder.PREVIOUS_TAG = "v5"
builder.VERSION = "task11f3_confirmatory_v7"
builder.SEED = "braf_rag_confirmatory_v3_mechanistic_only"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    parser.add_argument("--variants", type=int, default=builder.BENCHMARK_VARIANTS)
    args = parser.parse_args()
    print(builder.build(args.gene, args.variants))
