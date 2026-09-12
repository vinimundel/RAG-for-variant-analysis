from __future__ import annotations

import argparse
from pathlib import Path

from healthrag.pipeline.config import ROOT, get_gene_config, paths_for
from healthrag.data.pubmed_abstracts import download_missing_fulltext_abstracts


def run(gene: str = "BRAF", csv_path: Path | None = None) -> Path:
    paths = paths_for(gene)
    cfg = get_gene_config(gene)
    return download_missing_fulltext_abstracts(
        csv_path or ROOT / cfg["literature_csv"],
        paths["input"] / "evidence" / "pdfs",
        paths["input"] / "evidence" / "abstracts",
        gene=gene,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    print(run(args.gene, args.csv))
