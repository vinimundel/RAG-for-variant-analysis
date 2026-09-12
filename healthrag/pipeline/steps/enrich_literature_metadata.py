"""Restore PMID metadata for PDFs already present before a resumed download run."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from healthrag.pipeline.config import paths_for
from healthrag.data.pdf_downloader import PDFDownloader


def run(gene: str = "BRAF", workers: int = 8):
    paths = paths_for(gene)
    manifest = paths["input"] / "evidence" / f"{gene.lower()}_literature_evidence.csv"
    frame = pd.read_csv(manifest)
    mask = frame["doi"].astype(str).isin({"Cached", "N/A", "nan"}) & frame["pubmed_id"].notna()
    indices = frame.index[mask].tolist()
    downloader = PDFDownloader()

    def resolve(index):
        pmid = str(frame.loc[index, "pubmed_id"]).removesuffix(".0")
        return index, downloader.resolve_pubmed_metadata(pmid)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for completed, (index, metadata) in enumerate(executor.map(resolve, indices), 1):
            frame.loc[index, "pmcid"] = metadata.get("pmcid") or "N/A"
            frame.loc[index, "doi"] = metadata.get("doi") or "N/A"
            if completed % 50 == 0:
                print(f"Metadata enrichment: {completed}/{len(indices)}")
    frame.to_csv(manifest, index=False)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(run(args.gene, args.workers))
