"""Download the PMID-bearing open-access corpus listed in a literature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.pdf_downloader import PDFDownloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="?")
    parser.add_argument("--literature-csv", type=Path)
    parser.add_argument("--gene", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source = args.literature_csv or args.csv
    if source is None:
        parser.error("provide CSV positionally or with --literature-csv")
    print(PDFDownloader().process_literature_csv_concurrent(source, args.gene, args.workers))
