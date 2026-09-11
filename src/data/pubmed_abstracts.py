"""Cached PubMed abstract acquisition through the official NCBI EFetch API."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from lxml import etree

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _text(node, xpath: str) -> str:
    values = node.xpath(xpath)
    return " ".join("".join(value.itertext()) if hasattr(value, "itertext") else str(value)
                    for value in values).strip()


def _parse_pubmed_xml(payload: bytes) -> list[dict]:
    root = etree.fromstring(payload)
    records = []
    for article in root.xpath(".//PubmedArticle"):
        pmid = _text(article, ".//MedlineCitation/PMID[1]")
        title = _text(article, ".//Article/ArticleTitle[1]")
        abstract_parts = []
        for part in article.xpath(".//Article/Abstract/AbstractText"):
            label = part.get("Label")
            value = " ".join(part.itertext()).strip()
            abstract_parts.append(f"{label}: {value}" if label else value)
        abstract = "\n".join(part for part in abstract_parts if part)
        diseases = [str(value) for value in article.xpath(
            ".//MeshHeading/DescriptorName/@UI"
        )]
        if pmid:
            records.append({"pmid": pmid, "title": title, "abstract": abstract,
                            "mesh_descriptor_ids": diseases})
    return records


def download_missing_fulltext_abstracts(csv_path: Path, pdf_dir: Path, output_dir: Path,
                                        batch_size: int = 150, retries: int = 4,
                                        gene: str = "UNKNOWN") -> Path:
    frame = pd.read_csv(csv_path)
    pmid_column = next(column for column in frame if column.lower() in {"pubmed", "pmid", "pubmed_id"})
    source_column = next((column for column in frame if column.lower() == "source_collection"), None)
    source_by_pmid = {
        str(row[pmid_column]).removesuffix(".0"): str(row[source_column])
        for _, row in frame.iterrows()
        if source_column and str(row[pmid_column]).removesuffix(".0").isdigit()
    }
    requested = sorted({str(value).removesuffix(".0") for value in frame[pmid_column].dropna()
                        if str(value).removesuffix(".0").isdigit()}, key=int)
    fulltext = set()
    for path in pdf_dir.glob("*.pdf"):
        if path.stat().st_size < 1000:
            continue
        match = __import__("re").search(r"PMID[_-]?(\d+)", path.stem, __import__("re").I)
        if match:
            fulltext.add(match.group(1))
    targets = [pmid for pmid in requested if pmid not in fulltext]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pubmed_abstracts.jsonl"
    cached = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cached[str(record["pmid"])] = record
    pending = [pmid for pmid in targets if pmid not in cached]
    session = requests.Session()
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        data = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "rettype": "abstract",
                "tool": f"{gene.lower()}_structural_variant_ranker",
                "email": "researcher@epistasis-predictor.local"}
        api_key = os.getenv("NCBI_API_KEY")
        if api_key:
            data["api_key"] = api_key
        error = None
        for attempt in range(retries):
            try:
                response = session.post(EFETCH_URL, data=data, timeout=90)
                response.raise_for_status()
                for record in _parse_pubmed_xml(response.content):
                    if record["pmid"] in targets:
                        record["source_collection"] = source_by_pmid.get(record["pmid"], "literature_csv")
                        cached[record["pmid"]] = record
                error = None
                break
            except Exception as exc:
                error = exc
                time.sleep(min(8, 2 ** attempt))
        if error is not None:
            raise RuntimeError(f"PubMed EFetch failed for batch beginning {batch[0]}") from error
        time.sleep(0.34 if not api_key else 0.11)
        print(f"PubMed abstracts: {min(start + len(batch), len(pending))}/{len(pending)}")
    with output.open("w", encoding="utf-8") as handle:
        for pmid in sorted(cached, key=int):
            handle.write(json.dumps(cached[pmid], ensure_ascii=False) + "\n")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "gene": gene.upper(),
        "source_csv": str(csv_path), "requested_unique_pmids": len(requested),
        "fulltext_pmids": len(fulltext.intersection(requested)), "abstract_targets": len(targets),
        "abstract_records": len(cached), "abstracts_with_text": sum(bool(r.get("abstract")) for r in cached.values()),
        "source": "NCBI_PubMed_EFetch_XML", "abstracts_only_when_fulltext_absent": True,
    }
    output.with_name("pubmed_abstracts_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output
