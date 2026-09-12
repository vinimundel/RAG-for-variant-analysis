"""GROBID-only, traceable full-text PDF ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lxml import etree

from healthrag.literature.corpus_layers import UNLAYERED, load_layer_map
from healthrag.literature.pubmed_records import load_cache

CHUNK_SIZE_CHARS = 2200
CHUNK_OVERLAP_CHARS = 250
_PMID = re.compile(r"PMID[_-]?(\d+)", re.I)
_TEI = {"tei": "http://www.tei-c.org/ns/1.0"}


def _pmid(filename: str) -> str:
    match = _PMID.search(filename)
    return match.group(1) if match else Path(filename).stem


def layer_map_for_collection(data_root: Path, collection: str) -> dict[str, str]:
    """Load the curated ``pmid -> corpus_layer`` map produced by corpus layering."""
    collection = collection.upper()
    return load_layer_map(data_root / "data" / "input" / collection / "evidence" /
                          f"{collection.lower()}_corpus_layers.csv")


def literature_metadata_for_collection(data_root: Path, collection: str) -> dict[str, dict]:
    """Load bibliographic provenance once for metadata carried into every chunk."""
    collection = collection.upper()
    cache = load_cache(data_root / "data" / "input" / collection / "evidence" /
                       "records" / "pubmed_records.jsonl")
    return {
        pmid: {
            "pubmed_title": record.title,
            "doi": record.doi,
            "publication": record.journal,
            "publication_types": list(record.publication_types),
            "publication_type": record.publication_types[0] if record.publication_types else "",
            "publication_year": record.year,
            "pub_date": record.pub_date,
            "pmcid": record.pmcid,
            "provenance": "pubmed_record",
        }
        for pmid, record in cache.items()
    }


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " "],
    )


def grobid_healthcheck(base_url: str, timeout: int = 10) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/api/isalive", timeout=timeout)
    if response.status_code != 200 or "true" not in response.text.lower():
        raise RuntimeError(f"GROBID is unavailable or unhealthy at {base_url}")


def _extract_tei(pdf: Path, tei_path: Path, base_url: str) -> None:
    with pdf.open("rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/processFulltextDocument",
            files={"input": (pdf.name, handle, "application/pdf")},
            data={"consolidateHeader": "1", "consolidateCitations": "0", "includeRawCitations": "1"},
            timeout=300,
        )
    if response.status_code != 200 or not response.content.lstrip().startswith(b"<"):
        raise RuntimeError(f"GROBID failed for {pdf.name}: HTTP {response.status_code}")
    tei_path.parent.mkdir(parents=True, exist_ok=True)
    tei_path.write_bytes(response.content)


def _documents_from_tei(tei_path: Path, metadata: dict) -> list[Document]:
    root = etree.parse(str(tei_path))
    title = " ".join(root.xpath("//tei:titleStmt/tei:title[1]//text()", namespaces=_TEI)).strip()
    dois = [str(value).strip() for value in root.xpath(
        "//tei:idno[translate(@type, 'doi', 'DOI')='DOI']/text()", namespaces=_TEI
    ) if str(value).strip()]
    tei_pmids = [str(value).strip() for value in root.xpath(
        "//tei:idno[contains(translate(@type, 'pubmed', 'PUBMED'), 'PUBMED')]/text()", namespaces=_TEI
    ) if str(value).strip()]
    metadata = {
        **metadata,
        "doi": (metadata.get("doi") or (dois[0] if dois else "")),
        "tei_pmid": tei_pmids[0] if tei_pmids else None,
    }
    sections = []
    abstract = " ".join(root.xpath("//tei:profileDesc/tei:abstract//text()", namespaces=_TEI)).strip()
    if abstract:
        sections.append(("Abstract", abstract))
    for division in root.xpath("//tei:text/tei:body//tei:div", namespaces=_TEI):
        heading = " ".join(division.xpath("./tei:head//text()", namespaces=_TEI)).strip() or "Body"
        paragraphs = [" ".join(p.xpath(".//text()", namespaces=_TEI)).strip()
                      for p in division.xpath("./tei:p", namespaces=_TEI)]
        text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
        if text:
            sections.append((heading[:160], text))
    documents = []
    for section, text in sections:
        # First line is deliberately the article title: MedCPT Article Encoder
        # consumes it as sequence A and the section/passage as sequence B.
        content = f"{title}\n{section}\n{text}" if title else f"{section}\n{text}"
        base = Document(page_content=content, metadata={
            **metadata, "section": section, "title": title or metadata.get("pubmed_title", ""),
            "parser": "grobid_tei", "source_type": "grobid_full_text",
            "full_text_available": True,
        })
        for index, chunk in enumerate(_splitter().split_documents([base])):
            chunk.metadata["section_chunk_index"] = index
            documents.append(chunk)
    if sections and sum(len(text) for _, text in sections) < 500:
        raise RuntimeError(f"GROBID TEI has insufficient scientific text: {tei_path.name}")
    return documents


def extract_tei_for_collection(data_root: Path, collection: str, *,
                         grobid_url: str = "http://localhost:8070",
                         workers: int = 1) -> dict:
    """Incrementally materialize the GROBID cache without retaining the corpus.

    This is deliberately separate from ``load_pdfs_for_collection``. Extraction can
    involve hundreds of large papers, and an extraction-only command must not
    also parse every paper into LangChain chunks and load abstracts into RAM.
    Each successful TEI file is an independent restart checkpoint.
    """
    from concurrent.futures import ThreadPoolExecutor

    collection = collection.upper()
    candidates = [data_root / "data" / "input" / collection / "literature" / "pdfs",
                  data_root / "data" / "input" / collection / "evidence" / "pdfs"]
    pdf_dir = next((path for path in candidates if path.exists()), candidates[0])
    discovered_pdfs = sorted(pdf_dir.glob("*.pdf"))
    invalid_pdfs = [pdf for pdf in discovered_pdfs if pdf.stat().st_size < 1000]
    pdfs = [pdf for pdf in discovered_pdfs if pdf.stat().st_size >= 1000]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    tei_dir = data_root / "data" / "output" / collection / "rag" / "tei"
    tei_dir.mkdir(parents=True, exist_ok=True)

    missing = [pdf for pdf in pdfs if not (tei_dir / f"{pdf.stem}.tei.xml").exists()]
    if missing:
        grobid_healthcheck(grobid_url)

    def process(pdf: Path) -> str:
        tei_path = tei_dir / f"{pdf.stem}.tei.xml"
        _extract_tei(pdf, tei_path, grobid_url)
        # Reject a malformed/truncated response before treating it as a checkpoint.
        etree.parse(str(tei_path))
        return pdf.name

    worker_count = max(1, int(workers))
    completed_now = []
    if worker_count == 1:
        for pdf in missing:
            completed_now.append(process(pdf))
            print(f"GROBID checkpoint {len(completed_now)}/{len(missing)}: {pdf.name}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for name in executor.map(process, missing):
                completed_now.append(name)
                print(f"GROBID checkpoint {len(completed_now)}/{len(missing)}: {name}", flush=True)

    available = sum((tei_dir / f"{pdf.stem}.tei.xml").exists() for pdf in pdfs)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "pdfs": len(pdfs),
        "pdf_files_discovered": len(discovered_pdfs),
        "eligible_pdf_count": len(pdfs),
        "invalid_pdf_count": len(invalid_pdfs),
        "invalid_pdfs": [
            {"source_file": pdf.name, "bytes": int(pdf.stat().st_size),
             "status": "invalid_download_not_full_text"}
            for pdf in invalid_pdfs
        ],
        "cached_before": len(pdfs) - len(missing),
        "extracted_now": len(completed_now),
        "tei_available": available,
        "workers": worker_count,
        "complete": available == len(pdfs),
    }
    checkpoint = tei_dir.parent / f"{collection.lower()}_grobid_extraction_manifest.json"
    checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["complete"]:
        raise RuntimeError(f"GROBID cache remains incomplete: {available}/{len(pdfs)}")
    return summary


def load_pdfs_for_collection(data_root: Path, collection: str, *,
                       grobid_url: str = "http://localhost:8070",
                       workers: int = 4) -> list[Document]:
    from concurrent.futures import ThreadPoolExecutor
    collection = collection.upper()
    candidates = [data_root / "data" / "input" / collection / "literature" / "pdfs",
                  data_root / "data" / "input" / collection / "evidence" / "pdfs"]
    pdf_dir = next((path for path in candidates if path.exists()), candidates[0])
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    grobid_healthcheck(grobid_url)
    tei_dir = data_root / "data" / "output" / collection / "rag" / "tei"
    manifest_path = (data_root / "data" / "input" / collection / "evidence" /
                     f"{collection.lower()}_literature_evidence.csv")
    source_by_pmid = {}
    if manifest_path.exists():
        import pandas as pd
        manifest = pd.read_csv(manifest_path).fillna("")
        source_by_pmid = {str(row.get("pubmed_id", "")).removesuffix(".0"):
                          str(row.get("source_collection", "literature_csv"))
                          for row in manifest.to_dict("records")}
    layers = layer_map_for_collection(data_root, collection)
    literature_metadata = literature_metadata_for_collection(data_root, collection)
    def process(pdf: Path) -> list[Document]:
        if not pdf.exists() or pdf.stat().st_size < 1000:
            return []
        checksum = hashlib.sha256(pdf.read_bytes()).hexdigest()
        pmid = _pmid(pdf.name)
        metadata = {"pmid": pmid, "collection": collection, "source_file": pdf.name,
                    "source_sha256": checksum, "source_type": "grobid_full_text",
                    "corpus_layer": layers.get(pmid, UNLAYERED),
                    "evidence_type": layers.get(pmid, UNLAYERED),
                    "full_text_available": True}
        metadata.update(literature_metadata.get(pmid, {}))
        metadata["source_collection"] = source_by_pmid.get(pmid, "literature_csv")
        tei_path = tei_dir / f"{pdf.stem}.tei.xml"
        if not tei_path.exists():
            _extract_tei(pdf, tei_path, grobid_url)
        return _documents_from_tei(tei_path, metadata)

    # executor.map preserves input order, making chunk order and the persisted
    # corpus reproducible while still overlapping independent GROBID requests.
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        nested = list(executor.map(process, pdfs))
    output = [document for documents in nested for document in documents]
    fulltext_pmids = {
        _pmid(pdf.name) for pdf, documents in zip(pdfs, nested) if documents
    }
    output.extend(load_pubmed_abstracts_for_collection(
        data_root, collection, fulltext_pmids, layers, literature_metadata))
    if not output:
        raise RuntimeError("PDF ingestion produced zero non-empty chunks")
    return output


def load_pubmed_abstracts_for_collection(data_root: Path, collection: str,
                                         fulltext_pmids: set[str],
                                         layers: dict[str, str] | None = None,
                                         literature_metadata: dict[str, dict] | None = None) -> list[Document]:
    path = (data_root / "data" / "input" / collection.upper() / "evidence" /
            "abstracts" / "pubmed_abstracts.jsonl")
    if not path.exists():
        return []
    layers = layer_map_for_collection(data_root, collection) if layers is None else layers
    literature_metadata = (literature_metadata_for_collection(data_root, collection)
                           if literature_metadata is None else literature_metadata)
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        pmid = str(record.get("pmid", ""))
        abstract, title = str(record.get("abstract", "")).strip(), str(record.get("title", "")).strip()
        if not pmid or pmid in fulltext_pmids or not abstract:
            continue
        content = f"{title}\nAbstract\n{abstract}"
        checksum = hashlib.sha256(content.encode()).hexdigest()
        base = Document(page_content=content, metadata={
            "pmid": pmid, "collection": collection.upper(), "title": title,
            "section": "Abstract", "source_file": "pubmed_abstracts.jsonl",
            "source_sha256": checksum, "source_type": "pubmed_abstract",
            "parser": "pubmed_xml", "mesh_descriptor_ids": record.get("mesh_descriptor_ids", []),
            "source_collection": record.get("source_collection", "literature_csv"),
            "corpus_layer": layers.get(pmid, UNLAYERED),
            "evidence_type": layers.get(pmid, UNLAYERED),
            "full_text_available": False,
            **literature_metadata.get(pmid, {}),
        })
        for index, chunk in enumerate(_splitter().split_documents([base])):
            chunk.metadata["section_chunk_index"] = index
            documents.append(chunk)
    return documents


def load_cached_tei_for_collection(data_root: Path, collection: str,
                                   include_abstracts: bool = True) -> list[Document]:
    """Load a complete cached TEI corpus without requiring a running GROBID JVM."""
    collection = collection.upper()
    pdf_dirs = [data_root / "data" / "input" / collection / "literature" / "pdfs",
                data_root / "data" / "input" / collection / "evidence" / "pdfs"]
    pdf_dir = next((path for path in pdf_dirs if path.exists()), pdf_dirs[0])
    pdfs = {pdf.stem: pdf for pdf in pdf_dir.glob("*.pdf") if pdf.stat().st_size >= 1000}
    tei_dir = data_root / "data" / "output" / collection / "rag" / "tei"
    manifest_path = (data_root / "data" / "input" / collection / "evidence" /
                     f"{collection.lower()}_literature_evidence.csv")
    source_by_pmid = {}
    if manifest_path.exists():
        import pandas as pd
        manifest = pd.read_csv(manifest_path).fillna("")
        source_by_pmid = {str(row.get("pubmed_id", "")).removesuffix(".0"):
                          str(row.get("source_collection", "literature_csv"))
                          for row in manifest.to_dict("records")}
    missing = sorted(set(pdfs).difference(path.name.removesuffix(".tei.xml") for path in tei_dir.glob("*.tei.xml")))
    if missing:
        raise RuntimeError(f"TEI cache incomplete: {len(missing)} PDFs lack GROBID output")
    layers = layer_map_for_collection(data_root, collection)
    literature_metadata = literature_metadata_for_collection(data_root, collection)
    output = []
    for stem, pdf in sorted(pdfs.items()):
        checksum = hashlib.sha256(pdf.read_bytes()).hexdigest()
        pmid = _pmid(pdf.name)
        metadata = {"pmid": pmid, "collection": collection, "source_file": pdf.name,
                    "source_sha256": checksum, "source_type": "grobid_full_text",
                    "corpus_layer": layers.get(pmid, UNLAYERED),
                    "evidence_type": layers.get(pmid, UNLAYERED),
                    "full_text_available": True}
        metadata.update(literature_metadata.get(pmid, {}))
        metadata["source_collection"] = source_by_pmid.get(pmid, "literature_csv")
        output.extend(_documents_from_tei(tei_dir / f"{stem}.tei.xml", metadata))
    if include_abstracts:
        fulltext_pmids = {
            _pmid(pdf.name) for stem, pdf in pdfs.items()
            if any(document.metadata.get("pmid") == _pmid(pdf.name)
                   for document in output)
        }
        output.extend(load_pubmed_abstracts_for_collection(
            data_root, collection, fulltext_pmids, layers, literature_metadata))
    if not output:
        raise RuntimeError("Cached TEI ingestion produced zero chunks")
    return output
