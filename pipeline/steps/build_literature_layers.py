"""Build the layered, versioned literature corpus for one gene.

The candidate spreadsheet is snapshotted as ``raw_candidate_corpus`` and never
edited.  Every candidate PMID is then resolved to a full PubMed record and
assigned to exactly one layer by :mod:`src.literature.corpus_layers`, with the
rule and the exclusion reason preserved.  Nothing is dropped: excluded records
stay in the table so the corpus can be audited and re-litigated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.config import ROOT, get_gene_config, paths_for
from src.literature.corpus_layers import build_corpus_layers
from src.literature.pubmed_records import fetch_records, load_cache

PMID_IN_FILENAME = re.compile(r"PMID[_-]?(\d+)", re.I)
MIN_PDF_BYTES = 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_raw_corpus(gene: str, source_csv: Path, evidence_dir: Path) -> dict:
    """Copy the candidate spreadsheet verbatim and record its checksum."""
    target = evidence_dir / f"{gene.lower()}_raw_candidate_corpus.csv"
    shutil.copyfile(source_csv, target)
    digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    return {"raw_candidate_corpus": str(target.relative_to(ROOT)),
            "source_csv": str(source_csv.relative_to(ROOT)), "source_csv_sha256": digest,
            "raw_rows": len(pd.read_csv(source_csv))}


def full_text_pmids(pdf_dir: Path) -> set[str]:
    """PMIDs with a locally stored, non-empty PDF; used for verifiability only."""
    found = set()
    for path in Path(pdf_dir).glob("*.pdf"):
        match = PMID_IN_FILENAME.search(path.stem)
        if match and path.stat().st_size >= MIN_PDF_BYTES:
            found.add(match.group(1))
    return found


def candidate_pmids(evidence_csv: Path) -> list[str]:
    frame = pd.read_csv(evidence_csv)
    column = next(name for name in frame if name.lower() in {"pubmed", "pmid", "pubmed_id"})
    values = (str(value).strip().removesuffix(".0") for value in frame[column].dropna())
    return sorted({value for value in values if value.isdigit()}, key=int)


def _unresolved_candidate_rows(evidence_csv: Path, resolved_pmids: set[str]) -> pd.DataFrame:
    """Keep candidate rows that never became a verified PubMed record.

    These rows are part of the audit trail, but they cannot enter any index. A
    missing PMID is classified as missing provenance; a PMID absent from the
    local EFetch cache is classified as an unresolved record.
    """
    frame = pd.read_csv(evidence_csv).fillna("")
    pmid_column = next(name for name in frame
                       if name.lower() in {"pubmed", "pmid", "pubmed_id"})
    title_column = next((name for name in frame
                         if name.lower() in {"title", "titulo", "article_title"}), None)
    doi_column = next((name for name in frame if name.lower() == "doi"), None)
    pmcid_column = next((name for name in frame if name.lower() == "pmcid"), None)
    rows = []
    for _, source in frame.iterrows():
        raw_pmid = str(source[pmid_column]).strip().removesuffix(".0")
        pmid = raw_pmid if raw_pmid.isdigit() else ""
        if pmid and pmid in resolved_pmids:
            continue
        title = str(source[title_column]).strip() if title_column else ""
        lowered = title.lower()
        junk = any(token in lowered for token in (
            "access denied", "page not found", "search results", "captcha",
            "javascript", "cookie policy", "internal server error",
        ))
        if junk:
            rule_id, reason = "scraping_junk", "scraping_junk"
        elif not pmid:
            rule_id, reason = "missing_provenance", "missing_pmid_or_doi_or_title"
        else:
            rule_id, reason = "unresolved_pubmed_record", "pubmed_record_not_in_cache"
        layer = "excluded"
        rows.append({
            "pmid": pmid, "layer": layer, "rule_id": rule_id,
            "exclusion_reason": reason,
            "gene_link": "unverified_no_pubmed_record",
            "publication_types": "", "duplicate_of": "", "affecting_notices": "",
            "full_text_available": False, "manual_review_status": "not_reviewed",
            "manual_review_note": "", "title": title, "journal": "", "year": None,
            "doi": str(source[doi_column]).strip() if doi_column else "",
            "pmcid": str(source[pmcid_column]).strip() if pmcid_column else "",
            "has_abstract": False, "gene_mentions": 0,
            "section": "", "parser": "candidate_catalog",
            "publication": "", "evidence_type": layer,
            "source_row": int(source.get("source_row", 0) or 0),
        })
    return pd.DataFrame(rows)


def build(gene: str, offline: bool = False) -> Path:
    gene = gene.upper()
    cfg, paths = get_gene_config(gene), paths_for(gene)
    evidence_dir = paths["evidence"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_csv = evidence_dir / f"{gene.lower()}_literature_evidence.csv"
    if not evidence_csv.exists():
        raise FileNotFoundError(f"Acquisition manifest not found: {evidence_csv}")

    raw_snapshot = snapshot_raw_corpus(gene, ROOT / cfg["literature_csv"], evidence_dir)
    pmids = candidate_pmids(evidence_csv)
    cache_path = evidence_dir / "records" / "pubmed_records.jsonl"
    records = (load_cache(cache_path) if offline
               else fetch_records(pmids, cache_path, gene=gene))
    resolved = [records[pmid] for pmid in pmids if pmid in records]

    overrides_path = evidence_dir / f"{gene.lower()}_manual_corpus_overrides.json"
    overrides = (json.loads(overrides_path.read_text(encoding="utf-8"))
                 if overrides_path.exists() else {})
    layers = build_corpus_layers(
        resolved, aliases=cfg["literature_aliases"],
        mesh_terms=cfg.get("literature_mesh_terms", ()),
        full_text_pmids=full_text_pmids(paths["input"] / "evidence" / "pdfs"),
        manual_overrides=overrides,
    )
    unresolved = _unresolved_candidate_rows(evidence_csv, set(records))
    if not unresolved.empty:
        layers = pd.concat([layers, unresolved], ignore_index=True, sort=False)
    source_rows = pd.read_csv(evidence_csv).fillna("")
    source_pmid_column = next(name for name in source_rows
                              if name.lower() in {"pubmed", "pmid", "pubmed_id"})
    source_rows["_normalized_pmid"] = source_rows[source_pmid_column].map(
        lambda value: str(value).strip().removesuffix(".0")
        if str(value).strip().removesuffix(".0").isdigit() else ""
    )
    first_source_row = (
        source_rows.loc[source_rows["_normalized_pmid"].ne(""),
                       ["_normalized_pmid", "source_row"]]
        .drop_duplicates("_normalized_pmid")
        .set_index("_normalized_pmid")["source_row"]
        .to_dict()
    )
    if "source_row" in layers:
        layers["source_row"] = layers["source_row"].fillna(
            layers["pmid"].map(first_source_row))
    output = evidence_dir / f"{gene.lower()}_corpus_layers.csv"
    layers.to_csv(output, index=False)
    layers.to_parquet(output.with_suffix(".parquet"), index=False)
    for layer in ("primary_evidence", "context_reference", "excluded"):
        if layer not in set(layers["layer"]):
            (evidence_dir / "records" / f"{layer}_pmids.json").write_text(
                "[]", encoding="utf-8"
            )
            continue
        subset = layers.loc[layers["layer"] == layer, "pmid"].tolist()
        (evidence_dir / "records" / f"{layer}_pmids.json").write_text(
            json.dumps(sorted({pmid for pmid in subset if str(pmid).isdigit()}, key=int), indent=2),
            encoding="utf-8"
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "gene": gene,
        **raw_snapshot,
        "acquisition_csv": str(evidence_csv.relative_to(ROOT)),
        "acquisition_csv_sha256": _sha256(evidence_csv),
        "candidate_pmids": len(pmids), "resolved_records": len(resolved),
        "unresolved_pmids": len(pmids) - len(resolved),
        "audited_candidate_rows": int(len(layers)),
        "unresolved_candidate_rows": int(len(unresolved)),
        "offline_build": offline,
        "layer_counts": {str(k): int(v) for k, v in layers["layer"].value_counts().items()},
        "rule_counts": {str(k): int(v) for k, v in layers["rule_id"].value_counts().items()},
        "exclusion_reason_counts": {
            str(k): int(v) for k, v in
            layers.loc[layers["exclusion_reason"].ne(""), "exclusion_reason"].value_counts().items()
        },
        "gene_link_counts": {str(k): int(v) for k, v in layers["gene_link"].value_counts().items()},
        "full_text_in_primary_evidence": int(
            layers.loc[layers["layer"] == "primary_evidence", "full_text_available"].sum()),
        "preprints_excluded": int(
            ((layers["rule_id"] == "preprint") & layers["layer"].eq("excluded")).sum()),
        "retractions_excluded": int(
            (layers["rule_id"] == "retracted").sum()),
        "missing_provenance_excluded": int(
            (layers["rule_id"] == "missing_provenance").sum()),
        "scraping_junk_excluded": int(
            (layers["rule_id"] == "scraping_junk").sum()),
        "manual_overrides": len(overrides),
        "layer_definition_version": "braf_corpus_layers_v2",
        "indexed_layers": ["primary_evidence", "context_reference"],
        "excluded_is_never_indexed": True,
        "abstract_policy": "use only when no successfully parsed full text exists",
        "claim_policy": "reviews and other non-primary records cannot support direct mechanistic claims",
        "artifacts": {
            "layers_csv": str(output.relative_to(ROOT)),
            "layers_csv_sha256": _sha256(output),
            "layers_parquet": str(output.with_suffix(".parquet").relative_to(ROOT)),
            "layers_parquet_sha256": _sha256(output.with_suffix(".parquet")),
            "pubmed_records_cache": str(cache_path.relative_to(ROOT)),
            "pubmed_records_cache_sha256": _sha256(cache_path),
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--offline", action="store_true",
                        help="Layer only PMIDs already present in the record cache")
    args = parser.parse_args()
    print(build(args.gene, args.offline))
