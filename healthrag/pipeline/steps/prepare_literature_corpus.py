"""Normalize, resolve and audit a per-gene literature CSV before acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

from healthrag.pipeline.config import ROOT, get_gene_config, paths_for

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _pmid(value) -> str:
    value = str(value).strip().removesuffix(".0")
    return value if value.isdigit() else ""


def _request(session: requests.Session, endpoint: str, params: dict, retries: int = 4) -> dict:
    error = None
    for attempt in range(retries):
        try:
            response = session.get(f"{EUTILS}/{endpoint}", params=params, timeout=60)
            response.raise_for_status()
            time.sleep(0.11 if params.get("api_key") else 0.34)
            return response.json()
        except Exception as exc:
            error = exc
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"NCBI {endpoint} failed") from error


def _resolve_title(session: requests.Session, title: str, api_key: str | None) -> dict:
    params = {"db": "pubmed", "term": f'"{title}"[Title]', "retmode": "json", "retmax": 5,
              "tool": "biomedical_evidence_rag", "email": "researcher@example.org"}
    if api_key:
        params["api_key"] = api_key
    search = _request(session, "esearch.fcgi", params)
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"resolution_status": "not_found"}
    summary_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json",
                      "tool": params["tool"], "email": params["email"]}
    if api_key:
        summary_params["api_key"] = api_key
    result = _request(session, "esummary.fcgi", summary_params).get("result", {})
    candidates = []
    query_key = _title_key(title)
    for pmid in ids:
        item = result.get(str(pmid), {})
        candidate_title = str(item.get("title", ""))
        similarity = SequenceMatcher(None, query_key, _title_key(candidate_title)).ratio()
        article_ids = {str(x.get("idtype", "")).lower(): str(x.get("value", ""))
                       for x in item.get("articleids", [])}
        candidates.append((similarity, str(pmid), candidate_title, article_ids.get("doi", "")))
    candidates.sort(reverse=True)
    best = candidates[0]
    # A PMID is accepted only for an essentially identical normalized title and
    # only when there is no similarly good competing record.
    ambiguous = len(candidates) > 1 and candidates[1][0] >= best[0] - 0.01
    if best[0] < 0.97 or ambiguous:
        return {"resolution_status": "ambiguous", "candidate_pmid": best[1],
                "candidate_title": best[2], "title_similarity": round(best[0], 5)}
    return {"resolution_status": "resolved_exact_title", "resolved_pmid": best[1],
            "doi": best[3], "candidate_title": best[2], "title_similarity": round(best[0], 5)}


def prepare(gene: str, csv_path: Path | None = None, previous_csv: Path | None = None,
            resolve_missing: bool = False) -> Path:
    gene = gene.upper()
    cfg, paths = get_gene_config(gene), paths_for(gene)
    source = Path(csv_path or ROOT / cfg["literature_csv"])
    frame = pd.read_csv(source).fillna("")
    title_col = next((c for c in frame if c.lower() in {"title", "titulo", "article_title"}), None)
    pmid_col = next((c for c in frame if c.lower() in {"pubmed", "pmid", "pubmed_id"}), None)
    if title_col is None or pmid_col is None:
        raise ValueError("Literature CSV must contain Title and PubMed/PMID columns")
    old_pmids: set[str] = set()
    if previous_csv and Path(previous_csv).exists():
        old = pd.read_csv(previous_csv).fillna("")
        old_column = next(c for c in old if c.lower() in {"pubmed", "pmid", "pubmed_id"})
        old_pmids = {_pmid(value) for value in old[old_column] if _pmid(value)}
    records = []
    for index, row in frame.iterrows():
        pmid, title = _pmid(row[pmid_col]), str(row[title_col]).strip()
        records.append({"source_row": int(index) + 1, "title": title, "title_key": _title_key(title),
                        "pmid": pmid, "doi": "", "source_collection":
                        ("csv_previous" if pmid and pmid in old_pmids else "csv_aug_12_new"),
                        "resolution_status": "provided_pmid" if pmid else "unresolved_no_pmid"})
    normalized = pd.DataFrame(records)
    if resolve_missing:
        import os
        session, api_key = requests.Session(), os.getenv("NCBI_API_KEY")
        evidence_dir = paths["evidence"]
        evidence_dir.mkdir(parents=True, exist_ok=True)
        cache_path = evidence_dir / f"{gene.lower()}_title_resolution_cache.json"
        cache = (json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {})
        for row_index in normalized.index[normalized["pmid"].eq("")]:
            key = normalized.at[row_index, "title_key"]
            if key not in cache:
                cache[key] = _resolve_title(session, normalized.at[row_index, "title"], api_key)
                cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            result = cache[key]
            normalized.at[row_index, "resolution_status"] = result["resolution_status"]
            normalized.at[row_index, "pmid"] = result.get("resolved_pmid", "")
            normalized.at[row_index, "doi"] = result.get("doi", "")
            normalized.at[row_index, "candidate_pmid"] = result.get("candidate_pmid", "")
            normalized.at[row_index, "candidate_title"] = result.get("candidate_title", "")
            normalized.at[row_index, "title_similarity"] = result.get("title_similarity", pd.NA)
    normalized["duplicate_reason"] = ""
    valid = normalized["pmid"].ne("")
    normalized.loc[valid & normalized.duplicated("pmid", keep="first"), "duplicate_reason"] = "duplicate_pmid"
    no_pmid = ~valid
    normalized.loc[no_pmid & normalized.duplicated("title_key", keep="first"), "duplicate_reason"] = "duplicate_title"
    normalized["included"] = normalized["duplicate_reason"].eq("") & normalized["pmid"].ne("")
    evidence_dir = paths["evidence"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    audit_path = evidence_dir / f"{gene.lower()}_literature_catalog_audit.csv"
    normalized.to_csv(audit_path, index=False)
    canonical = normalized.loc[normalized["included"], ["title", "pmid", "doi", "source_collection"]].rename(
        columns={"pmid": "PubMed", "title": "Title", "doi": "DOI"}
    )
    output = evidence_dir / f"{gene.lower()}_literature_canonical.csv"
    canonical.to_csv(output, index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "gene": gene,
        "source_csv": str(source), "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(frame), "provided_unique_pmids": int(frame[pmid_col].map(_pmid).replace("", pd.NA).nunique()),
        "canonical_unique_pmids": int(canonical["PubMed"].nunique()),
        "unresolved_rows": int(normalized["pmid"].eq("").sum()),
        "duplicate_rows": int(normalized["duplicate_reason"].ne("").sum()),
        "resolution_status_counts": {str(k): int(v) for k, v in normalized["resolution_status"].value_counts().items()},
        "canonical_csv": str(output), "audit_csv": str(audit_path),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--literature-csv", type=Path)
    parser.add_argument("--previous-csv", type=Path)
    parser.add_argument("--resolve-missing", action="store_true")
    parser.add_argument("--condition-profile", help="Validated for provenance; expansion is a separate step")
    args = parser.parse_args()
    print(prepare(args.gene, args.literature_csv, args.previous_csv, args.resolve_missing))
