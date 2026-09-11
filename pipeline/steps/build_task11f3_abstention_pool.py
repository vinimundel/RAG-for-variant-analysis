"""Build the 12-case abstention adjudication pool without labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from pipeline.config import ROOT, paths_for
from src.literature.task11 import sha256


def build(gene: str = "BRAF"):
    gene = gene.upper()
    rag = paths_for(gene)["rag"]
    source = rag / "benchmark_v6_confirmatory" / \
        f"{gene.lower()}_rag_benchmark_v6_confirmatory_abstention_sheet.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    abstentions = pd.read_csv(source)
    rows = []
    for item in abstentions.to_dict("records"):
        variant = str(item["mutation"])
        caches = sorted((rag / "retrieval_cache").glob(f"{variant}__discovery_v5__*.json"))
        if len(caches) != 1:
            raise RuntimeError(f"expected one V6 retrieval cache for {variant}, found {len(caches)}")
        payload = json.loads(caches[0].read_text(encoding="utf-8"))
        context = payload.get("context_reference", [])
        rows.append({
            "pool_item_id": str(item["item_id"]), "mutation": variant,
            "claim_question": str(item["claim_question"]),
            "mechanistic_evidence_status": str(item["mechanistic_evidence_status"]),
            "context_reference_count": len(context),
            "context_pmids": ";".join(dict.fromkeys(str(value.get("pmid", "")) for value in context)),
            "context_passages": " || ".join(
                f"[{value.get('evidence_id')}] {value.get('passage', '')}" for value in context
            ),
            "adjudication_decision": "",
            "adjudication_note": "",
            "mechanistic_score_eligible": False,
        })
    if len(rows) != 12 or len({row["mutation"] for row in rows}) != 12:
        raise RuntimeError("abstention adjudication pool must contain 12 unique cases")
    output_dir = rag / "benchmark_v7_confirmatory"
    output = output_dir / f"{gene.lower()}_rag_benchmark_v7_abstention_adjudication_pool.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    manifest = {
        "manifest_version": "task11f3_abstention_pool_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "gene": gene,
        "status": "unannotated", "cases": len(rows),
        "source_abstention_sheet": str(source.relative_to(ROOT)),
        "source_abstention_sheet_sha256": sha256(source),
        "output": str(output.relative_to(ROOT)), "output_sha256": sha256(output),
        "purpose": "adjudicate whether no_mechanistic_evidence_retrieved is correct",
        "mechanistic_score_eligible": False,
        "allowed_decisions": ["correct_abstention", "missed_evidence"],
        "labels_read": False,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(build(args.gene))
