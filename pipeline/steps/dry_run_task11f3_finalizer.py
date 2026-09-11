"""Run the V7 finalizer against synthetic labels in a temporary root."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.config import ROOT
from src.literature.task11 import sha256
from src.literature.task11f2 import finalize_v7


def run(gene: str = "BRAF") -> Path:
    gene = gene.upper()
    source_rag = ROOT / "data" / "output" / gene / "rag"
    source_dir = source_rag / "benchmark_v7_confirmatory"
    with tempfile.TemporaryDirectory(prefix="task11f3_dry_run_") as temporary:
        temp_root = Path(temporary)
        temp_rag = temp_root / "data" / "output" / gene / "rag"
        temp_rag.mkdir(parents=True)
        shutil.copytree(source_dir, temp_rag / "benchmark_v7_confirmatory")
        for name in (
            f"{gene.lower()}_task11f2_v6_control_manifest.json",
            f"{gene.lower()}_rag_stack_manifest.json",
            f"{gene.lower()}_rag_index_manifest.json",
            f"{gene.lower()}_task11_gate_manifest.json",
        ):
            shutil.copy2(source_rag / name, temp_rag / name)
        out = temp_rag / "benchmark_v7_confirmatory"
        sheet_path = out / f"{gene.lower()}_rag_benchmark_v7_confirmatory_sheet_template.csv"
        abstention_path = out / f"{gene.lower()}_rag_benchmark_v7_confirmatory_abstention_sheet_template.csv"
        repeat_path = out / f"{gene.lower()}_rag_benchmark_v7_confirmatory_blind_repeats_template.csv"
        sheet = pd.read_csv(sheet_path)
        abstention = pd.read_csv(abstention_path)
        repeats = pd.read_csv(repeat_path)
        key = json.loads((out / f"{gene.lower()}_rag_benchmark_v7_confirmatory_repeat_key.json").read_text())
        scope_map = {"exact_variant": "exact_variant", "same_residue_analogy": "same_residue",
                     "functional_region": "functional_region"}
        for column in ("relevance", "specificity", "entailment", "support_type", "quality", "conflict"):
            sheet[column] = sheet[column].astype(object)
        abstention["abstention_judgment"] = abstention["abstention_judgment"].astype(object)
        for column in ("relevance", "specificity", "entailment", "support_type", "quality", "conflict",
                       "abstention_judgment"):
            repeats[column] = repeats[column].astype(object)
        for column, value in {
            "entailment": "supports", "support_type": "structural_biophysical",
            "quality": "controlled", "conflict": "no",
        }.items():
            sheet[column] = value
        sheet["relevance"] = ["relevant" if index % 2 else "directly_on_point"
                               for index in range(len(sheet))]
        sheet["specificity"] = sheet["automatic_evidence_scope"].map(scope_map)
        for index in range(len(abstention)):
            abstention.loc[index, "abstention_judgment"] = (
                "correct_abstention" if index < max(len(abstention) - 1, 1) else "missed_evidence"
            )
        original_mechanistic = sheet.set_index("item_id")
        original_abstention = abstention.set_index("item_id")
        for mapping in key["mapping"]:
            row_index = repeats.index[repeats["repeat_item_id"].eq(mapping["repeat_item_id"])][0]
            if mapping["annotation_domain"] == "mechanistic":
                source = original_mechanistic.loc[mapping["original_item_id"]]
                for column in ("relevance", "specificity", "entailment", "support_type", "quality", "conflict"):
                    repeats.loc[row_index, column] = source[column]
            else:
                source = original_abstention.loc[mapping["original_item_id"]]
                repeats.loc[row_index, "abstention_judgment"] = source["abstention_judgment"]
        sheet.to_csv(out / f"{gene.lower()}_rag_benchmark_v7_confirmatory_sheet_filled.csv", index=False)
        abstention.to_csv(out / f"{gene.lower()}_rag_benchmark_v7_confirmatory_abstention_sheet_filled.csv", index=False)
        repeats.to_csv(out / f"{gene.lower()}_rag_benchmark_v7_confirmatory_blind_repeats_filled.csv", index=False)
        result = finalize_v7(temp_root, gene)
        payload = json.loads(result.read_text(encoding="utf-8"))
        output = source_dir / f"{gene.lower()}_rag_benchmark_v7_dry_run.json"
        output.write_text(json.dumps({
            "status": "passed" if payload.get("status") == "passed" else payload.get("status"),
            "created_utc": datetime.now(timezone.utc).isoformat(), "gene": gene,
            "synthetic_labels": True, "production_labels_written": False,
            "temporary_result": payload,
            "temporary_result_sha256": sha256(result),
            "single_use_production_marker_created": False,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="BRAF")
    args = parser.parse_args()
    print(run(args.gene))
