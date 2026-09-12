# Validation record

This file records checks that were actually executed. Planned checks are not reported as results.

## Automated tests

```bash
python -m pytest -q
```

The suite uses synthetic passages and mocks. It checks ingestion failure behavior, GROBID
provenance, corpus-layer isolation, BM25 persistence, API contracts, citation validation,
abstention, module independence, and RAGAS report handling. It does not measure biomedical quality.

Result on 12 September 2026: **105 passed** in the CPU-only development environment.

## Real BRAF smoke test

The local smoke test uses the two open-access articles in `examples/braf/corpus.json`:

1. verify or download publisher PDFs;
2. extract TEI with GROBID, or load previously generated GROBID TEI;
3. build BM25 and MedCPT/Qdrant indexes;
4. retrieve and generate answers with Ollama;
5. score saved answers with deterministic metrics and RAGAS.

The complete CPU-only path was executed on 12 September 2026:

- 2 open-access PLOS papers were represented by GROBID TEI artifacts;
- 41 section-aware chunks were indexed with MedCPT and embedded Qdrant;
- the expected colorectal cancer paper was ranked first for the 504-patient survival question;
- the request completed in 48.96 seconds: 26.87 seconds retrieval and 22.07 seconds generation;
- Qwen 2.5 1.5B abstained, so the transparent extractive fallback returned the exact E01 sentence
  reporting overall survival of 14.0 versus 34.6 months (`p<0.001`);
- deterministic expected-source recall and expected-status agreement were both 1.0;
- a one-case local RAGAS run scored faithfulness 1.0 and context recall 1.0.

The RAGAS values describe one engineering smoke test judged by the same small local model family.
They are not estimates of clinical quality or broad-domain accuracy. The compact machine-readable
record is in `examples/braf/real_smoke_test.json`; full generated outputs remain ignored by Git.

The constrained WSL incident and mitigations are documented in
[local runtime safety](LOCAL_RUNTIME_SAFETY.md). The successful final run used CPU execution and did
not restart WSL.
