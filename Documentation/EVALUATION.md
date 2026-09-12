# Evaluation design

The project measures retrieval, generation, and failure behavior separately.

## Deterministic checks

Each JSONL case declares a question, collection, profile, expected source PMIDs, expected response
status, and a reference answer. The runner persists the complete output after every case and records:

- source recall against the declared PMID set;
- answer or abstention status agreement;
- citation ID and exact-quote validity through the serving contract;
- index, prompt, model, and dataset provenance;
- retrieval and generation latency.

These checks require no LLM judge. Expected PMIDs still represent a human test-set decision and
must be reviewed when the corpus changes.

## RAGAS checks

RAGAS 0.4 collection metrics evaluate saved answers, allowing generation and judging to run in
separate processes on memory-constrained machines.

- `Faithfulness` measures whether response statements are supported by retrieved context.
- `ContextRecall` measures whether retrieved passages cover the reference answer.

The evaluator uses an OpenAI-compatible endpoint and defaults to local Ollama. Reports contain the
judge model, endpoint, RAGAS version, per-case errors, and input hash. Abstentions and technical
failures are skipped by judge metrics and stay visible in deterministic measurements.

The evaluation extra pins `langchain-community` below 0.4.2 because RAGAS 0.4.3 imports a
VertexAI compatibility module removed in 0.4.2. The evaluator does not use VertexAI, but the module
is imported when RAGAS initializes. Keeping the constraint explicit makes clean installs
reproducible until RAGAS removes that import.

## Interpretation

Judge metrics are stochastic and model-dependent. Exact-quote validation establishes traceability,
not semantic entailment. The included two-paper, three-question BRAF dataset is an engineering
smoke test and cannot support claims about biomedical accuracy.
