from types import SimpleNamespace

import pytest

from healthrag.rag.biomedical import (
    AnswerDraft,
    BiomedicalRAG,
    Citation,
    GroundedClaim,
    Question,
    extractive_support,
    validate_citations,
)
from healthrag.rag.schemas import LiteratureEvidence


def source(evidence_id="E01", passage="BRAF V600E was associated with shorter survival."):
    return LiteratureEvidence(
        evidence_id=evidence_id,
        pmid="123",
        passage=passage,
        source_file="PMID123.pdf",
        corpus_layer="primary_evidence",
        evidence_type="primary_evidence",
    )


def test_citation_validator_accepts_an_exact_quote():
    draft = AnswerDraft(
        status="answered",
        claims=[GroundedClaim(
            text="The study reported shorter survival.",
            citations=[Citation(
                evidence_id="E01",
                quote="BRAF V600E was associated with shorter survival.",
            )],
        )],
        limitations="One observational study.",
    )
    assert validate_citations(draft, [source()]) is draft


@pytest.mark.parametrize(
    "citation",
    [
        Citation(evidence_id="E99", quote="BRAF V600E was associated with shorter survival."),
        Citation(evidence_id="E01", quote="The model invented this sentence."),
    ],
)
def test_citation_validator_rejects_unknown_ids_and_fabricated_quotes(citation):
    draft = AnswerDraft(
        status="answered",
        claims=[GroundedClaim(text="Claim", citations=[citation])],
        limitations="",
    )
    with pytest.raises(ValueError):
        validate_citations(draft, [source()])


def test_citation_validator_rejects_a_title_as_support_for_an_outcome_claim():
    passage = (
        "Study of BRAF V600E: impact on patient outcomes. "
        "BRAF V600E was associated with reduced overall survival."
    )
    draft = AnswerDraft(
        status="answered",
        claims=[GroundedClaim(
            text="BRAF V600E was associated with reduced overall survival.",
            citations=[Citation(evidence_id="E01", quote="Study of BRAF V600E: impact on patient outcomes.")],
        )],
        limitations="",
    )
    with pytest.raises(ValueError, match="does not state the outcome"):
        validate_citations(draft, [source(passage=passage)])


def test_empty_retrieval_abstains_without_loading_the_llm(monkeypatch, tmp_path):
    rag_dir = tmp_path / "data/output/TEST/rag"
    rag_dir.mkdir(parents=True)
    (rag_dir / "test_rag_index_manifest.json").write_text("{}")
    monkeypatch.setattr("healthrag.rag.biomedical.HybridRetriever.retrieve", lambda *args: [])
    monkeypatch.setattr(
        "healthrag.rag.biomedical.get_llm",
        lambda *args, **kwargs: pytest.fail("The LLM must not load for an empty retrieval"),
    )
    answer = BiomedicalRAG(tmp_path).invoke(Question(question="What evidence is available?", collection="TEST"))
    assert answer.status == "insufficient_evidence"
    assert answer.claims == []
    assert answer.sources == []


def test_langchain_pipeline_exposes_named_retrieval_and_generation_steps():
    pipeline = BiomedicalRAG().chain
    assert [step.config["run_name"] for step in pipeline.steps] == [
        "hybrid_retrieval", "grounded_generation"
    ]


def test_extractive_fallback_requires_strong_question_coverage():
    evidence = [source(passage=(
        "In metastatic colorectal cancer, BRAF V600E was associated with reduced overall survival."
    ))]
    claim = extractive_support(
        "How was BRAF V600E associated with overall survival in metastatic colorectal cancer?",
        evidence,
    )
    assert claim is not None
    assert claim.citations[0].quote in evidence[0].passage
    assert extractive_support("What treats Huntington disease?", evidence) is None


def test_extractive_fallback_selects_the_best_document_before_its_best_sentence():
    colorectal = source(
        evidence_id="E01",
        passage=(
            "Patients and Methods: 504 metastatic colorectal cancer patients were analyzed. "
            "Patients with BRAFV600E tumors had reduced overall survival: 14.0 versus 34.6 months."
        ),
    )
    melanoma = source(
        evidence_id="E02",
        passage=(
            "BRAF V600E metastatic melanoma trials reported improved overall survival with inhibitors."
        ),
    )
    claim = extractive_support(
        "In the 504-patient metastatic colorectal cancer study, how was BRAF V600E associated "
        "with overall survival?",
        [melanoma, colorectal],
    )
    assert claim is not None
    assert claim.citations[0].evidence_id == "E01"
    assert "14.0 versus 34.6 months" in claim.text
