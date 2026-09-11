"""Evidence-first literature mechanism analysis."""

from __future__ import annotations

import gc
import re
import json

from langchain_core.documents import Document

from typing import Sequence

from src.literature.corpus_layers import UNLAYERED
from src.rag.document_retrieval import (
    MAX_PASSAGES_PER_DOCUMENT, document_ranking, retrieval_diagnostics,
    rrf_over_documents, select_passages,
)
from src.rag.llm import get_llm
from src.rag.bm25_index import PersistentBM25, bm25_corpus_path
from src.rag.prompts import MECHANISTIC_PROMPT, DISCOVERY_PROMPT
from src.rag.reranker import MedCPTReranker
from src.rag.schemas import (
    LiteratureEvidence, MechanisticLiteratureInference, MechanisticQuery, MechanisticRAGResult,
    DiscoveryHypothesis, DiscoveryInference, DiscoveryQuery, DiscoveryRAGResult, DiseaseAssociation,
)
from src.rag.retrieval_v5 import (
    MECHANISTIC_SCOPES, SCOPE_PRIORITY, classify_evidence_scope,
    normalized_rank_scores, split_evidence_blocks, strongest_scope,
)
from src.rag.query_profiles import get_condition_profile, normalized_condition_terms
from src.rag.variant_mentions import (
    contains_exact_variant,
    local_variant_windows,
    same_residue_variant_mentioned,
)
from src.rag.vector_store import load_vector_store, retrieve_many_with_scores, retrieve_with_scores


# Mechanistic retrieval scores only the primary layer.  Reviews and guidelines
# live in a physically separate index and may only be shown as context.
EVIDENCE_LAYER = "primary_evidence"
CONTEXT_LAYER = "context_reference"

# Evidence channels are scored separately so a mechanistic question is not
# answered with therapeutic response data.
QUERY_CHANNELS = (
    "variant_exact", "conformational_state", "structure_motif",
    "ppi_dimerization", "disease", "pharmacology",
)
MECHANISTIC_CHANNELS = (
    "variant_exact", "conformational_state", "structure_motif",
    "ppi_dimerization", "disease",
)
# The only mechanisms for which pharmacological evidence is on-topic.  Bare
# "binding" is deliberately absent: a binding-site annotation is a structural
# fact, and matching it once opened this channel for 21 of 171 BRAF variants.
PHARMACOLOGY_TERMS = ("resistance", "resistencia", "resistência", "inhibitor", "inibidor",
                      "drug binding", "drug resistance", "drug response", "vemurafenib",
                      "dabrafenib", "encorafenib", "belvarafenib", "plx4720", "plx8394",
                      "trametinib")
_PHARMACOLOGY_PATTERN = re.compile(
    "|".join(rf"(?<![a-z]){re.escape(term)}(?![a-z])" for term in PHARMACOLOGY_TERMS),
    re.IGNORECASE,
)
# Documents reranked before any passage is chosen from them.
DOCUMENT_CANDIDATES = 20


def _evidence_id(index: int) -> str:
    return f"E{index:02d}"


def _normalize_citation_ids(values: list[str], allowed: set[str]) -> list[str]:
    """Normalize harmless formatting variants, retaining the strict whitelist."""
    normalized = []
    for value in values:
        match = re.fullmatch(r"\s*\[?E?0*(\d+)\]?\s*", str(value), flags=re.IGNORECASE)
        candidate = f"E{int(match.group(1)):02d}" if match else str(value).strip().upper()
        if candidate not in allowed:
            raise ValueError(f"LLM cited unavailable evidence ID {value!r}")
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


_MECHANISM_TERMS = {
    "BRAF": ("inactive", "active", "kinase", "dimer", "mek phosphorylation", "signaling"),
    "GSDMD": ("cleavage", "cleaved", "caspase", "pore", "oligomer", "membrane", "pyroptosis", "autoinhibit"),
}

_REGION_ALIASES = {
    "activation_segment": ("activation segment", "activation loop"),
    "p_loop": ("p-loop", "phosphate-binding loop"),
    "alpha_c_helix": ("alpha-c helix", "αc helix"),
    "interdomain_linker": ("interdomain linker", "cleavage region", "cleavage site"),
    "pore_forming_n_terminal_domain": ("n-terminal domain", "pore-forming domain"),
    "autoinhibitory_c_terminal_domain": ("c-terminal domain", "autoinhibitory domain"),
    "caspase_3_7_inactivating_site": ("caspase-3", "caspase 3", "asp87"),
    "inflammatory_caspase_cleavage_region": ("caspase cleavage", "asp275", "interdomain linker"),
    "cys191_regulatory_site": ("cys191", "c191"),
}


def _mechanistic(text: str, gene: str) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in _MECHANISM_TERMS.get(gene.upper(), ()))


def _exact_mechanistic_windows(passage: str, variant: str, gene: str) -> list[str]:
    return [window for window in local_variant_windows(passage, variant) if _mechanistic(window, gene)]


def _therapy_context_only(item: LiteratureEvidence) -> bool:
    """Identify therapeutic context that cannot carry a mechanistic claim."""
    channels = set(item.retrieval_channels)
    if "pharmacology" not in channels:
        return False
    direct_terms = (
        "kinase activity", "phosphorylation", "dimerization", "dimer interface",
        "conformation", "structural", "biochemical assay", "cell-free assay",
        "binding interface", "oligomer", "cleavage", "pore",
    )
    return channels.issubset({"pharmacology", "disease"}) and not any(
        term in item.passage.lower() for term in direct_terms
    )


def _attach_evidence_scopes(inference: DiscoveryInference,
                            evidence: list[LiteratureEvidence]) -> DiscoveryInference:
    by_id = {item.evidence_id: item.evidence_scope for item in evidence}
    hypotheses = []
    for hypothesis in inference.hypotheses:
        scopes = [by_id[evidence_id] for evidence_id in hypothesis.cited_evidence_ids
                  if evidence_id in by_id]
        hypotheses.append(hypothesis.model_copy(update={
            "evidence_scope": strongest_scope(scopes),
        }))
    return inference.model_copy(update={"hypotheses": hypotheses})


def _condition_support_windows(
    passage: str, variant: str, disease_terms: tuple[str, ...]
) -> list[tuple[str, list[str]]]:
    output = []
    # Disease scoring is stricter than mechanistic retrieval: the exact variant
    # and condition must occur in the same sentence, not merely adjacent ones.
    for window in local_variant_windows(passage, variant, radius=0):
        lowered = window.lower()
        matched = [term for term in disease_terms if term in lowered]
        if matched:
            output.append((lowered, matched))
    return output


class LiteratureMechanismAnalyzer:
    def __init__(self, gene: str, host: str = "localhost", port: int = 6333,
                 candidate_k: int = 20, rag_dir=None, layer: str | None = EVIDENCE_LAYER):
        self.gene, self.host, self.port, self.candidate_k = gene.upper(), host, port, candidate_k
        if self.gene == "BRAF" and layer != EVIDENCE_LAYER:
            raise ValueError("BRAF mechanistic retrieval is restricted to primary_evidence")
        from pathlib import Path
        self.rag_dir = Path(rag_dir) if rag_dir else Path("data/output") / self.gene / "rag"
        # ``layer=None`` addresses a corpus indexed before curation existed; it
        # must be requested explicitly and is reported in the retrieval manifest.
        self.layer = layer
        self._store = None
        self._bm25 = None
        self._context_store = None
        self._context_bm25 = None
        self._document_sizes = None
        # Diagnostics of the most recent discovery retrieval, for auditing bias.
        self.last_retrieval_diagnostics: dict = {}
        self.last_retrieval_blocks: dict[str, list[LiteratureEvidence]] = {
            "mechanistic_evidence": [], "context_reference": [],
        }
        # Loading the cross-encoder eagerly makes an inference-only process pay
        # the full retrieval memory cost.  Keep it lazy so retrieval and Ollama
        # can run in disjoint processes on modest WSL installations.
        self._reranker = None
        self._llm = get_llm(temperature=0.0)

    def _ensure_store(self):
        if self._store is None:
            local = self.rag_dir / "qdrant"
            self._store = load_vector_store(self.gene, self.host, self.port,
                                            local_path=local if local.exists() else None,
                                            layer=self.layer)
        if self._bm25 is None:
            self._bm25 = PersistentBM25.load(bm25_corpus_path(self.rag_dir, self.layer))
        if self._reranker is None:
            self._reranker = MedCPTReranker()

    def _ensure_context_store(self):
        """Load the review/guideline index, kept apart from the evidence index."""
        if self._context_store is None:
            local = self.rag_dir / "qdrant"
            self._context_store = load_vector_store(
                self.gene, self.host, self.port,
                local_path=local if local.exists() else None, layer=CONTEXT_LAYER)
        if self._context_bm25 is None:
            self._context_bm25 = PersistentBM25.load(bm25_corpus_path(self.rag_dir, CONTEXT_LAYER))
        if self._reranker is None:
            self._reranker = MedCPTReranker()

    def release_retrieval_models(self) -> None:
        """Release MedCPT/Qdrant memory before the local LLM is loaded.

        On modest WSL installations, keeping the document/query encoders, the
        cross-encoder and an 8B Ollama model resident together can exceed RAM.
        Retrieval results are persisted before this method is called.
        """
        for store in (self._store, self._context_store):
            close = getattr(getattr(store, "client", None), "close", None)
            if callable(close):
                close()
        self._store = None
        self._bm25 = None
        self._context_store = None
        self._context_bm25 = None
        self._reranker = None
        gc.collect()

    @staticmethod
    def _key(doc: Document) -> str:
        metadata = doc.metadata
        return "|".join(str(metadata.get(name, "")) for name in
                        ("source_sha256", "section", "section_chunk_index", "page_number", "page_chunk_index"))

    def _assert_evidence_layer(self, evidence: list[LiteratureEvidence]) -> list[LiteratureEvidence]:
        """Fail loudly if a non-primary passage reached mechanistic evidence.

        The layers are separate collections, so this can only trigger on a
        mis-built index.  Silently dropping the passage would hide that.
        """
        expected = self.layer or UNLAYERED
        intruders = sorted({item.corpus_layer for item in evidence if item.corpus_layer != expected})
        if intruders:
            raise RuntimeError(
                f"Mechanistic retrieval for layer {expected!r} returned passages from "
                f"{intruders}; rebuild the layered index"
            )
        return evidence

    def retrieve_context(self, query: str, k: int = 3) -> list[LiteratureEvidence]:
        """Retrieve reviews and guidelines for a separate contextual section.

        These passages are never merged into mechanistic evidence and never
        become citable evidence IDs for the LLM.
        """
        self._ensure_context_store()
        dense = retrieve_with_scores(self._context_store, query, self.candidate_k)
        lexical = self._context_bm25.search(query, self.candidate_k)
        dense_scores = {self._key(doc): score for doc, score in dense}
        bm25_scores = {self._key(doc): score for doc, score in lexical}
        documents, rrf_scores = {}, {}
        for results in (dense, lexical):
            for rank, (doc, _) in enumerate(results, 1):
                key = self._key(doc)
                documents[key] = doc
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [
            LiteratureEvidence(
                # Context IDs are deliberately not E-prefixed: they are outside
                # the whitelist the LLM is allowed to cite.
                evidence_id=f"C{index:02d}",
                pmid=str(documents[key].metadata.get("pmid", "unknown")),
                article_title=str(documents[key].metadata.get("title", "")) or None,
                doi=documents[key].metadata.get("doi"),
                publication=documents[key].metadata.get("publication"),
                publication_types=list(documents[key].metadata.get("publication_types", [])),
                publication_type=documents[key].metadata.get("publication_type"),
                passage=documents[key].page_content.strip()[:2500],
                retrieval_score=dense_scores.get(key), bm25_score=bm25_scores.get(key),
                rrf_score=rrf_scores.get(key),
                page_number=documents[key].metadata.get("page_number"),
                section=documents[key].metadata.get("section"),
                source_file=str(documents[key].metadata.get("source_file", "unknown")),
                parser=documents[key].metadata.get("parser"),
                source_collection=str(documents[key].metadata.get("source_collection", "literature_csv")),
                source_type=str(documents[key].metadata.get("source_type", "grobid_full_text")),
                full_text_available=bool(documents[key].metadata.get("full_text_available", False)),
                provenance=str(documents[key].metadata.get("provenance", "")),
                evidence_type=str(documents[key].metadata.get("evidence_type", UNLAYERED)),
                corpus_layer=str(documents[key].metadata.get("corpus_layer", UNLAYERED)),
            )
            for index, key in enumerate(ordered, 1)
        ]

    def retrieve(self, query: str, k: int) -> list[LiteratureEvidence]:
        self._ensure_store()
        dense = retrieve_with_scores(self._store, query, self.candidate_k)
        lexical = self._bm25.search(query, self.candidate_k)
        dense_scores = {self._key(doc): score for doc, score in dense}
        bm25_scores = {self._key(doc): score for doc, score in lexical}
        documents, rrf_scores = {}, {}
        for results in (dense, lexical):
            for rank, (doc, _) in enumerate(results, 1):
                key = self._key(doc)
                documents[key] = doc
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
        fused_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:self.candidate_k]
        reranked = self._reranker.rerank(query, [documents[key] for key in fused_keys], top_k=k)
        result = []
        for index, (doc, rerank_score) in enumerate(reranked, 1):
            key = self._key(doc)
            result.append(LiteratureEvidence(
                evidence_id=_evidence_id(index),
                pmid=str(doc.metadata.get("pmid", "unknown")),
                article_title=str(doc.metadata.get("title", "")) or None,
                doi=doc.metadata.get("doi"),
                publication=doc.metadata.get("publication"),
                publication_types=list(doc.metadata.get("publication_types", [])),
                publication_type=doc.metadata.get("publication_type"),
                passage=doc.page_content.strip()[:2500],
                retrieval_score=dense_scores.get(key),
                bm25_score=bm25_scores.get(key),
                rrf_score=rrf_scores.get(key),
                reranker_score=rerank_score,
                page_number=doc.metadata.get("page_number"),
                section=doc.metadata.get("section"),
                source_file=str(doc.metadata.get("source_file", "unknown")),
                parser=doc.metadata.get("parser"),
                source_collection=str(doc.metadata.get("source_collection", "literature_csv")),
                source_type=str(doc.metadata.get("source_type", "grobid_full_text")),
                full_text_available=bool(doc.metadata.get("full_text_available", False)),
                provenance=str(doc.metadata.get("provenance", "")),
                evidence_type=str(doc.metadata.get("evidence_type", UNLAYERED)),
                corpus_layer=str(doc.metadata.get("corpus_layer", UNLAYERED)),
            ))
        return self._assert_evidence_layer(result)

    def infer(self, request: MechanisticQuery) -> MechanisticRAGResult:
        query = (
            f"{request.gene} {request.variant} {request.mechanism}; "
            f"residue region interface oligomerization structural mechanism"
        )
        evidence = self.retrieve(query, request.k)
        if not evidence:
            inference = MechanisticLiteratureInference(
                variant=request.variant, gene=request.gene.upper(), mechanism=request.mechanism,
                verdict="insufficient_evidence", predicted_direction="unclear",
                rationale="No passages were retrieved from the indexed corpus.", cited_evidence_ids=[],
                uncertainty="The local literature corpus may be incomplete.",
            )
            return MechanisticRAGResult(inference=inference, evidence=[], retrieval_query=query)

        context = "\n\n".join(
            f"[{e.evidence_id}] {e.pmid}, page {e.page_number}, section {e.section}:\n{e.passage}"
            for e in evidence
        )
        structured_llm = self._llm.with_structured_output(MechanisticLiteratureInference)
        payload = {
            "context": context, "gene": request.gene.upper(), "variant": request.variant,
            "mechanism": request.mechanism, "structural_observation": request.structural_observation,
        }
        allowed = {e.evidence_id for e in evidence}
        last_error = None
        for _ in range(3):
            try:
                inference = (MECHANISTIC_PROMPT | structured_llm).invoke(payload)
                if inference.gene.upper() != request.gene.upper() or inference.variant != request.variant:
                    raise ValueError("LLM changed the requested gene or variant")
                inference = inference.model_copy(update={
                    "cited_evidence_ids": _normalize_citation_ids(
                        inference.cited_evidence_ids, allowed
                    )
                })
                if inference.verdict != "insufficient_evidence" and not inference.cited_evidence_ids:
                    raise ValueError("A supported/contradicted verdict must cite retrieved evidence")
                break
            except Exception as error:
                last_error = error
        else:
            inference = MechanisticLiteratureInference(
                variant=request.variant, gene=request.gene.upper(), mechanism=request.mechanism,
                verdict="insufficient_evidence", predicted_direction="unclear",
                rationale=("Automated inference abstained because the local LLM did not satisfy "
                           "the structured citation contract after three attempts."),
                cited_evidence_ids=[],
                uncertainty=(f"Last contract error: {type(last_error).__name__}: {last_error}. "
                             "Retrieved passages remain available for human review."),
                requires_experimental_validation=True,
            )
            return MechanisticRAGResult(
                inference=inference, evidence=evidence, retrieval_query=query,
                llm_contract_status="abstained_after_invalid_output",
            )
        return MechanisticRAGResult(
            inference=inference, evidence=evidence, retrieval_query=query,
            llm_contract_status="passed",
        )


class DiscoveryLiteratureAnalyzer(LiteratureMechanismAnalyzer):
    """Multi-route, PMID-diverse retrieval and possibility-oriented inference."""

    retrieval_version = "discovery_v3"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_version = "gsdmd_discovery_v1" if self.gene == "GSDMD" else "discovery_v5"
        self.retrieval_stack_version = "rag_stack_v6" if self.gene == "BRAF" else "rag_stack_v4"
        # Ten passages plus a nested audit contract exceed the legacy 4k setup.
        self._llm = get_llm(temperature=0.0, num_ctx=8192, num_predict=1024)

    def _channel_queries(self, request: DiscoveryQuery,
                         channels: Sequence[str] | None = None) -> dict[str, list[str]]:
        """Group queries by evidence channel so they can be scored separately.

        A mechanistic question must not be answered with therapeutic response
        data, so ``pharmacology`` is off unless the mechanism under study is
        resistance or drug binding.  Channels stay separate all the way through
        fusion, which is what makes the contamination auditable afterwards.
        """
        gene = request.gene.upper()
        profile = get_condition_profile(request.condition_profile)
        conditions = normalized_condition_terms(profile, request.condition_terms)
        condition_query = " OR ".join(f'"{term}"' for term in conditions)
        active = tuple(channels) if channels is not None else self.active_channels(request)
        if gene == "GSDMD":
            partner = request.ppi_partner or "GSDMD oligomerization interface"
            cells = " OR ".join(profile.cell_context) or "cell tissue"
            by_channel = {
                "variant_exact": [
                    f'GSDMD {request.variant} mutation molecular mechanism',
                    f'GSDMD {request.variant} functional assay',
                ],
                "conformational_state": [
                    f'GSDMD {request.variant} autoinhibition caspase cleavage N-terminal C-terminal',
                ],
                "structure_motif": [
                    f'GSDMD residue {request.position} {request.functional_region} structure function',
                    f'GSDMD {request.variant} membrane lipid beta hairpin prepore pore',
                ],
                "ppi_dimerization": [
                    f'GSDMD {request.variant} oligomerization protein interaction {partner}',
                ],
                "disease": [
                    f'GSDMD {request.variant} ({condition_query})',
                    f'GSDMD pyroptosis ({cells}) ({condition_query})',
                    f'GSDMD {request.variant} IL-1beta IL-18 neuroinflammation',
                ],
                "pharmacology": [
                    f'GSDMD {request.variant} inhibitor disulfiram binding resistance',
                ],
            }
        else:
            partner = request.ppi_partner or f"{gene} protein interaction"
            by_channel = {
                "variant_exact": [
                    f'{gene} {request.variant} molecular mechanism mutation',
                    f'{gene} {request.variant} functional assay',
                ],
                "conformational_state": [
                    f'{gene} {request.variant} conformation {request.functional_region}',
                ],
                "structure_motif": [
                    f'{gene} residue {request.position} {request.functional_region} '
                    f'charge volume structure function',
                ],
                "ppi_dimerization": [
                    f'{gene} {request.variant} protein interaction dimerization {partner}',
                ],
                "disease": [
                    f'{gene} {request.variant} ({condition_query})',
                ],
                "pharmacology": [
                    f'{gene} {request.variant} inhibitor binding drug resistance',
                ],
            }
        # Compatibility alias for callers from the pre-curation API.  New
        # manifests use the three structural routes separately, while an old
        # explicit request for ``structural_mechanism`` still has a defined
        # meaning and cannot silently return an empty query set.
        by_channel["structural_mechanism"] = (
            by_channel.get("variant_exact", [])
            + by_channel.get("conformational_state", [])
            + by_channel.get("structure_motif", [])
        )
        return {channel: by_channel[channel] for channel in active if by_channel.get(channel)}

    @staticmethod
    def active_channels(request: DiscoveryQuery) -> tuple[str, ...]:
        """Channels allowed for this request; pharmacology is opt-in.

        The decision is explicit — ``pharmacology_requested``, or a mechanism
        stated in drug terms.  It is never inferred from the structural feature
        dump, where a word like "binding" describes a nucleotide site rather
        than a drug.
        """
        if bool(getattr(request, "pharmacology_requested", False)):
            return MECHANISTIC_CHANNELS + ("pharmacology",)
        mechanism = str(getattr(request, "mechanism", "") or "")
        if mechanism and _PHARMACOLOGY_PATTERN.search(mechanism):
            return MECHANISTIC_CHANNELS + ("pharmacology",)
        return MECHANISTIC_CHANNELS

    def _queries(self, request: DiscoveryQuery) -> list[str]:
        """Flattened active queries, in channel order."""
        return [query for queries in self._channel_queries(request).values() for query in queries]

    def corpus_document_sizes(self) -> dict[str, int]:
        """Chunks per PMID across the whole indexed layer, cached.

        This is the document-length baseline the diagnostics compare against.
        """
        if self._document_sizes is None:
            self._ensure_store()
            counts: dict[str, int] = {}
            for document in self._bm25.documents:
                pmid = str(document.metadata.get("pmid", "unknown"))
                counts[pmid] = counts.get(pmid, 0) + 1
            self._document_sizes = counts
        return self._document_sizes

    def _retrieve_discovery_v5(
        self, queries: list[str] | dict[str, list[str]], k: int = 10,
        max_per_document: int = MAX_PASSAGES_PER_DOCUMENT, variant: str | None = None,
        functional_region: str = "", reranker_query: str | None = None,
    ) -> list[LiteratureEvidence]:
        """Retrieve with channel-local ranks and a monotonic lexical scope."""
        self._ensure_store()
        by_channel = ({"unchannelled": list(queries)} if not isinstance(queries, dict)
                      else {channel: list(items) for channel, items in queries.items()})
        flat = [(channel, query) for channel, items in by_channel.items() for query in items]
        dense_batches = retrieve_many_with_scores(self._store, [q for _, q in flat], self.candidate_k)
        chunks: dict[str, Document] = {}
        chunks_by_document: dict[str, dict[str, None]] = {}
        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        rankings: dict[str, list] = {}
        normalized_by_channel: dict[str, dict[str, float]] = {}
        normalized_fused: dict[str, float] = {}
        exact_lexical_hits = 0
        for query_index, ((channel, query), dense) in enumerate(zip(flat, dense_batches, strict=True)):
            lexical = self._bm25.search(query, self.candidate_k)
            if channel == "variant_exact" and variant:
                lexical = [(doc, score) for doc, score in lexical
                            if contains_exact_variant(doc.page_content, variant)]
                exact_lexical_hits += len(lexical)
            for method, results in (("medcpt", dense), ("bm25", lexical)):
                kept = [(doc, score) for doc, score in results
                        if str(doc.metadata.get("gene", "")).upper() == self.gene]
                ranking = document_ranking(kept, self._key,
                                           lambda d: d.metadata.get("pmid", "unknown"))
                rankings[f"{channel}:{query_index}:{method}"] = ranking
                normalized = normalized_rank_scores(
                    [(pmid, score) for pmid, _key, score in ranking]
                )
                normalized_by_channel[f"{channel}:{query_index}:{method}"] = normalized
                for pmid, value in normalized.items():
                    normalized_fused[pmid] = normalized_fused.get(pmid, 0.0) + value
                for doc, score in kept:
                    key = self._key(doc)
                    chunks[key] = doc
                    pmid = str(doc.metadata.get("pmid", "unknown"))
                    chunks_by_document.setdefault(pmid, {})[key] = None
                    target = dense_scores if method == "medcpt" else bm25_scores
                    target[key] = max(target.get(key, float("-inf")), float(score))
        fused, contributions = rrf_over_documents(rankings)
        candidate_documents = sorted(fused, key=fused.get, reverse=True)[:DOCUMENT_CANDIDATES]
        candidate_chunks = [chunks[key] for pmid in candidate_documents
                            for key in chunks_by_document.get(pmid, {})]
        question = reranker_query or " ; ".join(query for _, query in flat)
        reranked = self._reranker.rerank(question, candidate_chunks,
                                         top_k=len(candidate_chunks)) if candidate_chunks else []
        rerank_scores = {self._key(doc): float(score) for doc, score in reranked}
        ranked_by_document: dict[str, list[tuple[Document, float, str, bool]]] = {}
        for doc, score in reranked:
            pmid = str(doc.metadata.get("pmid", "unknown"))
            scope = classify_evidence_scope(doc.page_content, variant or "", functional_region)
            ranked_by_document.setdefault(pmid, []).append(
                (doc, score, scope, bool(doc.metadata.get("full_text_available", False)))
            )
        document_order = sorted(
            ranked_by_document,
            key=lambda pmid: (
                SCOPE_PRIORITY.get(strongest_scope(item[2] for item in ranked_by_document[pmid]), 0),
                max(int(item[3]) for item in ranked_by_document[pmid]),
                normalized_fused.get(pmid, 0.0),
                max(item[1] for item in ranked_by_document[pmid]),
            ), reverse=True,
        )
        scoped_chunks = {
            pmid: [item[0] for item in sorted(
                items,
                key=lambda item: (SCOPE_PRIORITY.get(item[2], 0), int(item[3]), item[1]),
                reverse=True,
            )]
            for pmid, items in ranked_by_document.items()
        }
        chosen = select_passages(document_order, scoped_chunks, k, max_per_document)
        selected_pmids = {str(doc.metadata.get("pmid", "unknown")) for doc in chosen}
        if len(candidate_documents) >= 3 and len(selected_pmids) < min(3, k):
            raise RuntimeError(
                "retrieval diversity contract failed: fewer than three distinct PMIDs "
                "were returned despite three candidate documents"
            )
        channel_of = lambda label: str(label).split(":", 1)[0]
        selected_scopes = [classify_evidence_scope(doc.page_content, variant or "", functional_region)
                           for doc in chosen]
        self.last_retrieval_diagnostics = {
            **retrieval_diagnostics(
                [str(doc.metadata.get("pmid", "unknown")) for doc in chosen],
                fused, contributions, self.corpus_document_sizes()),
            "retrieval_stack_version": "rag_stack_v5",
            "queries_by_channel": {channel: len(items) for channel, items in by_channel.items()},
            "channels_used": sorted(by_channel),
            "max_passages_per_document": max_per_document,
            "global_pmid_cap_between_variants": None,
            "exact_channel_lexical_hits": exact_lexical_hits,
            "rank_normalization": "within_channel_and_method",
            "reranker_query": question,
            "scope_order": list(SCOPE_PRIORITY),
            "selected_scope_counts": {
                scope: selected_scopes.count(scope) for scope in SCOPE_PRIORITY
            },
            "candidate_scope_counts": {
                scope: sum(
                    1 for items in ranked_by_document.values()
                    for item in items if item[2] == scope
                ) for scope in SCOPE_PRIORITY
            },
            "channel_method_normalized_rank_counts": {
                key: len(value) for key, value in normalized_by_channel.items()
            },
            "normalized_fused_score_range": {
                "min": min(normalized_fused.values(), default=0.0),
                "max": max(normalized_fused.values(), default=0.0),
            },
        }
        result = []
        for index, doc in enumerate(chosen, 1):
            key = self._key(doc)
            pmid = str(doc.metadata.get("pmid", "unknown"))
            result.append(LiteratureEvidence(
                evidence_id=_evidence_id(index), pmid=pmid,
                article_title=str(doc.metadata.get("title", "")) or None,
                doi=doc.metadata.get("doi"), publication=doc.metadata.get("publication"),
                publication_types=list(doc.metadata.get("publication_types", [])),
                publication_type=doc.metadata.get("publication_type"),
                passage=doc.page_content.strip()[:2500],
                retrieval_score=dense_scores.get(key), bm25_score=bm25_scores.get(key),
                rrf_score=fused.get(pmid), reranker_score=rerank_scores.get(key),
                page_number=doc.metadata.get("page_number"), section=doc.metadata.get("section"),
                source_file=str(doc.metadata.get("source_file", "unknown")),
                parser=doc.metadata.get("parser"),
                source_collection=str(doc.metadata.get("source_collection", "literature_csv")),
                source_type=str(doc.metadata.get("source_type", "grobid_full_text")),
                full_text_available=bool(doc.metadata.get("full_text_available", False)),
                provenance=str(doc.metadata.get("provenance", "")),
                evidence_type=str(doc.metadata.get("evidence_type", UNLAYERED)),
                corpus_layer=str(doc.metadata.get("corpus_layer", UNLAYERED)),
                retrieval_channels=sorted({channel_of(label) for label in contributions.get(pmid, {})}),
                evidence_scope=classify_evidence_scope(doc.page_content, variant or "", functional_region),
            ))
        return self._assert_evidence_layer(result)

    def _retrieve_discovery_v6(
        self, queries: list[str] | dict[str, list[str]], k: int = 10,
        max_per_document: int = MAX_PASSAGES_PER_DOCUMENT, variant: str | None = None,
        functional_region: str = "", reranker_query: str | None = None,
    ) -> list[LiteratureEvidence]:
        """Return mechanistic evidence and context as separate blocks.

        The V5 ranker is reused only as a candidate/reranking engine.  It is
        asked for a larger candidate panel, then general-context passages are
        removed from the mechanistic block rather than being used to fill it.
        Both blocks remain available in the cache for explanation.
        """
        all_items = self._retrieve_discovery_v5(
            queries, max(k, 30), max_per_document, variant, functional_region, reranker_query,
        )
        mechanistic, context = split_evidence_blocks(all_items)
        mechanistic = mechanistic[:k]
        context = context[:k]
        self.last_retrieval_blocks = {
            "mechanistic_evidence": mechanistic,
            "context_reference": context,
        }
        self.last_retrieval_diagnostics = {
            **self.last_retrieval_diagnostics,
            "retrieval_stack_version": "rag_stack_v6",
            "mechanistic_evidence_count": len(mechanistic),
            "context_reference_count": len(context),
            "mechanistic_evidence_status": (
                "mechanistic_evidence_retrieved" if mechanistic
                else "no_mechanistic_evidence_retrieved"
            ),
            "context_can_fill_mechanistic_top_k": False,
            "mechanistic_scopes": sorted(MECHANISTIC_SCOPES),
        }
        # Compatibility callers still receive one auditable union; all new
        # caches persist the two blocks separately and benchmark only the first.
        return mechanistic + context

    def retrieve_discovery(self, queries: list[str] | dict[str, list[str]], k: int = 10,
                           max_per_document: int = MAX_PASSAGES_PER_DOCUMENT,
                           variant: str | None = None, functional_region: str = "",
                           reranker_query: str | None = None) -> list[LiteratureEvidence]:
        """Fuse at document level, rerank documents, then pick passages.

        A flat query list is accepted and treated as one unlabelled channel, so
        callers that do not separate channels keep working.
        """
        if self.retrieval_version == "discovery_v5":
            return self._retrieve_discovery_v6(
                queries, k, max_per_document, variant, functional_region, reranker_query,
            )
        if self.retrieval_version == "discovery_v4":
            return self._retrieve_discovery_v5(
                queries, k, max_per_document, variant, functional_region, reranker_query,
            )
        self._ensure_store()
        by_channel = ({"unchannelled": list(queries)} if not isinstance(queries, dict)
                      else {channel: list(items) for channel, items in queries.items()})
        flat = [(channel, query) for channel, items in by_channel.items() for query in items]
        dense_batches = retrieve_many_with_scores(self._store, [q for _, q in flat],
                                                  self.candidate_k)

        chunks: dict[str, Document] = {}
        dense_scores, bm25_scores, rankings = {}, {}, {}
        chunks_by_document: dict[str, dict[str, None]] = {}
        for (channel, query), dense in zip(flat, dense_batches, strict=True):
            lexical = self._bm25.search(query, self.candidate_k)
            for method, results in (("dense", dense), ("bm25", lexical)):
                kept = [(doc, score) for doc, score in results
                        if str(doc.metadata.get("gene", "")).upper() == self.gene]
                for doc, score in kept:
                    key = self._key(doc)
                    chunks[key] = doc
                    pmid = str(doc.metadata.get("pmid", "unknown"))
                    chunks_by_document.setdefault(pmid, {})[key] = None
                    target = dense_scores if method == "dense" else bm25_scores
                    target[key] = max(target.get(key, float("-inf")), float(score))
                # Each method contributes one entry per document, so a long paper
                # cannot occupy several slots of the same ranking.
                label = f"{channel}:{query[:24]}:{method}"
                rankings[label] = document_ranking(kept, self._key,
                                                   lambda d: d.metadata.get("pmid", "unknown"))
        fused, contributions = rrf_over_documents(rankings)
        candidate_documents = sorted(fused, key=fused.get, reverse=True)[:DOCUMENT_CANDIDATES]

        combined_query = " ; ".join(query for _, query in flat)
        candidate_chunks = [chunks[key] for pmid in candidate_documents
                            for key in chunks_by_document.get(pmid, {})]
        reranked = self._reranker.rerank(combined_query, candidate_chunks,
                                         top_k=len(candidate_chunks)) if candidate_chunks else []
        rerank_scores = {self._key(doc): float(score) for doc, score in reranked}
        ranked_by_document: dict[str, list] = {}
        for doc, score in reranked:
            ranked_by_document.setdefault(str(doc.metadata.get("pmid", "unknown")), []).append(doc)
        # A document is worth what its best passage is worth, not what its
        # passage count is worth.
        document_order = sorted(
            ranked_by_document,
            key=lambda pmid: rerank_scores[self._key(ranked_by_document[pmid][0])],
            reverse=True,
        )
        chosen = select_passages(document_order, ranked_by_document, k, max_per_document)
        selected_pmids = {str(doc.metadata.get("pmid", "unknown")) for doc in chosen}
        # The diversity contract is conditional: if at least three candidate
        # papers exist, the first result must expose three distinct PMIDs.
        # This is checked after reranking, where a hidden fallback cannot
        # silently collapse the result back onto one long paper.
        if len(candidate_documents) >= 3 and len(selected_pmids) < min(3, k):
            raise RuntimeError(
                "retrieval diversity contract failed: fewer than three distinct PMIDs "
                "were returned despite three candidate documents"
            )
        selected = [(doc, rerank_scores[self._key(doc)]) for doc in chosen]
        channel_of = lambda label: str(label).split(":", 1)[0]

        self.last_retrieval_diagnostics = {
            **retrieval_diagnostics(
                [str(doc.metadata.get("pmid", "unknown")) for doc, _ in selected],
                fused, contributions, self.corpus_document_sizes()),
            "queries_by_channel": {channel: len(items) for channel, items in by_channel.items()},
            "channels_used": sorted(by_channel),
            "max_passages_per_document": max_per_document,
            "distinct_pmids": len(selected_pmids),
            "minimum_distinct_pmids_when_available": min(3, k),
            "document_candidates_reranked": len(candidate_documents),
        }
        result = []
        for index, (doc, rerank_score) in enumerate(selected, 1):
            key = self._key(doc)
            result.append(LiteratureEvidence(
                evidence_id=_evidence_id(index), pmid=str(doc.metadata.get("pmid", "unknown")),
                article_title=str(doc.metadata.get("title", "")) or None,
                doi=doc.metadata.get("doi"),
                publication=doc.metadata.get("publication"),
                publication_types=list(doc.metadata.get("publication_types", [])),
                publication_type=doc.metadata.get("publication_type"),
                passage=doc.page_content.strip()[:2500], retrieval_score=dense_scores.get(key),
                bm25_score=bm25_scores.get(key),
                rrf_score=fused.get(str(doc.metadata.get("pmid", "unknown"))),
                reranker_score=rerank_score, page_number=doc.metadata.get("page_number"),
                section=doc.metadata.get("section"),
                source_file=str(doc.metadata.get("source_file", "unknown")),
                parser=doc.metadata.get("parser"),
                source_collection=str(doc.metadata.get("source_collection", "literature_csv")),
                source_type=str(doc.metadata.get("source_type", "grobid_full_text")),
                full_text_available=bool(doc.metadata.get("full_text_available", False)),
                provenance=str(doc.metadata.get("provenance", "")),
                evidence_type=str(doc.metadata.get("evidence_type", UNLAYERED)),
                corpus_layer=str(doc.metadata.get("corpus_layer", UNLAYERED)),
            ))
        return self._assert_evidence_layer(result)

    @staticmethod
    def _deterministic(request: DiscoveryQuery, evidence: list[LiteratureEvidence]) -> DiscoveryInference:
        exact = [(item, " ".join(local_variant_windows(item.passage, request.variant)).lower())
                 for item in evidence if contains_exact_variant(item.passage, request.variant)]
        hypotheses = []
        gene = request.gene.upper()
        for item, passage in exact:
            if not _mechanistic(passage, gene):
                continue
            if gene == "BRAF" and "inactive" in passage and "active" in passage:
                hypotheses.append(DiscoveryHypothesis(
                    mechanism="conformational_equilibrium", target="BRAF kinase",
                    predicted_direction="shift_toward_active_state", conformational_state="inactive_to_active",
                    classification="literature_reported",
                    rationale="The cited passage explicitly links the exact variant to a shift from inactive toward active kinase conformation.",
                    cited_evidence_ids=[item.evidence_id], rule_ids=[],
                ))
                break
            if gene == "GSDMD" and any(term in passage for term in ("cleavage", "cleaved")):
                hypotheses.append(DiscoveryHypothesis(
                    mechanism="proteolytic_activation_or_inactivation", target="GSDMD cleavage control",
                    predicted_direction="alteration_reported", conformational_state="cleavage_control",
                    classification="literature_reported",
                    rationale="The cited passage explicitly discusses the exact variant in a GSDMD cleavage context.",
                    cited_evidence_ids=[item.evidence_id], rule_ids=[],
                ))
                break
            if gene == "GSDMD" and any(term in passage for term in ("pore", "oligomer", "membrane", "pyroptosis")):
                hypotheses.append(DiscoveryHypothesis(
                    mechanism="pore_formation_or_oligomerization", target="GSDMD pore-forming activity",
                    predicted_direction="alteration_reported", conformational_state="pore_pathway",
                    classification="literature_reported",
                    rationale="The cited passage explicitly discusses the exact variant in a pore, oligomerization, membrane or pyroptosis context.",
                    cited_evidence_ids=[item.evidence_id], rule_ids=[],
                ))
                break
        if not hypotheses:
            for item, passage in exact:
                if any(term in passage for term in ("kinase activity", "dimer", "oligomer", "mek phosphorylation")):
                    hypotheses.append(DiscoveryHypothesis(
                        mechanism=("kinase_or_interaction_regulation" if gene == "BRAF" else
                                   "protein_interaction_regulation"),
                        target=("BRAF signaling" if gene == "BRAF" else f"{gene} function"),
                        predicted_direction="alteration_reported", classification="literature_reported",
                        rationale="The cited passage explicitly discusses the exact variant in a mechanistic signaling context.",
                        cited_evidence_ids=[item.evidence_id], rule_ids=[],
                    ))
                    break
        if not hypotheses:
            for item in evidence:
                passage = item.passage.lower()
                if same_residue_variant_mentioned(passage, request.variant) and _mechanistic(passage, gene):
                    hypotheses.append(DiscoveryHypothesis(
                        mechanism=("protein_interaction_regulation" if gene == "GSDMD" else
                                   "kinase_or_interaction_regulation"),
                        target=f"{gene} residue {request.position}",
                        predicted_direction="alteration_possible",
                        classification="literature_supported_hypothesis",
                        rationale=("A cited mechanistic passage describes another substitution at the same "
                                   "residue; transfer to the queried substitution remains indirect."),
                        cited_evidence_ids=[item.evidence_id], rule_ids=[],
                    ))
                    break
        if not hypotheses:
            aliases = _REGION_ALIASES.get(
                request.functional_region,
                (request.functional_region.replace("_", " "),),
            )
            for item in evidence:
                passage = item.passage.lower()
                if any(alias in passage for alias in aliases) and _mechanistic(passage, gene):
                    hypotheses.append(DiscoveryHypothesis(
                        mechanism=("pore_or_cleavage_regulation" if gene == "GSDMD" else
                                   "kinase_or_interaction_regulation"),
                        target=f"{gene} {request.functional_region}",
                        predicted_direction="alteration_possible",
                        classification="literature_supported_hypothesis",
                        rationale=("A cited passage supports a mechanism for the same functional region, "
                                   "without establishing an exact-variant effect."),
                        cited_evidence_ids=[item.evidence_id], rule_ids=[],
                    ))
                    break
        profile = get_condition_profile(request.condition_profile)
        disease_terms = tuple(term.lower() for term in normalized_condition_terms(profile, request.condition_terms))
        if gene == "BRAF" and request.condition_profile == "braf_oncology":
            disease_terms += ("cancer", "tumor", "tumour", "melanoma", "oncogene", "malignan")
        relation_terms = ("associated", "association", "drives", "causes", "patients", "mutation in",
                          "risk", "susceptibility", "enriched", "identified in")
        disease = DiseaseAssociation()
        for item, _ in exact:
            support = _condition_support_windows(item.passage, request.variant, disease_terms)
            if support:
                passage, matched_terms = support[0]
                associated = any(term in passage for term in relation_terms)
                disease = DiseaseAssociation(
                    status="associated" if associated else "cooccurrence",
                    disease_context=", ".join(matched_terms[:3]),
                    rationale=("The exact variant and a disease context occur in the cited passage; "
                               "relationship wording was detected." if associated else
                               "The exact variant co-occurs with a disease term without a directional relationship."),
                    cited_evidence_ids=[item.evidence_id],
                )
                break
        try:
            rules = json.loads(request.biophysical_hypotheses_json)
        except Exception:
            rules = []
        for rule in rules:
            if len(hypotheses) >= 3:
                break
            hypotheses.append(DiscoveryHypothesis(**rule))
        return DiscoveryInference(
            variant=request.variant, gene=request.gene.upper(), disease_association=disease,
            hypotheses=hypotheses[:3], evidence_conflict_flag=False,
            uncertainty="Discovery hypotheses prioritize sensitivity and require experimental validation.",
        )

    def infer_discovery(self, request: DiscoveryQuery,
                        evidence: list[LiteratureEvidence] | None = None,
                        retrieval_queries: list[str] | None = None) -> DiscoveryRAGResult:
        queries = retrieval_queries or self._queries(request)
        evidence = evidence if evidence is not None else self.retrieve_discovery(queries, request.k)
        mechanistic_evidence, context_reference = split_evidence_blocks(evidence)
        mechanistic_status = (
            "mechanistic_evidence_retrieved" if mechanistic_evidence
            else "no_mechanistic_evidence_retrieved"
        )
        fallback = self._deterministic(request, evidence)
        fallback = _attach_evidence_scopes(fallback, evidence)
        if not evidence:
            return DiscoveryRAGResult(
                inference=fallback, evidence=[], retrieval_queries=queries,
                mechanistic_evidence=[], context_reference=[],
                mechanistic_evidence_status=mechanistic_status,
                retrieval_version=self.retrieval_version,
                condition_profile=request.condition_profile, condition_terms=request.condition_terms,
                query_version=request.query_version, llm_contract_status="deterministic_fallback",
                inference_status="no_evidence", evidence_status="none",
            )
        context = "\n\n".join(
            f"[{e.evidence_id}] PMID {e.pmid}; evidence_scope={e.evidence_scope}; "
            f"source={e.source_type}; section={e.section}:\n{e.passage}"
            for e in evidence
        )
        payload = {"context": context, "biophysical_hypotheses": request.biophysical_hypotheses_json,
                   "gene": request.gene.upper(), "variant": request.variant, "position": request.position,
                   "functional_region": request.functional_region,
                   "structural_observation": request.structural_observation,
                   "condition_profile": request.condition_profile,
                   "condition_terms": ", ".join(request.condition_terms)}
        structured = self._llm.with_structured_output(DiscoveryInference)
        allowed = {e.evidence_id for e in evidence}
        last_error = None
        retries = 0
        for attempt in range(3):
          retries = attempt
          try:
            inference = (DISCOVERY_PROMPT | structured).invoke(payload)
            if inference.variant != request.variant or inference.gene.upper() != request.gene.upper():
                raise ValueError("LLM changed requested gene or variant")
            disease_ids = _normalize_citation_ids(inference.disease_association.cited_evidence_ids, allowed)
            if inference.disease_association.status != "none":
                profile = get_condition_profile(request.condition_profile)
                disease_terms = tuple(
                    term.lower() for term in normalized_condition_terms(profile, request.condition_terms)
                )
                if request.gene.upper() == "BRAF" and request.condition_profile == "braf_oncology":
                    disease_terms += ("cancer", "tumor", "tumour", "melanoma", "oncogene", "malignan")
                supports = [
                    support for item in evidence if item.evidence_id in disease_ids
                    for support in _condition_support_windows(item.passage, request.variant, disease_terms)
                ]
                if not supports:
                    raise ValueError("Variant-disease claim lacks a local exact-variant/condition passage")
                if inference.disease_association.status == "associated":
                    relation_terms = ("associated", "association", "drives", "causes", "patients",
                                      "mutation in", "risk", "susceptibility", "enriched", "identified in")
                    if not any(any(term in window for term in relation_terms) for window, _ in supports):
                        raise ValueError("Associated claim lacks local relationship wording")
            hypotheses = []
            for hypothesis in inference.hypotheses[:3]:
                ids = _normalize_citation_ids(hypothesis.cited_evidence_ids, allowed)
                if hypothesis.classification.startswith("literature") and not ids:
                    raise ValueError("Literature hypothesis has no citation")
                cited_scopes = [item.evidence_scope for item in evidence if item.evidence_id in ids]
                strongest = strongest_scope(cited_scopes)
                if hypothesis.classification == "literature_reported":
                    if any(scope != "exact_variant" for scope in cited_scopes):
                        raise ValueError("Exact literature claim cites lower-scope evidence")
                    supporting_items = [item for item in evidence if item.evidence_id in ids]
                    supported = any(
                        _exact_mechanistic_windows(item.passage, request.variant, request.gene)
                        for item in supporting_items
                    )
                    if not supported:
                        raise ValueError("Exact literature claim lacks a local exact-variant mechanism")
                    if supporting_items and all(_therapy_context_only(item) for item in supporting_items):
                        raise ValueError("Therapeutic context cannot sustain a mechanistic claim")
                if hypothesis.classification == "literature_supported_hypothesis" and strongest == "general_context":
                    hypothesis = hypothesis.model_copy(update={"classification": "context_only"})
                hypotheses.append(hypothesis.model_copy(update={
                    "cited_evidence_ids": ids,
                    "evidence_scope": strongest,
                }))
            inference = inference.model_copy(update={
                "disease_association": inference.disease_association.model_copy(
                    update={"cited_evidence_ids": disease_ids}), "hypotheses": hypotheses,
            })
            # Deterministic exact claims are recall safeguards. Merge them before
            # rule-only hypotheses when the local model omits an explicit passage.
            merged = list(inference.hypotheses)
            for hypothesis in fallback.hypotheses:
                signature = (hypothesis.mechanism, hypothesis.classification)
                if signature not in {(h.mechanism, h.classification) for h in merged}:
                    merged.append(hypothesis)
            disease = (fallback.disease_association if
                       inference.disease_association.status == "none" and
                       fallback.disease_association.status != "none" else inference.disease_association)
            inference = inference.model_copy(update={"hypotheses": merged[:3], "disease_association": disease})
            status = "passed"
            break
          except Exception as error:
            last_error = error
            retries = attempt + 1
        else:
            inference, status = fallback, "deterministic_fallback"
        exact_ids = {
            evidence_id for hypothesis in inference.hypotheses
            if hypothesis.classification == "literature_reported"
            for evidence_id in hypothesis.cited_evidence_ids
        }
        indirect_ids = {
            evidence_id for hypothesis in inference.hypotheses
            if hypothesis.classification == "literature_supported_hypothesis"
            for evidence_id in hypothesis.cited_evidence_ids
        }
        if exact_ids:
            evidence_status = "exact"
        elif indirect_ids:
            evidence_status = "indirect"
        else:
            evidence_status = "none"
        inference_status = "llm_validated" if status == "passed" else "technical_llm_failure"
        return DiscoveryRAGResult(
            inference=inference, evidence=evidence,
            mechanistic_evidence=mechanistic_evidence,
            context_reference=context_reference,
            mechanistic_evidence_status=mechanistic_status,
            retrieval_queries=queries,
            retrieval_version=self.retrieval_version,
            condition_profile=request.condition_profile, condition_terms=request.condition_terms,
            query_version=request.query_version, llm_contract_status=status,
            inference_status=inference_status, evidence_status=evidence_status,
            llm_retry_count=retries,
            llm_failure_reason=(None if status == "passed" or last_error is None else str(last_error)),
        )
