"""Contract tests for PubMed record parsing and deterministic corpus layering."""

from __future__ import annotations

import pytest

from healthrag.literature.corpus_layers import (
    assign_layer, build_corpus_layers, gene_link_status,
)
from healthrag.literature.pubmed_records import PubMedRecord, parse_pubmed_articles

BRAF_ALIASES = ["BRAF", "B-RAF", "P15056"]
BRAF_MESH = ["Proto-Oncogene Proteins B-raf"]


def record(pmid: str = "1", **fields) -> PubMedRecord:
    base = {"pmid": pmid, "title": "BRAF V600E kinase activity", "abstract": "Primary study.",
            "publication_types": ["Journal Article"]}
    return PubMedRecord(**{**base, **fields})


def layer_of(rec: PubMedRecord, **kwargs) -> str:
    return assign_layer(rec, BRAF_ALIASES, BRAF_MESH, **kwargs).layer


PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <DateRevised><Year>2024</Year><Month>03</Month><Day>02</Day></DateRevised>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2019</Year><Month>May</Month></PubDate></JournalIssue>
          <Title>Journal of Testing</Title>
          <ISOAbbreviation>J Test</ISOAbbreviation>
        </Journal>
        <ArticleTitle>BRAF V600E drives dimerization</ArticleTitle>
        <Abstract><AbstractText Label="RESULTS">The mutant dimerizes.</AbstractText></Abstract>
        <AuthorList>
          <Author><LastName>Silva</LastName><ForeName>Ana</ForeName>
            <AffiliationInfo><Affiliation>Universidade X</Affiliation></AffiliationInfo>
          </Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType UI="D016428">Journal Article</PublicationType>
          <PublicationType UI="D016454">Review</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName UI="D048493">Proto-Oncogene Proteins B-raf</DescriptorName></MeshHeading>
      </MeshHeadingList>
      <CommentsCorrectionsList>
        <CommentsCorrections RefType="RetractionIn">
          <RefSource>J Test. 2021;1:1</RefSource><PMID>999</PMID>
        </CommentsCorrections>
      </CommentsCorrectionsList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1000/test</ArticleId>
        <ArticleId IdType="pmc">PMC7654321</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parser_captures_the_fields_layering_depends_on():
    parsed = parse_pubmed_articles(PUBMED_XML)
    assert len(parsed) == 1
    article = parsed[0]
    assert article.pmid == "12345678"
    assert article.doi == "10.1000/test" and article.pmcid == "PMC7654321"
    assert article.publication_types == ["Journal Article", "Review"]
    assert article.publication_type_uis == ["D016428", "D016454"]
    assert article.mesh_terms == ["Proto-Oncogene Proteins B-raf"]
    assert article.authors == ["Silva, Ana"] and article.affiliations == ["Universidade X"]
    assert article.year == 2019 and article.pub_date == "2019-May"
    assert article.date_revised == "2024-03-02"
    assert [item.ref_type for item in article.affecting_notices] == ["RetractionIn"]
    assert article.abstract.startswith("RESULTS:")


def test_parser_is_pure_and_reproducible():
    assert parse_pubmed_articles(PUBMED_XML) == parse_pubmed_articles(PUBMED_XML)


def test_primary_article_reaches_the_evidence_layer():
    assert layer_of(record()) == "primary_evidence"


@pytest.mark.parametrize("publication_type", ["Review", "Meta-Analysis", "Practice Guideline"])
def test_reviews_and_guidelines_never_enter_the_evidence_layer(publication_type):
    assert layer_of(record(publication_types=["Journal Article", publication_type])) \
        == "context_reference"


def test_preprints_are_excluded_from_the_braf_corpus():
    assignment = assign_layer(record(publication_types=["Preprint"]), BRAF_ALIASES, BRAF_MESH)
    assert (assignment.layer, assignment.rule_id) == ("excluded", "preprint")


@pytest.mark.parametrize("publication_type", ["Editorial", "Comment", "News", "Published Erratum"])
def test_notices_are_excluded(publication_type):
    assignment = assign_layer(record(publication_types=[publication_type]),
                              BRAF_ALIASES, BRAF_MESH)
    assert assignment.layer == "excluded"
    assert assignment.exclusion_reason.startswith("publication_type:")


def test_retracted_publication_is_excluded_by_type():
    assignment = assign_layer(record(publication_types=["Journal Article", "Retracted Publication"]),
                              BRAF_ALIASES, BRAF_MESH)
    assert (assignment.layer, assignment.exclusion_reason) == ("excluded", "retracted_publication")


def test_retraction_notice_outranks_an_attractive_publication_type():
    retracted = record(comments_corrections=[{"ref_type": "RetractionIn", "pmid": "999"}])
    assignment = assign_layer(retracted, BRAF_ALIASES, BRAF_MESH)
    assert (assignment.layer, assignment.exclusion_reason) == ("excluded", "retracted_publication")


def test_unresolved_expression_of_concern_is_excluded():
    flagged = record(comments_corrections=[{"ref_type": "ExpressionOfConcernIn", "pmid": "998"}])
    assignment = assign_layer(flagged, BRAF_ALIASES, BRAF_MESH)
    assert assignment.exclusion_reason == "expression_of_concern_unresolved"


def test_record_without_a_gene_link_is_excluded():
    unrelated = record(title="GNA14 in renal cell carcinoma",
                       abstract="Proximity labeling proteomics of lipid droplets.")
    assignment = assign_layer(unrelated, BRAF_ALIASES, BRAF_MESH)
    assert (assignment.layer, assignment.gene_link) == ("excluded", "absent")


def test_missing_text_is_not_evidence_of_a_missing_gene_link():
    empty = record(title="", abstract="")
    assignment = assign_layer(empty, BRAF_ALIASES, BRAF_MESH)
    assert assignment.gene_link == "unverifiable_no_text"
    assert assignment.exclusion_reason == "missing_pmid_or_doi_or_title"


def test_missing_provenance_is_excluded_before_scientific_classification():
    assignment = assign_layer(record(pmid="", doi=""), BRAF_ALIASES, BRAF_MESH)
    assert (assignment.layer, assignment.rule_id) == ("excluded", "missing_provenance")


def test_obvious_scraping_junk_has_a_dedicated_exclusion_reason():
    assignment = assign_layer(
        record(pmid="", title="Search results", abstract="", doi=""),
        BRAF_ALIASES, BRAF_MESH,
    )
    assert (assignment.layer, assignment.rule_id) == ("excluded", "scraping_junk")


def test_mesh_only_link_is_accepted_and_reported():
    mesh_linked = record(title="A kinase study", abstract="No lexical gene symbol here.",
                         mesh_terms=["Proto-Oncogene Proteins B-raf"])
    assignment = assign_layer(mesh_linked, BRAF_ALIASES, BRAF_MESH)
    assert (assignment.layer, assignment.gene_link) == ("primary_evidence", "mesh")


def test_alias_matching_is_boundary_safe():
    assert gene_link_status(record(title="BRAFLIKE protein", abstract="unrelated"),
                            BRAF_ALIASES, BRAF_MESH) == "absent"
    assert gene_link_status(record(title="The B-RAF kinase", abstract=""),
                            BRAF_ALIASES, BRAF_MESH) == "title"


def test_a_drug_name_containing_the_symbol_is_not_a_gene_link():
    for drug in ("dabrafenib", "encorafenib", "vemurafenib/dabrafenib"):
        assert gene_link_status(record(title=f"Response to {drug}", abstract="Melanoma cohort."),
                                BRAF_ALIASES, BRAF_MESH) == "absent"


@pytest.mark.parametrize("mention", ["BRAFV600E", "(BRAFV600E)", "BRAFV600E-mutant",
                                     "BRAFV600", "B-RAFV600E", "BrafV600MUT",
                                     "BRAFV600WT"])
def test_symbol_fused_to_its_variant_still_counts_as_a_gene_link(mention):
    """BRAFV600E is written without a separator throughout the corpus."""
    assert gene_link_status(record(title=f"Melanoma driven by {mention}", abstract=""),
                            BRAF_ALIASES, BRAF_MESH) == "title"


@pytest.mark.parametrize("mention", ["antiBRAF", "panBRAF", "nonBRAF", "mutBRAF"])
def test_a_fused_modifier_prefix_still_counts_as_a_gene_link(mention):
    assert gene_link_status(record(title=f"{mention} therapy in melanoma", abstract=""),
                            BRAF_ALIASES, BRAF_MESH) == "title"


def test_the_modifier_prefixes_do_not_reopen_drug_names():
    """dabrafenib ends in -brafenib and must never satisfy the leading boundary."""
    for drug in ("dabrafenib", "vemurafenib", "encorafenib", "sorafenib"):
        assert gene_link_status(record(title=f"Treated with {drug}", abstract="Cohort study."),
                                BRAF_ALIASES, BRAF_MESH) == "absent"


def test_duplicate_doi_keeps_one_record_and_marks_the_other():
    frame = build_corpus_layers(
        [record("10", doi="10.1/x"), record("11", doi="10.1/X")],
        BRAF_ALIASES, BRAF_MESH,
    )
    kept = frame.set_index("pmid")
    assert kept.loc["10", "layer"] == "primary_evidence"
    assert kept.loc["11", "layer"] == "excluded"
    assert kept.loc["11", "exclusion_reason"] == "duplicate_of_kept_record"
    assert kept.loc["11", "duplicate_of"] == "10"


def test_every_record_lands_in_exactly_one_layer():
    records = [record("10"), record("11", publication_types=["Review"]),
               record("12", publication_types=["Preprint"]),
               record("13", publication_types=["Editorial"])]
    frame = build_corpus_layers(records, BRAF_ALIASES, BRAF_MESH)
    assert len(frame) == len(records) == frame["pmid"].nunique()
    assert set(frame["layer"]) <= {"primary_evidence", "context_reference", "excluded"}


def test_manual_override_is_recorded_and_never_silent():
    frame = build_corpus_layers(
        [record("10", comments_corrections=[{"ref_type": "ExpressionOfConcernIn"}])],
        BRAF_ALIASES, BRAF_MESH,
        manual_overrides={"10": {"layer": "primary_evidence", "reason": "",
                                 "note": "concern resolved in 2025 correction"}},
    )
    row = frame.iloc[0]
    assert row["layer"] == "primary_evidence"
    assert row["manual_review_status"] == "overridden"
    assert row["rule_id"] == "manual_override"
    assert "2025 correction" in row["manual_review_note"]


def test_full_text_availability_does_not_change_the_layer():
    with_text = assign_layer(record(), BRAF_ALIASES, BRAF_MESH, full_text_available=True)
    without_text = assign_layer(record(), BRAF_ALIASES, BRAF_MESH, full_text_available=False)
    assert with_text.layer == without_text.layer == "primary_evidence"
    assert with_text.full_text_available and not without_text.full_text_available


def test_mention_count_is_metadata_and_does_not_gate_the_layer():
    """One peripheral mention is still a verifiable link; centrality is per claim."""
    peripheral = record(title="Antisense targeting of Raf-1",
                        abstract="Raf-1 signalling was compared with B-Raf activity.")
    frame = build_corpus_layers([peripheral], BRAF_ALIASES, BRAF_MESH)
    assert frame.iloc[0]["layer"] == "primary_evidence"
    assert frame.iloc[0]["gene_mentions"] == 1


def test_journal_metadata_cannot_influence_the_layer():
    prestigious = record(journal="Nature")
    obscure = record(journal="Journal of Small Results")
    assert layer_of(prestigious) == layer_of(obscure) == "primary_evidence"
