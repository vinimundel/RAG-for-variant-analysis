"""Literature retrieval and evidence-bounded mechanistic inference."""


def get_mechanism_analyzer(gene: str, **kwargs):
    from src.rag.chain import LiteratureMechanismAnalyzer
    return LiteratureMechanismAnalyzer(gene=gene, **kwargs)


__all__ = ["get_mechanism_analyzer"]
