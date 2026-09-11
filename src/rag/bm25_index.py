"""Persistent lexical BM25 index preserving biomedical symbols and variants."""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[A-Za-z]+\d+[A-Za-z*]+|[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def bm25_corpus_path(rag_dir: Path, layer: str | None = None) -> Path:
    """One lexical corpus per layer, matching the per-layer Qdrant collections."""
    name = "bm25_corpus.jsonl" if layer is None else f"bm25_corpus_{layer}.jsonl"
    return Path(rag_dir) / name


class PersistentBM25:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._index = BM25Okapi([tokenize(document.page_content) for document in documents])

    def search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        scores = self._index.get_scores(tokenize(query))
        indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)[:k]
        return [(self.documents[index], float(scores[index])) for index in indices if scores[index] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for document in self.documents:
                handle.write(json.dumps({"page_content": document.page_content,
                                         "metadata": document.metadata}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "PersistentBM25":
        if not path.exists():
            raise FileNotFoundError(f"BM25 corpus absent: {path}")
        documents = []
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        return cls(documents)
