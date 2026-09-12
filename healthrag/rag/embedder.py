"""Asymmetric MedCPT embeddings for biomedical information retrieval."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DOCUMENT_MODEL = "ncbi/MedCPT-Article-Encoder"
EMBEDDING_DIM = 768


def article_pair(text: str) -> list[str]:
    """Split the canonical title-first chunk for MedCPT's two-segment input."""
    title, separator, body = text.partition("\n")
    return [title, body] if separator else ["", text]


class MedCPTEmbeddings(Embeddings):
    """Use the query and article encoders in the shared MedCPT vector space."""

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device, self.batch_size = device, batch_size
        self._models = {}

    def _load(self, kind: str):
        if kind not in self._models:
            from transformers import AutoModel, AutoTokenizer
            name = QUERY_MODEL if kind == "query" else DOCUMENT_MODEL
            # Models are pinned/downloaded during Pixi setup; inference must not
            # make surprise network calls or stall on Hugging Face HEAD retries.
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            model = AutoModel.from_pretrained(name, local_files_only=True).to(self.device).eval()
            self._models[kind] = (tokenizer, model)
        return self._models[kind]

    def _encode(self, texts: list[str], kind: str) -> list[list[float]]:
        import torch
        tokenizer, model = self._load(kind)
        result = []
        max_length = 64 if kind == "query" else 512
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if kind == "query":
                inputs = batch
            else:
                inputs = [article_pair(text) for text in batch]
            encoded = tokenizer(inputs, truncation=True, padding=True, max_length=max_length,
                                return_tensors="pt").to(self.device)
            with torch.inference_mode():
                vectors = model(**encoded).last_hidden_state[:, 0, :]
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            result.extend(vectors.cpu().tolist())
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query")[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Batch query encoding without confusing queries with article documents."""
        return self._encode(texts, "query")


def get_biomedical_embedder(device: str | None = None) -> MedCPTEmbeddings:
    # Keep document ingestion bounded while avoiding thousands of tiny CPU
    # batches when the curated corpus is rebuilt.
    return MedCPTEmbeddings(device=device or os.getenv("MEDCPT_DEVICE", "cpu"),
                            batch_size=int(os.getenv("MEDCPT_BATCH_SIZE", "4")))
