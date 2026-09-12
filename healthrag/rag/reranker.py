"""MedCPT second-stage cross-encoder reranking."""

from __future__ import annotations

import os

from langchain_core.documents import Document

CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"


class MedCPTReranker:
    def __init__(self, device: str | None = None, batch_size: int = 8):
        self.device = device or os.getenv("MEDCPT_DEVICE", "cpu")
        self.batch_size = max(1, int(batch_size))
        self._model = self._tokenizer = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                CROSS_ENCODER_MODEL, local_files_only=True
            ).to(self.device).eval()

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[tuple[Document, float]]:
        import torch
        if not documents:
            return []
        self._load()
        logits = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            pairs = [[query, doc.page_content] for doc in batch]
            encoded = self._tokenizer(pairs, truncation=True, padding=True, max_length=512,
                                      return_tensors="pt").to(self.device)
            with torch.inference_mode():
                values = self._model(**encoded).logits.squeeze(-1).cpu().tolist()
            logits.extend(values if isinstance(values, list) else [float(values)])
        return sorted(zip(documents, map(float, logits)), key=lambda item: item[1], reverse=True)[:top_k]
