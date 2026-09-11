"""Strict protein-variant mention matching shared by acquisition and RAG."""

from __future__ import annotations

import re


# A trailing compound suffix such as S3I-201 is deliberately excluded.  This
# is stricter than a word boundary because hyphens are common in drug names.
ONE_LETTER_VARIANT = re.compile(
    r"(?<![A-Za-z0-9])(?:p\.)?([ACDEFGHIKLMNPQRSTVWY])(\d+)"
    r"([ACDEFGHIKLMNPQRSTVWY])(?![A-Za-z0-9]|[-–—]\d)",
    re.I,
)


def exact_variant_pattern(variant: str) -> re.Pattern[str]:
    """Return a boundary-safe regex for an already normalized variant."""
    normalized = str(variant).upper().removeprefix("P.")
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]", normalized):
        raise ValueError(f"Invalid one-letter protein variant: {variant}")
    return re.compile(
        rf"(?<![A-Za-z0-9])(?:p\.)?{re.escape(normalized)}"
        rf"(?![A-Za-z0-9]|[-–—]\d)",
        re.I,
    )


def contains_exact_variant(text: str, variant: str) -> bool:
    return bool(exact_variant_pattern(variant).search(str(text)))


def mentioned_one_letter_variants(text: str) -> set[str]:
    return {
        f"{match.group(1).upper()}{match.group(2)}{match.group(3).upper()}"
        for match in ONE_LETTER_VARIANT.finditer(str(text))
    }


def local_variant_windows(text: str, variant: str, radius: int = 1) -> list[str]:
    """Return sentence-local windows containing an exact variant mention."""
    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(text))
        if part.strip()
    ]
    pattern = exact_variant_pattern(variant)
    windows = []
    for index, sentence in enumerate(sentences):
        if pattern.search(sentence):
            windows.append(" ".join(sentences[max(0, index - radius):index + radius + 1]))
    return windows


def same_residue_variant_mentioned(text: str, variant: str) -> bool:
    target = re.search(r"\d+", str(variant))
    if not target:
        return False
    position = int(target.group())
    normalized = str(variant).upper().removeprefix("P.")
    return any(
        mention != normalized and int(re.search(r"\d+", mention).group()) == position
        for mention in mentioned_one_letter_variants(text)
    )
