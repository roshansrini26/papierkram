"""Split documents into chunks: first by markdown headers, then by sentences.

Ported from chunk_by_sentence / chunk_text in the old ingest.py, with one fix
that matters for German: abbreviations such as "z. B.", "Nr." and "gem." end
in a full stop and were being treated as sentence ends.
"""

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter

# Common abbreviations in German official letters. Stored lowercase, without the final dot.
GERMAN_ABBREVIATIONS = {
    "z. b",
    "z.b",
    "u. a",
    "u.a",
    "d. h",
    "d.h",
    "bzw",
    "ca",
    "ggf",
    "inkl",
    "evtl",
    "nr",
    "str",
    "abs",
    "gem",
    "vgl",
    "usw",
    "etc",
    "sog",
    "dr",
    "prof",
    "hr",
    "fr",
    # Legal references and letter vocabulary
    "art",
    "ziff",
    "lit",
    "rn",
    "tz",
    "anl",
    "az",
    "tel",
    "kto",
    "mwst",
    "ust",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _ends_with_abbreviation(piece: str) -> bool:
    """True if the text ends in a known abbreviation, e.g. '... siehe z. B.'"""
    lowered = piece.lower().rstrip(".")
    if not lowered:
        return False
    # A single letter before the dot ("z." in "z. B.", "u." in "u. a.") is almost never a sentence end.
    last_word = lowered.split()[-1]
    if len(last_word) == 1 and last_word.isalpha():
        return True
    # Street names: "Musterstr." is one word, so it never matches " str".
    if last_word.endswith("str"):
        return True
    return any(lowered == abbr or lowered.endswith(" " + abbr) for abbr in GERMAN_ABBREVIATIONS)


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace, then glue pieces back together when the split was wrong:
    the previous piece ended in an abbreviation, or this piece starts lowercase (a German sentence
    never does, so "bis zum 31.12. den Betrag" is one sentence)."""
    pieces = [p for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        starts_lowercase = piece[0].islower()
        if sentences and (starts_lowercase or _ends_with_abbreviation(sentences[-1])):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def chunk_by_sentence(text: str, max_chars: int = 800, overlap_sentences: int = 2) -> list[str]:
    """Pack whole sentences into chunks of up to max_chars, repeating the last
    overlap_sentences of each chunk at the start of the next one."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        # Measure the chunk exactly as it will be stored, spaces included.
        if current and len(" ".join([*current, sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_markdown(text: str, max_chars: int = 800, overlap_sentences: int = 2) -> list[str]:
    """Split on headers first so no chunk spans two sections, then by sentence.
    Each chunk starts with its section path, e.g. 'Beitragspflicht > Befreiung'."""
    sections = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS).split_text(text)
    if not sections:
        return chunk_by_sentence(text, max_chars, overlap_sentences)

    chunks: list[str] = []
    for section in sections:
        content = section.page_content.strip()
        if not content:
            continue
        prefix = " > ".join(section.metadata.values())
        for piece in chunk_by_sentence(content, max_chars, overlap_sentences):
            chunks.append(f"{prefix}\n{piece}" if prefix else piece)
    return chunks
