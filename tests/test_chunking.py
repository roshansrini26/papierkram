from papierkram.ingestion.chunking import chunk_by_sentence, chunk_markdown, split_sentences


def test_splits_plain_sentences():
    text = "Erster Satz. Zweiter Satz! Dritter Satz?"
    assert split_sentences(text) == ["Erster Satz.", "Zweiter Satz!", "Dritter Satz?"]


def test_german_abbreviations_do_not_end_a_sentence():
    text = "Bitte senden Sie z. B. Ihre Immatrikulationsbescheinigung gem. Abs. 2. Danke."
    assert split_sentences(text) == [
        "Bitte senden Sie z. B. Ihre Immatrikulationsbescheinigung gem. Abs. 2.",
        "Danke.",
    ]


def test_chunks_respect_max_chars_and_overlap():
    sentences = [f"Satz Nummer {i} hat etwas Text." for i in range(20)]
    chunks = chunk_by_sentence(" ".join(sentences), max_chars=120, overlap_sentences=1)

    assert len(chunks) > 1
    for chunk in chunks:
        # One sentence of slack, because a chunk is only closed when the NEXT sentence would not fit.
        assert len(chunk) <= 120 + len(sentences[0]) + 5
    # With overlap 1, the last sentence of a chunk starts the next chunk.
    last_sentence_of_first = chunks[0].split(". ")[-1]
    assert chunks[1].startswith(last_sentence_of_first.rstrip("."))


def test_no_overlap_means_no_repeated_sentences():
    text = " ".join(f"Satz {i}." for i in range(30))
    chunks = chunk_by_sentence(text, max_chars=50, overlap_sentences=0)
    joined = " ".join(chunks)
    assert joined.count("Satz 5.") == 1


def test_markdown_chunks_carry_their_section_path():
    text = "# Rundfunkbeitrag\n## Befreiung\nStudierende mit BAföG können eine Befreiung beantragen."
    chunks = chunk_markdown(text)
    assert chunks == ["Rundfunkbeitrag > Befreiung\nStudierende mit BAföG können eine Befreiung beantragen."]


def test_text_without_headers_still_chunks():
    assert chunk_markdown("Nur ein Satz ohne Überschrift.") == ["Nur ein Satz ohne Überschrift."]


def test_lowercase_continuation_is_not_a_new_sentence():
    text = "Bitte zahlen Sie bis zum 31.12. den Betrag. Danke."
    assert split_sentences(text) == ["Bitte zahlen Sie bis zum 31.12. den Betrag.", "Danke."]


def test_legal_references_and_street_names_do_not_split():
    text = "Gemäß Art. 5 und Ziff. 3 gilt das. Wir sitzen in der Musterstr. 5 in Hamburg. Danke."
    assert split_sentences(text) == [
        "Gemäß Art. 5 und Ziff. 3 gilt das.",
        "Wir sitzen in der Musterstr. 5 in Hamburg.",
        "Danke.",
    ]


def test_chunks_never_exceed_max_chars_when_sentences_fit():
    text = " ".join(f"Satz {i} hat genau etwas Text." for i in range(40))
    for chunk in chunk_by_sentence(text, max_chars=100, overlap_sentences=1):
        assert len(chunk) <= 100
