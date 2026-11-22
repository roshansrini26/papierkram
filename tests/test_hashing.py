from papierkram.ingestion.hashing import hash_file, hash_text


def test_same_text_same_hash():
    assert hash_text("Anmeldung") == hash_text("Anmeldung")


def test_different_text_different_hash():
    assert hash_text("Anmeldung") != hash_text("Abmeldung")


def test_file_hash_matches_text_hash(tmp_path):
    # tmp_path is a pytest fixture: a fresh temporary folder for each test.
    path = tmp_path / "letter.txt"
    path.write_text("Sehr geehrter Herr Srinivasan", encoding="utf-8")
    assert hash_file(path) == hash_text("Sehr geehrter Herr Srinivasan")
