import hashlib
from pathlib import Path


def hash_text(test: str) -> str:
    """Hash a string using SHA256 and return the hexadecimal digest."""
    return hashlib.sha256(test.encode("utf-8")).hexdigest()


def hash_file(file_path: Path) -> str:
    """hash files in 1 MB chunks to avoid memory issues with large files"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
