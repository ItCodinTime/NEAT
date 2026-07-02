import hashlib

import pytest
from scripts.fetch_dataset import fetch


def test_fetch_verifies_checksum_before_install(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"neat dataset fixture")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "installed.bin"
    fetch(source.as_uri(), destination, expected)
    assert destination.read_bytes() == source.read_bytes()


def test_fetch_removes_invalid_download(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"invalid")
    destination = tmp_path / "installed.bin"
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch(source.as_uri(), destination, "0" * 64)
    assert not destination.exists()
