import io
import zipfile

import pytest

from bioscrolls.corpora.biored import download_biored, load_split
from bioscrolls.normalize.lexicon import download_lexicons, lexicon_sources


class Resp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


class Session:
    def __init__(self, response):
        self.response = response
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self.response


def biored_zip(fixtures_dir):
    buffer = io.BytesIO()
    sample = (fixtures_dir / "biored_sample.PubTator").read_text()
    with zipfile.ZipFile(buffer, "w") as archive:
        for split in ("Train", "Dev", "Test"):
            archive.writestr(f"BioRED/{split}.PubTator", sample)
        archive.writestr("BioRED/Train.BioC.JSON", "{}")
    return buffer.getvalue()


def test_download_biored_extracts_and_is_idempotent(tmp_path, fixtures_dir):
    session = Session(Resp(content=biored_zip(fixtures_dir)))
    target = download_biored(tmp_path, session=session)
    assert (target / "Test.PubTator").exists()
    assert not (target / "Train.BioC.JSON").exists()
    download_biored(tmp_path, session=session)
    assert len(session.urls) == 1
    assert len(load_split("Dev", tmp_path)) == 2


def test_download_biored_http_error(tmp_path):
    with pytest.raises(RuntimeError, match="HTTP 503"):
        download_biored(tmp_path, session=Session(Resp(503)))


def test_load_split_validation(tmp_path):
    with pytest.raises(ValueError):
        load_split("Validation", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_split("Test", tmp_path)


def test_download_lexicons(tmp_path):
    session = Session(Resp(content=b"data"))
    paths = download_lexicons(tmp_path, session=session)
    assert len(paths) == 3 and all(p.read_bytes() == b"data" for p in paths)
    download_lexicons(tmp_path, session=session)
    assert len(session.urls) == 3
    assert set(lexicon_sources(tmp_path)) == {"Gene", "Disease", "Chemical"}


def test_download_lexicons_error_and_missing(tmp_path):
    with pytest.raises(RuntimeError):
        download_lexicons(tmp_path, session=Session(Resp(404)))
    with pytest.raises(FileNotFoundError):
        lexicon_sources(tmp_path / "empty")
