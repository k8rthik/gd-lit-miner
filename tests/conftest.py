from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def efetch_xml() -> str:
    return (FIXTURES / "efetch_sample.xml").read_text(encoding="utf-8")
