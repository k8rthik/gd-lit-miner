import gzip

import pytest

from bioscrolls.normalize.lexicon import (
    LexiconEntry,
    LexiconFormatError,
    build_index,
    iter_ctd,
    iter_hgnc,
)

HGNC = (
    "hgnc_id\tsymbol\tname\talias_symbol\talias_name\tprev_symbol\tprev_name\tentrez_id\n"
    'HGNC:613\tAPOE\tapolipoprotein E\t\t\tAD2\t"Alzheimer disease 2"\t348\n'
    'HGNC:11138\tSNCA\tsynuclein alpha\t"NACP|PD1"\t"alpha-synuclein|α-synuclein"\t"PARK1|PARK4"\t\t6622\n'
    "HGNC:99\tNOENT\tno entrez\t\t\t\t\t\n"
)

CTD = (
    "# header\n# Fields:\n# DiseaseName\tDiseaseID\tAltDiseaseIDs\tSynonyms\n#\n"
    "Alzheimer Disease\tMESH:D000544\t\tAlzheimer's Disease|Alzheimer Dementia\n"
    "Parkinson Disease\tMESH:D010300\tDO:1\tParkinson's Disease|Idiopathic Parkinson Disease\n"
)


@pytest.fixture
def hgnc_path(tmp_path):
    path = tmp_path / "hgnc.txt"
    path.write_text(HGNC, encoding="utf-8")
    return path


@pytest.fixture
def ctd_path(tmp_path):
    path = tmp_path / "ctd.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(CTD)
    return path


def test_iter_hgnc_entries(hgnc_path):
    entries = list(iter_hgnc(hgnc_path))
    apoe = [e for e in entries if e.entity_id == "NCBIGene:348"]
    assert {e.surface for e in apoe} == {"APOE", "apolipoprotein E", "AD2", "Alzheimer disease 2"}
    assert all(e.name == "APOE" for e in apoe)
    symbol = next(e for e in apoe if e.surface == "APOE")
    alias = next(e for e in apoe if e.surface == "AD2")
    assert symbol.priority < alias.priority
    assert any(e.entity_id == "HGNC:99" for e in entries)
    assert any(e.surface == "α-synuclein" for e in entries)


def test_iter_ctd_reads_field_header(ctd_path):
    entries = list(iter_ctd(ctd_path, id_field="DiseaseID", name_field="DiseaseName"))
    assert LexiconEntry("Alzheimer Disease", "MESH:D000544", "Alzheimer Disease", 0) in entries
    assert any(e.surface == "Idiopathic Parkinson Disease" and e.priority == 1 for e in entries)


def test_iter_ctd_reads_split_synonym_columns(tmp_path):
    """The CTD chemical report uses MESHSynonyms + CTDCuratedSynonyms, not Synonyms."""
    raw = (
        "# Fields:\n# ChemicalName\tChemicalID\tMESHSynonyms\tCTDCuratedSynonyms\n#\n"
        "Levodopa\tMESH:D007980\tL-DOPA|L-Dopa\tlevodopa anhydrous\n"
    )
    path = tmp_path / "chem.tsv"
    path.write_text(raw, encoding="utf-8")
    entries = list(iter_ctd(path, id_field="ChemicalID", name_field="ChemicalName"))
    assert {e.surface for e in entries} == {"Levodopa", "L-DOPA", "L-Dopa", "levodopa anhydrous"}
    assert all(e.entity_id == "MESH:D007980" and e.name == "Levodopa" for e in entries)


def test_iter_ctd_without_any_synonym_column_raises(tmp_path):
    path = tmp_path / "chem.tsv"
    path.write_text("# Fields:\n# ChemicalName\tChemicalID\n#\nX\tMESH:D1\n", encoding="utf-8")
    with pytest.raises(LexiconFormatError, match="synonym"):
        list(iter_ctd(path, id_field="ChemicalID", name_field="ChemicalName"))


def test_iter_ctd_missing_header_raises(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("a\tb\n")
    with pytest.raises(LexiconFormatError):
        list(iter_ctd(path, id_field="DiseaseID", name_field="DiseaseName"))


def test_build_index_filters_to_wanted_and_prefers_priority(hgnc_path):
    index = build_index(iter_hgnc(hgnc_path), wanted={"apoe", "alpha synuclein", "pd1", "zzz"})
    assert index["apoe"] == ("NCBIGene:348", "APOE")
    assert index["alpha synuclein"] == ("NCBIGene:6622", "SNCA")
    assert index["pd1"] == ("NCBIGene:6622", "SNCA")
    assert "zzz" not in index


def test_build_index_without_filter(ctd_path):
    index = build_index(iter_ctd(ctd_path, id_field="DiseaseID", name_field="DiseaseName"))
    assert index["alzheimer disease"] == ("MESH:D000544", "Alzheimer Disease")
