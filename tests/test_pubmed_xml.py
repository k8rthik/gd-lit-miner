import pytest

from bioscrolls.ingest.pubmed_xml import PubmedParseError, parse_efetch_xml, parse_year


def test_parses_all_articles(efetch_xml):
    docs = parse_efetch_xml(efetch_xml)
    assert [d.pmid for d in docs] == ["31210462", "30869416", "31367748"]


def test_records_language_and_vernacular_title(efetch_xml):
    docs = {d.pmid: d for d in parse_efetch_xml(efetch_xml)}
    dutch = docs["31210462"]
    assert dutch.languages == ("dut",)
    assert not dutch.is_english
    assert dutch.vernacular_title.startswith("Zorgdeclaraties")
    # Bracketed English translation of the title is unwrapped.
    assert dutch.title == "Healthcare claims, a valuable source of information."


def test_keeps_other_language_abstracts(efetch_xml):
    docs = {d.pmid: d for d in parse_efetch_xml(efetch_xml)}
    french = docs["30869416"]
    assert french.languages == ("fre",)
    assert french.abstract.startswith("Therapeutic and pharmacologic")
    assert len(french.other_abstracts) == 1
    lang, text = french.other_abstracts[0]
    assert lang == "fre"
    assert "maladie de parkinson" in text


def test_inline_markup_is_flattened(efetch_xml):
    docs = {d.pmid: d for d in parse_efetch_xml(efetch_xml)}
    assert "Perera et al.." in docs["31367748"].abstract
    assert docs["31367748"].year == 2018
    assert docs["31367748"].journal.startswith("Brain")


def test_structured_abstract_sections_are_labelled():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>
    <Article><Journal><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue>
    <Title>J</Title></Journal><ArticleTitle>T</ArticleTitle>
    <Abstract><AbstractText Label="BACKGROUND">Bg.</AbstractText><AbstractText Label="RESULTS">Res.</AbstractText></Abstract>
    <Language>eng</Language></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    (doc,) = parse_efetch_xml(xml)
    assert doc.abstract == "BACKGROUND: Bg. RESULTS: Res."
    assert doc.year == 2019


def test_falls_back_to_article_date_and_skips_records_without_pmid():
    xml = """<PubmedArticleSet>
    <PubmedArticle><MedlineCitation><Article><ArticleTitle>No pmid</ArticleTitle></Article></MedlineCitation></PubmedArticle>
    <PubmedArticle><MedlineCitation><PMID>7</PMID><Article><Journal><JournalIssue><PubDate/></JournalIssue></Journal>
    <ArticleTitle>T</ArticleTitle><ArticleDate><Year>2021</Year></ArticleDate></Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>"""
    docs = parse_efetch_xml(xml)
    assert [d.pmid for d in docs] == ["7"]
    assert docs[0].year == 2021
    assert docs[0].languages == ("und",)
    assert docs[0].abstract == ""


def test_year_is_earliest_of_issue_and_electronic_date():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>3</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>T</ArticleTitle><ArticleDate DateType="Electronic"><Year>2024</Year></ArticleDate>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    assert parse_efetch_xml(xml)[0].year == 2024


def test_malformed_xml_raises():
    with pytest.raises(PubmedParseError):
        parse_efetch_xml("<PubmedArticleSet><oops>")


@pytest.mark.parametrize(
    "raw,expected",
    [("2019", 2019), ("2019 Jan-Feb", 2019), ("Winter 1998-1999", 1998), ("", None), (None, None), ("n.d.", None)],
)
def test_parse_year(raw, expected):
    assert parse_year(raw) == expected
