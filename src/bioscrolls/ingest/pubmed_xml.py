"""Parse PubMed efetch XML into immutable ``Document`` records."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # noqa: S405 - only used for types/ParseError

from defusedxml import ElementTree as SafeET

from bioscrolls.models import Document, ValidationError

_YEAR_RE = re.compile(r"(1[89]\d\d|20\d\d)")
_UNDETERMINED_LANGUAGE = "und"


class PubmedParseError(ValueError):
    """Raised when efetch output is not parseable XML."""


def parse_year(raw: str | None) -> int | None:
    """Extract the first plausible 4-digit year from a PubMed date string."""
    if not raw:
        return None
    match = _YEAR_RE.search(raw)
    return int(match.group(1)) if match else None


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _unwrap_translated_title(title: str) -> str:
    """PubMed wraps English translations of non-English titles in brackets: ``[Title].``"""
    stripped = title.strip()
    if stripped.startswith("[") and (stripped.endswith("].") or stripped.endswith("]")):
        inner = stripped[1:].rstrip(".").rstrip("]")
        return f"{inner}."
    return stripped


def _abstract_text(abstract: ET.Element | None) -> str:
    if abstract is None:
        return ""
    parts = []
    for node in abstract.findall("AbstractText"):
        body = _text(node)
        if not body:
            continue
        label = node.get("Label")
        parts.append(f"{label}: {body}" if label else body)
    return " ".join(parts)


def _publication_year(article: ET.Element) -> int | None:
    """Earliest of the journal issue date and the electronic article date.

    PubMed's ``[dp]`` filter matches either date, and e-publication usually
    precedes the print issue, so the earliest year best reflects first publication.
    """
    candidates = []
    pub_date = article.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        candidates.append(parse_year(_text(pub_date.find("Year"))) or parse_year(_text(pub_date.find("MedlineDate"))))
    candidates.extend(parse_year(_text(node)) for node in article.findall("ArticleDate/Year"))
    years = [y for y in candidates if y]
    return min(years) if years else None


def _other_abstracts(citation: ET.Element) -> tuple[tuple[str, str], ...]:
    found = []
    for node in citation.findall("OtherAbstract"):
        body = _abstract_text(node)
        if body:
            found.append((node.get("Language", _UNDETERMINED_LANGUAGE), body))
    return tuple(found)


def _parse_article(node: ET.Element) -> Document | None:
    citation = node.find("MedlineCitation")
    if citation is None:
        return None
    pmid = _text(citation.find("PMID"))
    article = citation.find("Article")
    if not pmid or article is None:
        return None
    languages = tuple(_text(n) for n in article.findall("Language") if _text(n))
    try:
        return Document(
            pmid=pmid,
            title=_unwrap_translated_title(_text(article.find("ArticleTitle"))),
            abstract=_abstract_text(article.find("Abstract")),
            year=_publication_year(article),
            languages=languages or (_UNDETERMINED_LANGUAGE,),
            journal=_text(article.find("Journal/Title")),
            vernacular_title=_text(article.find("VernacularTitle")),
            other_abstracts=_other_abstracts(citation),
        )
    except ValidationError:
        return None


def parse_efetch_xml(xml_text: str) -> list[Document]:
    """Parse a ``PubmedArticleSet`` document. Records without a PMID are skipped."""
    try:
        root = SafeET.fromstring(xml_text)
    except (ET.ParseError, SafeET.ParseError) as exc:
        raise PubmedParseError(f"invalid efetch XML: {exc}") from exc
    parsed = (_parse_article(node) for node in root.iter("PubmedArticle"))
    return [doc for doc in parsed if doc is not None]
