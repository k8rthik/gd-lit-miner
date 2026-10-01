import pytest

from bioscrolls.ingest.eutils import SearchResult
from bioscrolls.ingest.pipeline import QuerySpec, plan_queries, run_ingest
from bioscrolls.models import Document


def test_plan_queries_per_year_and_non_english():
    specs = plan_queries({"pd": '"Parkinson Disease"[MeSH Terms]'}, 2018, 2019, per_year=5, non_english=7)
    assert len(specs) == 4
    yearly = [s for s in specs if s.kind == "yearly"]
    assert [s.year for s in yearly] == [2018, 2019]
    assert yearly[0].term == '"Parkinson Disease"[MeSH Terms] AND hasabstract AND 2018[dp]'
    assert yearly[0].retmax == 5
    ne = [s for s in specs if s.kind == "non_english"]
    assert [s.year for s in ne] == [2018, 2019]
    assert ne[0].term.endswith("AND 2018[dp] NOT english[Language]")
    assert ne[0].retmax == 7


def test_plan_queries_skips_non_english_when_zero():
    specs = plan_queries({"pd": "x"}, 2018, 2018, per_year=1, non_english=0)
    assert [s.kind for s in specs] == ["yearly"]


@pytest.mark.parametrize(
    "kwargs",
    [dict(start=2020, end=2019), dict(per_year=0), dict(non_english=-1), dict(queries={})],
)
def test_plan_queries_validates(kwargs):
    args = dict(queries={"pd": "x"}, start=2018, end=2019, per_year=1, non_english=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        plan_queries(args["queries"], args["start"], args["end"], args["per_year"], args["non_english"])


class StubClient:
    def __init__(self, search_map, docs):
        self.search_map = search_map
        self.docs = {d.pmid: d for d in docs}
        self.fetched = []

    def esearch(self, term, retmax, retstart=0):
        return self.search_map[term]

    def efetch(self, pmids):
        self.fetched.append(list(pmids))
        return [self.docs[p] for p in pmids if p in self.docs]


def test_run_ingest_dedupes_and_labels():
    specs = [
        QuerySpec(label="pd", term="t1", retmax=2, year=2018, kind="yearly"),
        QuerySpec(label="ad", term="t2", retmax=2, year=2018, kind="yearly"),
    ]
    client = StubClient(
        {"t1": SearchResult(("1", "2"), 100), "t2": SearchResult(("2", "3"), 40)},
        [Document(pmid=p, title="t", abstract="a", year=2018) for p in ("1", "2", "3")],
    )
    result = run_ingest(client, specs)
    assert client.fetched == [["1", "2", "3"]]
    labels = {d.pmid: d.query_label for d in result.documents}
    assert labels == {"1": "pd", "2": "pd", "3": "ad"}
    assert result.search_counts == (("pd", 2018, "yearly", 100, 2), ("ad", 2018, "yearly", 40, 2))
