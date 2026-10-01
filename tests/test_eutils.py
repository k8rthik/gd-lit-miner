import json

import pytest

from bioscrolls.ingest.eutils import EutilsClient, EutilsError, RateLimiter, ResponseCache


class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data=None, timeout=None):
        self.calls.append((url, dict(data or {})))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(tmp_path, responses, api_key=None):
    clock = FakeClock()
    session = FakeSession(responses)
    client = EutilsClient(
        cache=ResponseCache(tmp_path),
        session=session,
        api_key=api_key,
        email="test@example.org",
        clock=clock.time,
        sleep=clock.sleep,
    )
    return client, session, clock


def esearch_payload(ids, count=None):
    return json.dumps({"esearchresult": {"count": str(count or len(ids)), "idlist": ids}})


def test_rate_limiter_enforces_min_interval():
    clock = FakeClock()
    limiter = RateLimiter(0.34, clock=clock.time, sleep=clock.sleep)
    limiter.wait()
    limiter.wait()
    clock.now += 1.0
    limiter.wait()
    assert clock.sleeps == [pytest.approx(0.34)]


def test_rate_limiter_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_esearch_returns_ids_and_count(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(text=esearch_payload(["1", "2"], 50))])
    result = client.esearch("parkinson", retmax=2)
    assert result.ids == ("1", "2")
    assert result.count == 50
    url, params = session.calls[0]
    assert url.endswith("esearch.fcgi")
    assert params["term"] == "parkinson"
    assert params["tool"] == "bioscrolls"
    assert params["email"] == "test@example.org"
    assert "api_key" not in params


def test_responses_are_cached(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(text=esearch_payload(["1"]))])
    first = client.esearch("q", retmax=1)
    second = client.esearch("q", retmax=1)
    assert first == second
    assert len(session.calls) == 1


def test_cache_key_ignores_credentials(tmp_path):
    cache = ResponseCache(tmp_path)
    a = cache.key("esearch", {"term": "x", "api_key": "secret", "email": "a"})
    b = cache.key("esearch", {"term": "x"})
    assert a == b


def test_retries_on_server_error_then_succeeds(tmp_path):
    responses = [FakeResponse(503), FakeResponse(429), FakeResponse(text=esearch_payload(["9"]))]
    client, session, clock = make_client(tmp_path, responses)
    assert client.esearch("q", retmax=1).ids == ("9",)
    assert len(session.calls) == 3
    assert any(s >= 1.0 for s in clock.sleeps)


def test_gives_up_after_max_retries(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(500)] * 10)
    with pytest.raises(EutilsError, match="500"):
        client.esearch("q", retmax=1)


def test_non_retryable_status_fails_fast(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(400, "bad")])
    with pytest.raises(EutilsError, match="400"):
        client.esearch("q", retmax=1)
    assert len(session.calls) == 1


def test_network_exception_is_retried(tmp_path):
    import requests

    responses = [requests.ConnectionError("boom"), FakeResponse(text=esearch_payload(["3"]))]
    client, _, _ = make_client(tmp_path, responses)
    assert client.esearch("q", retmax=1).ids == ("3",)


def test_invalid_json_raises(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(text="not json")])
    with pytest.raises(EutilsError, match="JSON"):
        client.esearch("q", retmax=1)


def test_esearch_error_payload_raises(tmp_path):
    payload = json.dumps({"esearchresult": {"ERROR": "Invalid query"}})
    client, _, _ = make_client(tmp_path, [FakeResponse(text=payload)])
    with pytest.raises(EutilsError, match="Invalid query"):
        client.esearch("q", retmax=1)


def test_efetch_batches_and_parses(tmp_path, efetch_xml):
    client, session, _ = make_client(tmp_path, [FakeResponse(text=efetch_xml)])
    docs = client.efetch(["31210462", "30869416", "31367748"])
    assert len(docs) == 3
    assert session.calls[0][1]["id"] == "31210462,30869416,31367748"


def test_efetch_validates_ids(tmp_path):
    client, _, _ = make_client(tmp_path, [])
    with pytest.raises(ValueError):
        client.efetch(["12a"])
    assert client.efetch([]) == []


def test_esearch_validates_inputs(tmp_path):
    client, _, _ = make_client(tmp_path, [])
    with pytest.raises(ValueError):
        client.esearch("", retmax=1)
    with pytest.raises(ValueError):
        client.esearch("q", retmax=0)


def test_api_key_tightens_rate_and_is_sent(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(text=esearch_payload([]))], api_key="k")
    client.esearch("q", retmax=1)
    assert session.calls[0][1]["api_key"] == "k"
    assert client.min_interval < 0.2
