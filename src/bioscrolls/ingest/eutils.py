"""Minimal, polite NCBI E-utilities client (esearch + efetch) with an on-disk cache.

* Rate limited to NCBI's published ceilings (3 req/s anonymous, 10 req/s with a key).
* Every successful response is cached by request parameters (credentials excluded),
  so re-running ingestion is free and reproducible.
* Retries transient failures (429/5xx, connection errors) with exponential backoff.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import requests

from bioscrolls import config
from bioscrolls.ingest.pubmed_xml import parse_efetch_xml
from bioscrolls.models import Document

log = logging.getLogger(__name__)

_CREDENTIAL_PARAMS = frozenset({"api_key", "email", "tool"})


class EutilsError(RuntimeError):
    """Raised when an E-utilities request fails permanently."""


@dataclass(frozen=True)
class SearchResult:
    ids: tuple[str, ...]
    count: int


class RateLimiter:
    """Blocks so that consecutive calls are at least ``min_interval`` seconds apart."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval <= 0:
            raise ValueError("min_interval must be positive")
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self._min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


class ResponseCache:
    """Content-addressed text cache keyed on endpoint + non-credential parameters."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def key(self, endpoint: str, params: Mapping[str, str]) -> str:
        stable = {k: v for k, v in sorted(params.items()) if k not in _CREDENTIAL_PARAMS}
        blob = json.dumps({"endpoint": endpoint, "params": stable}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.txt"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def put(self, key: str, text: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


class EutilsClient:
    def __init__(
        self,
        cache: ResponseCache | None = None,
        session: requests.Session | None = None,
        api_key: str | None = None,
        email: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache = cache or ResponseCache(config.EUTILS_CACHE_DIR)
        self._session = session or requests.Session()
        self._api_key = api_key if api_key is not None else os.environ.get(config.NCBI_API_KEY_ENV)
        self._email = email if email is not None else os.environ.get(config.NCBI_EMAIL_ENV, "")
        self.min_interval = config.MIN_INTERVAL_WITH_KEY_S if self._api_key else config.MIN_INTERVAL_NO_KEY_S
        self._sleep = sleep
        self._limiter = RateLimiter(self.min_interval, clock=clock, sleep=sleep)

    # ------------------------------------------------------------------ public API

    def esearch(self, term: str, retmax: int, retstart: int = 0) -> SearchResult:
        if not term or not term.strip():
            raise ValueError("search term must be non-empty")
        if not 0 < retmax <= config.ESEARCH_MAX_RETMAX:
            raise ValueError(f"retmax must be in 1..{config.ESEARCH_MAX_RETMAX}")
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": str(retmax),
            "retstart": str(retstart),
            "retmode": "json",
        }
        text = self._request("esearch", params)
        try:
            payload = json.loads(text)["esearchresult"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EutilsError(f"esearch returned invalid JSON for {term!r}") from exc
        if "ERROR" in payload:
            raise EutilsError(f"esearch error for {term!r}: {payload['ERROR']}")
        return SearchResult(ids=tuple(payload.get("idlist", [])), count=int(payload.get("count", 0)))

    def efetch(self, pmids: Sequence[str]) -> list[Document]:
        ids = list(pmids)
        bad = [p for p in ids if not str(p).isdigit()]
        if bad:
            raise ValueError(f"invalid PMIDs: {bad[:5]}")
        documents: list[Document] = []
        for start in range(0, len(ids), config.EFETCH_BATCH_SIZE):
            batch = ids[start : start + config.EFETCH_BATCH_SIZE]
            params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            documents.extend(parse_efetch_xml(self._request("efetch", params)))
        return documents

    # ------------------------------------------------------------------ internals

    def _request(self, endpoint: str, params: Mapping[str, str]) -> str:
        key = self._cache.key(endpoint, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        text = self._fetch_with_retries(endpoint, self._with_credentials(params))
        self._cache.put(key, text)
        return text

    def _with_credentials(self, params: Mapping[str, str]) -> dict[str, str]:
        full = {**params, "tool": config.EUTILS_TOOL_NAME}
        if self._email:
            full["email"] = self._email
        if self._api_key:
            full["api_key"] = self._api_key
        return full

    def _fetch_with_retries(self, endpoint: str, params: Mapping[str, str]) -> str:
        url = f"{config.EUTILS_BASE_URL}/{endpoint}.fcgi"
        last_error = "unknown error"
        for attempt in range(config.HTTP_MAX_RETRIES + 1):
            if attempt:
                self._sleep(config.HTTP_BACKOFF_BASE_S * 2 ** (attempt - 1))
            self._limiter.wait()
            try:
                response = self._session.post(url, data=params, timeout=config.HTTP_TIMEOUT_S)
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                log.warning("%s attempt %d failed: %s", endpoint, attempt + 1, last_error)
                continue
            if response.status_code == 200:
                return response.text
            last_error = f"HTTP {response.status_code}"
            if response.status_code not in config.RETRYABLE_STATUS:
                break
            log.warning("%s attempt %d failed: %s", endpoint, attempt + 1, last_error)
        raise EutilsError(f"{endpoint} failed: {last_error}")
