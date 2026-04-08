from __future__ import annotations

from dataclasses import dataclass
import ssl
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from leopardi.data_pipeline.registry import (
    SourceEndpointEntry,
    load_source_endpoints,
)


@dataclass(slots=True)
class SourceProbeResult:
    source_id: str
    probe_policy: str
    status: str
    detail: str
    http_status: int | None = None
    final_url: str | None = None


def _iter_selected_endpoints(
    selected_source_ids: set[str] | None = None,
) -> Iterable[SourceEndpointEntry]:
    endpoints = load_source_endpoints()
    for entry in endpoints:
        if selected_source_ids is None or entry.source_id in selected_source_ids:
            yield entry


def _probe_http(entry: SourceEndpointEntry, timeout_seconds: float) -> SourceProbeResult:
    ssl_context = ssl.create_default_context()
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    request = Request(
        entry.probe_url,
        method=entry.probe_method.upper(),
        headers={
            "User-Agent": "leopardi-ocr-data-probe/0.1",
            "Accept": "*/*",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
            response.read(1024)
            return SourceProbeResult(
                source_id=entry.source_id,
                probe_policy=entry.probe_policy,
                status="ok",
                detail="Endpoint reachable.",
                http_status=getattr(response, "status", None),
                final_url=response.geturl(),
            )
    except HTTPError as exc:
        return SourceProbeResult(
            source_id=entry.source_id,
            probe_policy=entry.probe_policy,
            status="failed",
            detail=f"HTTP error {exc.code}: {exc.reason}",
            http_status=exc.code,
            final_url=entry.probe_url,
        )
    except URLError as exc:
        detail = f"URL error: {exc.reason}"
        if "CERTIFICATE_VERIFY_FAILED" in detail:
            detail = (
                f"{detail}. Check the machine trust store or install/use a certifi-backed CA bundle "
                "before large dataset fetches."
            )
        return SourceProbeResult(
            source_id=entry.source_id,
            probe_policy=entry.probe_policy,
            status="failed",
            detail=detail,
            final_url=entry.probe_url,
        )


def probe_sources(
    *,
    selected_source_ids: set[str] | None = None,
    timeout_seconds: float = 8.0,
) -> list[SourceProbeResult]:
    results: list[SourceProbeResult] = []
    for entry in _iter_selected_endpoints(selected_source_ids):
        if entry.probe_policy == "internal":
            results.append(
                SourceProbeResult(
                    source_id=entry.source_id,
                    probe_policy=entry.probe_policy,
                    status="skipped",
                    detail="Derived internal source; no remote probe required.",
                )
            )
            continue
        if entry.probe_policy == "manual":
            results.append(
                SourceProbeResult(
                    source_id=entry.source_id,
                    probe_policy=entry.probe_policy,
                    status="manual",
                    detail="Manual or conditional acquisition path required.",
                )
            )
            continue
        results.append(_probe_http(entry, timeout_seconds=timeout_seconds))
    return results
