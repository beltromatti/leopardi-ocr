from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import shutil
import ssl
import statistics
import tarfile
from typing import Any, Iterator
from urllib.request import HTTPSHandler, HTTPCookieProcessor, Request, build_opener
import xml.etree.ElementTree as ET
import zipfile

from leopardi.data_pipeline.canonicalize import (
    jats_to_markdown,
    normalize_target_text,
    project_markdown_to_pages,
    tex_to_markdown,
)
from leopardi.data_pipeline.config import DataBuildStageConfig
from leopardi.data_pipeline.schemas import CanonicalAsset, CanonicalSample
from leopardi.data_pipeline.synthdog import iter_synthdog_european_samples


USER_AGENT = "leopardi-ocr-data-pipeline/0.1"

_HF_STREAMING_COLUMNS: dict[str, tuple[str, ...]] = {
    "publaynet": ("image", "image_id", "page_id", "id", "annotations"),
    "mathwriting": ("image", "latex", "label", "formula", "text", "transcription", "gt", "id"),
    "im2latex_100k": ("image", "latex", "label", "formula", "text", "transcription", "gt", "id"),
    "crohme": ("image", "latex", "label", "formula", "text", "transcription", "gt", "id"),
    "funsd": ("image", "id", "words", "bboxes", "ner_tags"),
    "cord": ("image", "id", "ground_truth"),
    "sroie": ("image", "id", "entities", "words", "bboxes"),
    "chartqa": ("image", "id", "question", "query", "answer", "label"),
    "plotqa": ("image", "id", "text"),
    "fintabnet_family": ("image", "id", "cells", "cols", "filename"),
    "iam": ("image", "id", "text", "transcription"),
}


@dataclass(slots=True)
class SourceBuildContext:
    stage: DataBuildStageConfig
    experiment_id: str
    bundle_id: str
    bundle_class: str
    raw_cache_dir: Path
    work_cache_dir: Path
    manual_source_root: Path | None = None
    render_pdf_pages: bool = True
    publish_enabled: bool = False
    keep_raw: bool = False
    source_limits: dict[str, int] | None = None
    max_pages_per_document: dict[str, int] | None = None

    def source_limit(self, source_id: str, default: int) -> int:
        if self.source_limits and source_id in self.source_limits:
            return self.source_limits[source_id]
        return default

    def page_limit(self, source_id: str, default: int) -> int:
        if self.max_pages_per_document and source_id in self.max_pages_per_document:
            return self.max_pages_per_document[source_id]
        return default


class SourceWorker:
    source_id: str

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        raise NotImplementedError


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return context


def _build_opener():
    import http.cookiejar

    cookie_jar = http.cookiejar.CookieJar()
    return build_opener(HTTPSHandler(context=_ssl_context()), HTTPCookieProcessor(cookie_jar))


def _request(url: str, *, headers: dict[str, str] | None = None) -> Request:
    return Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})},
    )


def _download_bytes(url: str) -> bytes:
    opener = _build_opener()
    with opener.open(_request(url), timeout=120) as response:
        return response.read()


def _download_to_path(url: str, target: Path) -> Path:
    opener = _build_opener()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f"{target.name}.part")
    if temp_target.exists():
        temp_target.unlink()
    with opener.open(_request(url), timeout=120) as response, temp_target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
        expected_size = response.headers.get("Content-Length")
    if expected_size is not None and temp_target.stat().st_size != int(expected_size):
        temp_target.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded size mismatch for {url}")
    temp_target.replace(target)
    return target


def _download_google_drive_file(file_id: str, target: Path) -> Path:
    opener = _build_opener()
    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with opener.open(_request(base_url), timeout=120) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            return target
        html = payload.decode("utf-8", "replace")
    confirm_match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
    if confirm_match is None:
        confirm_match = re.search(r'name="confirm" value="([^"]+)"', html)
    if confirm_match is None:
        raise RuntimeError("Unable to resolve Google Drive confirmation token")
    confirm = confirm_match.group(1)
    confirmed_url = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}"
    return _download_to_path(confirmed_url, target)


def _is_valid_cached_download(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    lowered = path.name.lower()
    if lowered.endswith(".zip"):
        return zipfile.is_zipfile(path)
    if lowered.endswith((".tar", ".tar.gz", ".tgz", ".tbz", ".tar.bz2")):
        return tarfile.is_tarfile(path)
    return True


def _render_pdf_pages(pdf_path: Path, *, max_pages: int) -> list[CanonicalAsset]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF rendering") from exc

    assets: list[CanonicalAsset] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            if page_index >= max_pages:
                break
            pixmap = page.get_pixmap(dpi=192, alpha=False)
            assets.append(
                CanonicalAsset(
                    name=f"page-{page_index + 1:04d}.png",
                    media_type="image/png",
                    payload_bytes=pixmap.tobytes("png"),
                )
            )
    return assets


def _extract_pdf_page_texts(pdf_path: Path, *, max_pages: int) -> list[str]:
    try:
        import fitz
    except ImportError:
        return []
    texts: list[str] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            if page_index >= max_pages:
                break
            texts.append(normalize_target_text(page.get_text("text")))
    return texts


def _choose_main_tex_file(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.tex"))
    if not candidates:
        return None
    scored: list[tuple[int, int, Path]] = []
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        score = 0
        if "\\documentclass" in text:
            score += 10
        if "\\begin{document}" in text:
            score += 8
        score += text.count("\\section")
        score += text.count("\\subsection")
        scored.append((-score, len(str(candidate)), candidate))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(path=target_dir)
    return target_dir


def _iter_arxiv_oai_records(*, limit: int, from_date: str) -> Iterator[dict[str, str]]:
    namespace = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "arxiv": "http://arxiv.org/OAI/arXivRaw/",
    }
    opener = _build_opener()
    next_url = (
        "https://export.arxiv.org/oai2?verb=ListRecords"
        f"&metadataPrefix=arXivRaw&from={from_date}"
    )
    yielded = 0
    while next_url and yielded < limit:
        with opener.open(_request(next_url), timeout=120) as response:
            root = ET.fromstring(response.read())
        for record in root.findall(".//oai:record", namespace):
            if yielded >= limit:
                break
            metadata = record.find("./oai:metadata/arxiv:arXivRaw", namespace)
            if metadata is None:
                continue
            arxiv_id = metadata.findtext("arxiv:id", default="", namespaces=namespace).strip()
            if not arxiv_id:
                continue
            yielded += 1
            yield {
                "id": arxiv_id,
                "title": metadata.findtext("arxiv:title", default="", namespaces=namespace).strip(),
                "license": metadata.findtext("arxiv:license", default="", namespaces=namespace).strip(),
                "categories": metadata.findtext(
                    "arxiv:categories", default="", namespaces=namespace
                ).strip(),
            }
        token = root.findtext(".//oai:resumptionToken", default="", namespaces=namespace).strip()
        next_url = (
            f"https://export.arxiv.org/oai2?verb=ListRecords&resumptionToken={token}"
            if token
            else ""
        )


class ArxivSourceWorker(SourceWorker):
    source_id = "arxiv_source_pdf"

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 50000)
        max_pages = context.page_limit(self.source_id, 12)
        from_date = "2012-01-01" if context.stage.target_param_budget_m <= 200 else "2008-01-01"
        source_root = context.raw_cache_dir / self.source_id
        source_root.mkdir(parents=True, exist_ok=True)
        for record in _iter_arxiv_oai_records(limit=limit, from_date=from_date):
            doc_id = record["id"].replace("/", "_")
            doc_root = source_root / doc_id
            try:
                pdf_path = doc_root / f"{doc_id}.pdf"
                src_path = doc_root / f"{doc_id}.src"
                extract_root = doc_root / "src"
                if not pdf_path.exists():
                    _download_to_path(f"https://export.arxiv.org/pdf/{record['id']}.pdf", pdf_path)
                if not src_path.exists():
                    _download_to_path(f"https://export.arxiv.org/e-print/{record['id']}", src_path)
                if not extract_root.exists():
                    extract_root.mkdir(parents=True, exist_ok=True)
                    try:
                        _safe_extract_tar(src_path, extract_root)
                    except tarfile.ReadError:
                        try:
                            data = gzip.decompress(src_path.read_bytes())
                            (extract_root / "main.tex").write_bytes(data)
                        except OSError:
                            pass
                main_tex = _choose_main_tex_file(extract_root)
                if main_tex is None:
                    continue
                canonical_markdown = tex_to_markdown(
                    main_tex.read_text(encoding="utf-8", errors="ignore")
                )
                if not canonical_markdown:
                    continue
                if context.bundle_id in {"tokenizer_v1", "p1_text_warmup_v1"}:
                    yield CanonicalSample(
                        sample_id=f"{self.source_id}-{doc_id}-document",
                        source_id=self.source_id,
                        bundle_id=context.bundle_id,
                        doc_id=doc_id,
                        data_class="exact_pair",
                        task_family="document_parsing",
                        target_type="document_markdown",
                        canonical_target=canonical_markdown,
                        slice_tags=("exact", "scientific", "latex"),
                        metadata={"arxiv_id": record["id"], "title": record["title"]},
                        source_license=record["license"] or None,
                    )
                    continue
                page_texts = _extract_pdf_page_texts(pdf_path, max_pages=max_pages)
                page_targets = project_markdown_to_pages(canonical_markdown, page_texts)
                page_assets = _render_pdf_pages(pdf_path, max_pages=max_pages) if context.render_pdf_pages else ()
                for page_index, asset in enumerate(page_assets):
                    target = page_targets[page_index] if page_index < len(page_targets) else ""
                    if not target:
                        continue
                    yield CanonicalSample(
                        sample_id=f"{self.source_id}-{doc_id}-page-{page_index + 1:04d}",
                        source_id=self.source_id,
                        bundle_id=context.bundle_id,
                        doc_id=doc_id,
                        page_id=f"{doc_id}:{page_index + 1}",
                        data_class="exact_pair",
                        task_family="document_parsing",
                        target_type="page_markdown_projection",
                        canonical_target=target,
                        slice_tags=("exact", "scientific", "page_projection"),
                        metadata={
                            "arxiv_id": record["id"],
                            "title": record["title"],
                            "page_index": page_index,
                        },
                        assets=(asset,),
                        source_license=record["license"] or None,
                    )
            finally:
                if not context.keep_raw and doc_root.exists():
                    shutil.rmtree(doc_root)


def _pmc_oa_filelist_urls() -> tuple[str, ...]:
    return (
        "https://pmc-oa-opendata.s3.amazonaws.com/oa_comm/xml/metadata/csv/oa_comm.filelist.csv",
        "https://pmc-oa-opendata.s3.amazonaws.com/oa_noncomm/xml/metadata/csv/oa_noncomm.filelist.csv",
        "https://pmc-oa-opendata.s3.amazonaws.com/phe_timebound/xml/metadata/csv/phe_timebound.filelist.csv",
    )


def _iter_pmc_records(limit: int) -> Iterator[dict[str, str]]:
    yielded = 0
    opener = _build_opener()
    seen: set[str] = set()
    for url in _pmc_oa_filelist_urls():
        with opener.open(_request(url), timeout=120) as response:
            payload = response.read().decode("utf-8", "replace").splitlines()
        reader = csv.reader(payload)
        for row in reader:
            if yielded >= limit:
                return
            if not row or len(row) < 7:
                continue
            if row[0] == "Key":
                continue
            pmcid = row[3].strip()
            if not pmcid.startswith("PMC") or pmcid in seen:
                continue
            seen.add(pmcid)
            yielded += 1
            yield {
                "pmcid": pmcid,
                "citation": row[2].strip(),
                "license": row[6].strip(),
            }


class PMCSourceWorker(SourceWorker):
    source_id = "pmc_oa_pdf_xml"

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 2000)
        max_pages = context.page_limit(self.source_id, 8)
        source_root = context.raw_cache_dir / self.source_id
        source_root.mkdir(parents=True, exist_ok=True)
        opener = _build_opener()
        for record in _iter_pmc_records(limit):
            pmcid = record["pmcid"]
            doc_root = source_root / pmcid
            try:
                package_path = doc_root / f"{pmcid}.tar.gz"
                extract_root = doc_root / "package"
                if not _is_valid_cached_download(package_path):
                    package_path.unlink(missing_ok=True)
                    with opener.open(
                        _request(f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"),
                        timeout=120,
                    ) as response:
                        xml = ET.fromstring(response.read())
                    link = xml.find(".//record/link")
                    if link is None:
                        continue
                    href = link.attrib["href"].replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
                    _download_to_path(href, package_path)
                if not extract_root.exists():
                    _safe_extract_tar(package_path, extract_root)
                xml_files = list(extract_root.rglob("*.xml")) + list(extract_root.rglob("*.nxml"))
                pdf_files = list(extract_root.rglob("*.pdf"))
                if not xml_files or not pdf_files:
                    continue
                xml_text = xml_files[0].read_text(encoding="utf-8", errors="ignore")
                canonical_markdown = jats_to_markdown(xml_text)
                if not canonical_markdown:
                    continue
                if context.bundle_id in {"tokenizer_v1", "p1_text_warmup_v1"}:
                    yield CanonicalSample(
                        sample_id=f"{self.source_id}-{pmcid}-document",
                        source_id=self.source_id,
                        bundle_id=context.bundle_id,
                        doc_id=pmcid,
                        data_class="exact_pair",
                        task_family="document_parsing",
                        target_type="document_markdown",
                        canonical_target=canonical_markdown,
                        slice_tags=("exact", "jats", "biomedical"),
                        metadata={"pmcid": pmcid, "citation": record["citation"]},
                        source_license=record["license"] or None,
                    )
                    continue
                pdf_path = pdf_files[0]
                page_texts = _extract_pdf_page_texts(pdf_path, max_pages=max_pages)
                page_targets = project_markdown_to_pages(canonical_markdown, page_texts)
                page_assets = _render_pdf_pages(pdf_path, max_pages=max_pages) if context.render_pdf_pages else ()
                for page_index, asset in enumerate(page_assets):
                    target = page_targets[page_index] if page_index < len(page_targets) else ""
                    if not target:
                        continue
                    yield CanonicalSample(
                        sample_id=f"{self.source_id}-{pmcid}-page-{page_index + 1:04d}",
                        source_id=self.source_id,
                        bundle_id=context.bundle_id,
                        doc_id=pmcid,
                        page_id=f"{pmcid}:{page_index + 1}",
                        data_class="exact_pair",
                        task_family="document_parsing",
                        target_type="page_markdown_projection",
                        canonical_target=target,
                        slice_tags=("exact", "jats", "page_projection"),
                        metadata={
                            "pmcid": pmcid,
                            "citation": record["citation"],
                            "page_index": page_index,
                        },
                        assets=(asset,),
                        source_license=record["license"] or None,
                    )
            finally:
                if not context.keep_raw and doc_root.exists():
                    shutil.rmtree(doc_root)


def _preferred_hf_config_name(repo_id: str) -> str | None:
    from datasets import get_dataset_config_names

    config_names = list(get_dataset_config_names(repo_id))
    if not config_names:
        return None
    if "default" in config_names:
        return "default"
    return sorted(config_names)[0]


def _streaming_dataset_kwargs(repo_id: str, *, cache_dir: Path | None = None) -> dict[str, Any]:
    config_name = _preferred_hf_config_name(repo_id)
    kwargs: dict[str, Any] = {"path": repo_id, "streaming": True}
    if config_name is not None:
        kwargs["name"] = config_name
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def _select_streaming_columns(dataset: Any, *, columns: tuple[str, ...] | None) -> Any:
    if not columns:
        return dataset
    column_names = getattr(dataset, "column_names", None)
    if not isinstance(column_names, list):
        return dataset
    keep = [column for column in columns if column in column_names]
    if not keep or set(keep) == set(column_names):
        return dataset
    if hasattr(dataset, "select_columns"):
        return dataset.select_columns(keep)
    if hasattr(dataset, "remove_columns"):
        remove = [column for column in column_names if column not in keep]
        if remove:
            return dataset.remove_columns(remove)
    return dataset


def _iter_hf_parquet_rows(
    repo_id: str,
    *,
    source_id: str | None = None,
    cache_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    columns = _HF_STREAMING_COLUMNS.get(source_id or "", None)
    dataset_or_dict = load_dataset(**_streaming_dataset_kwargs(repo_id, cache_dir=cache_dir))
    if hasattr(dataset_or_dict, "items"):
        split_items = sorted(
            dataset_or_dict.items(),
            key=lambda item: ({"train": 0, "validation": 1, "test": 2}.get(item[0], 9), item[0]),
        )
        for _, split_dataset in split_items:
            split_dataset = _select_streaming_columns(split_dataset, columns=columns)
            for row in split_dataset:
                yield dict(row)
        return
    dataset = _select_streaming_columns(dataset_or_dict, columns=columns)
    for row in dataset:
        yield dict(row)


def _infer_hf_split_from_filename(filename: str, *, source_id: str | None = None) -> str:
    lowered = filename.lower()
    if "validation" in lowered or "/val" in lowered or lowered.endswith("val.parquet"):
        return "validation"
    if "test" in lowered:
        return "test"
    if source_id == "crohme":
        if any(token in lowered for token in ("/2014", "2014-", "/2016", "2016-", "/2019", "2019-", "/2023", "2023-")):
            return "test"
    return "train"


def _iter_hf_parquet_rows_with_split(
    repo_id: str,
    *,
    source_id: str | None = None,
    cache_dir: Path | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    from datasets import load_dataset

    columns = _HF_STREAMING_COLUMNS.get(source_id or "", None)
    dataset_or_dict = load_dataset(**_streaming_dataset_kwargs(repo_id, cache_dir=cache_dir))
    if hasattr(dataset_or_dict, "items"):
        split_items = sorted(
            dataset_or_dict.items(),
            key=lambda item: ({"train": 0, "validation": 1, "test": 2}.get(item[0], 9), item[0]),
        )
        for split_name, split_dataset in split_items:
            split_dataset = _select_streaming_columns(split_dataset, columns=columns)
            normalized_split = split_name if split_name in {"train", "validation", "test"} else "train"
            for row in split_dataset:
                yield normalized_split, dict(row)
        return
    dataset = _select_streaming_columns(dataset_or_dict, columns=columns)
    for row in dataset:
        yield "train", dict(row)


def _extract_image_asset(row: dict[str, Any]) -> CanonicalAsset | None:
    image = row.get("image")
    if isinstance(image, dict) and image.get("bytes"):
        suffix = Path(image.get("path") or "image.png").suffix.lstrip(".") or "png"
        media_type = f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else 'png'}"
        return CanonicalAsset(
            name=f"image.{suffix}",
            media_type=media_type,
            payload_bytes=image["bytes"],
        )
    return None


def _json_target(row: dict[str, Any]) -> str:
    def _json_safe(value: Any) -> Any:
        if isinstance(value, bytes):
            return {"bytes_len": len(value)}
        if isinstance(value, dict):
            return {key: _json_safe(item) for key, item in value.items() if key != "bytes"}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return value

    sanitized = {key: _json_safe(value) for key, value in row.items() if key != "image"}
    return normalize_target_text(json.dumps(sanitized, ensure_ascii=True, sort_keys=True))


def _markdown_escape_cell(text: str) -> str:
    return normalize_target_text(text).replace("|", r"\|")


def _simple_markdown_table(rows: list[list[str]], caption: str = "") -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:] or [["" for _ in header]]
    lines: list[str] = []
    if caption:
        lines.append(f"Table: {normalize_target_text(caption)}")
        lines.append("")
    lines.append("| " + " | ".join(_markdown_escape_cell(cell) for cell in header) + " |")
    lines.append("| " + " | ".join(":---" for _ in header) + " |")
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(_markdown_escape_cell(cell) for cell in padded[: len(header)]) + " |")
    return "\n".join(lines)


def _flatten_mapping(prefix: str, payload: Any) -> list[tuple[str, str]]:
    if isinstance(payload, dict):
        rows: list[tuple[str, str]] = []
        for key, value in payload.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_mapping(nested_prefix, value))
        return rows
    if isinstance(payload, list):
        if all(not isinstance(item, (dict, list)) for item in payload):
            return [(prefix, normalize_target_text(", ".join(str(item) for item in payload)))]
        rows: list[tuple[str, str]] = []
        for index, value in enumerate(payload):
            rows.extend(_flatten_mapping(f"{prefix}[{index}]", value))
        return rows
    return [(prefix, normalize_target_text(str(payload)))]


def _mapping_to_markdown(title: str, payload: dict[str, Any]) -> str:
    lines = [f"## {title}"]
    for key, value in _flatten_mapping("", payload):
        if key and value:
            lines.append(f"- {key}: {value}")
    return normalize_target_text("\n".join(lines))


def _standalone_formula_markdown(formula: str) -> str:
    normalized = normalize_target_text(formula)
    if not normalized:
        return ""
    normalized = re.sub(r"\.\s*\.\s*\.", r"\\dots", normalized)
    normalized = re.sub(r"\s*_\s*\{\s*", "_{", normalized)
    normalized = re.sub(r"\s*\^\s*\{\s*", "^{", normalized)
    normalized = re.sub(r"\{\s*", "{", normalized)
    normalized = re.sub(r"\s*\}", "}", normalized)
    normalized = re.sub(r"\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)", ")", normalized)
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = re.sub(r"\s+;", ";", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=[A-Za-z](?:\b|[^A-Za-z]))", "", normalized)
    normalized = normalize_target_text(normalized)
    if normalized.startswith("$$") and normalized.endswith("$$"):
        return normalized
    if normalized.startswith("$") and normalized.endswith("$"):
        inner = normalized.strip("$").strip()
    else:
        inner = normalized
    return normalize_target_text(f"$$\n{inner}\n$$")


def _words_to_lines(words: list[Any], bboxes: list[Any]) -> list[str]:
    if not words or not bboxes or len(words) != len(bboxes):
        fallback = normalize_target_text(" ".join(str(item) for item in words if str(item).strip()))
        return [fallback] if fallback else []
    items: list[tuple[float, float, float, str]] = []
    heights: list[float] = []
    for word, bbox in zip(words, bboxes):
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        text = normalize_target_text(str(word))
        if not text:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox]
        height = max(1.0, y1 - y0)
        heights.append(height)
        items.append(((y0 + y1) / 2.0, x0, height, text))
    if not items:
        return []
    line_threshold = max(10.0, statistics.median(heights) * 0.8)
    items.sort(key=lambda item: (item[0], item[1]))
    lines: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = []
    current_y = items[0][0]
    for y_center, x0, _, text in items:
        if current and abs(y_center - current_y) > line_threshold:
            lines.append(sorted(current, key=lambda item: item[0]))
            current = []
            current_y = y_center
        current.append((x0, text))
        current_y = (current_y + y_center) / 2.0
    if current:
        lines.append(sorted(current, key=lambda item: item[0]))
    return [
        normalize_target_text(" ".join(text for _, text in line if text))
        for line in lines
        if any(text for _, text in line)
    ]


def _ocr_lines_markdown(title: str, words: list[Any], bboxes: list[Any]) -> str:
    lines = _words_to_lines(words, bboxes)
    payload = [f"## {title}"] if title else []
    payload.extend(lines)
    return normalize_target_text("\n".join(payload))


def _cord_ocr_markdown(payload: dict[str, Any]) -> str:
    valid_line = payload.get("valid_line")
    if not isinstance(valid_line, list):
        return ""
    grouped_lines: list[tuple[int | None, list[str]]] = []
    for item in valid_line:
        if not isinstance(item, dict):
            continue
        words = item.get("words")
        if not isinstance(words, list):
            continue
        tokens: list[str] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            text = normalize_target_text(str(word.get("text") or ""))
            if text:
                tokens.append(text)
        line = normalize_target_text(" ".join(tokens))
        if not line:
            continue
        group_id = item.get("group_id")
        if grouped_lines and grouped_lines[-1][0] == group_id:
            grouped_lines[-1][1].append(line)
        else:
            grouped_lines.append((group_id, [line]))
    if not grouped_lines:
        return ""
    lines = ["## Receipt OCR"]
    for _, parts in grouped_lines:
        merged = normalize_target_text(" ".join(part for part in parts if part))
        if merged:
            lines.append(merged)
    return normalize_target_text("\n".join(lines))


_FUNSD_NER_LABELS = {
    0: "other",
    1: "header",
    2: "header",
    3: "question",
    4: "question",
    5: "answer",
    6: "answer",
}


def _funsd_fields_markdown(words: list[Any], ner_tags: list[Any]) -> str:
    if not isinstance(words, list) or not isinstance(ner_tags, list) or len(words) != len(ner_tags):
        return ""
    grouped: list[tuple[str, str]] = []
    current_label = ""
    current_tokens: list[str] = []
    for word, raw_tag in zip(words, ner_tags):
        try:
            tag = int(raw_tag)
        except (TypeError, ValueError):
            tag = 0
        label = _FUNSD_NER_LABELS.get(tag, "other")
        token = normalize_target_text(str(word))
        if not token:
            continue
        if label == "other":
            if current_tokens and current_label:
                grouped.append((current_label, normalize_target_text(" ".join(current_tokens))))
            current_label = ""
            current_tokens = []
            continue
        if label != current_label and current_tokens and current_label:
            grouped.append((current_label, normalize_target_text(" ".join(current_tokens))))
            current_tokens = []
        current_label = label
        current_tokens.append(token)
    if current_tokens and current_label:
        grouped.append((current_label, normalize_target_text(" ".join(current_tokens))))
    if not grouped:
        return ""
    lines = ["## Form Fields"]
    for label, text in grouped:
        lines.append(f"- {label}: {text}")
    return normalize_target_text("\n".join(lines))


def _fintabnet_table_markdown(row: dict[str, Any]) -> str:
    cells = row.get("cells")
    if not isinstance(cells, list) or not cells:
        return _json_target(row)
    row_groups: list[list[Any]]
    if len(cells) == 1 and isinstance(cells[0], list) and isinstance(row.get("cols"), int):
        flat_cells = cells[0]
        cols = max(1, int(row.get("cols") or 1))
        row_groups = [flat_cells[index : index + cols] for index in range(0, len(flat_cells), cols)]
    else:
        row_groups = [row_cells for row_cells in cells if isinstance(row_cells, list)]
    rows: list[list[str]] = []
    for row_cells in row_groups:
        rendered_row: list[str] = []
        for cell in row_cells:
            if isinstance(cell, dict):
                tokens = cell.get("tokens", [])
                text = "".join(str(token) for token in tokens) if isinstance(tokens, list) else str(tokens)
            else:
                text = str(cell)
            rendered_row.append(normalize_target_text(text))
        if rendered_row:
            rows.append(rendered_row)
    if not rows:
        return _json_target(row)
    max_cols = max(len(row_cells) for row_cells in rows)
    normalized_rows = [row_cells + [""] * (max_cols - len(row_cells)) for row_cells in rows]
    caption = str(row.get("filename") or "").strip()
    return _simple_markdown_table(normalized_rows, caption=caption)


def _plotqa_markdown(text: str) -> str:
    fields = re.findall(r"<s_([^>]+)>(.*?)</s_\1>", text, flags=re.DOTALL)
    if not fields:
        return normalize_target_text(text)
    lines = ["```chart"]
    for name, value in fields:
        if name == "bboxes":
            bbox_entries = [item for item in value.split("<sep/>") if item.strip()]
            rendered_boxes: list[str] = []
            for entry in bbox_entries:
                coords = {
                    match.group(1): normalize_target_text(match.group(2))
                    for match in re.finditer(r"<s_([^>]+)>(.*?)</s_\1>", entry, flags=re.DOTALL)
                }
                if coords:
                    rendered_boxes.append(
                        f"(x={coords.get('x','')}, y={coords.get('y','')}, w={coords.get('w','')}, h={coords.get('h','')})"
                    )
            normalized = " | ".join(rendered_boxes)
        else:
            normalized = normalize_target_text(value.replace("<sep/>", " | "))
        lines.append(f"{name}: {normalized}")
    lines.append("```")
    return "\n".join(lines)


def _hf_row_to_sample(
    *,
    source_id: str,
    bundle_id: str,
    row: dict[str, Any],
    row_index: int,
) -> CanonicalSample:
    image_asset = _extract_image_asset(row)
    doc_id = str(row.get("id") or row.get("image_id") or row.get("uid") or row_index)
    if source_id in {"mathwriting", "im2latex_100k", "crohme", "unimer_1m"}:
        target = ""
        for key in ("latex", "label", "formula", "text", "transcription", "gt"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                target = value
                break
        if not target:
            target = _json_target(row)
        return CanonicalSample(
            sample_id=f"{source_id}-{doc_id}",
            source_id=source_id,
            bundle_id=bundle_id,
            doc_id=doc_id,
            data_class="trusted_aux",
            task_family="formula",
            target_type="latex_formula",
            canonical_target=_standalone_formula_markdown(target),
            slice_tags=(
                ("formula", "handwritten")
                if source_id == "crohme"
                else ("formula", "printed", "large_scale")
                if source_id == "unimer_1m"
                else ("formula", "aux")
            ),
            metadata={key: value for key, value in row.items() if key != "image"},
            assets=(image_asset,) if image_asset else (),
        )
    if source_id == "iam":
        target = str(row.get("text") or row.get("transcription") or "").strip()
        if not target:
            target = _json_target(row)
        return CanonicalSample(
            sample_id=f"{source_id}-{doc_id}",
            source_id=source_id,
            bundle_id=bundle_id,
            doc_id=doc_id,
            data_class="trusted_aux",
            task_family="handwriting",
            target_type="text_line",
            canonical_target=normalize_target_text(target),
            slice_tags=("handwriting", "english", "line_level"),
            metadata={key: value for key, value in row.items() if key != "image"},
            assets=(image_asset,) if image_asset else (),
        )
    if source_id in {"chartqa", "plotqa"}:
        if source_id == "chartqa":
            question = str(row.get("question") or row.get("query") or "").strip()
            answer = row.get("answer") or row.get("label") or ""
            if isinstance(answer, list):
                answer = " | ".join(str(item) for item in answer)
            target = normalize_target_text(f"Question: {question}\nAnswer: {answer}")
            target_type = "chart_qa"
        else:
            target = _plotqa_markdown(str(row.get("text") or ""))
            target_type = "chart_serialization"
        return CanonicalSample(
            sample_id=f"{source_id}-{doc_id}",
            source_id=source_id,
            bundle_id=bundle_id,
            doc_id=doc_id,
            data_class="trusted_aux",
            task_family="graphics",
            target_type=target_type,
            canonical_target=target or _json_target(row),
            slice_tags=("graphics", "qa"),
            metadata={key: value for key, value in row.items() if key != "image"},
            assets=(image_asset,) if image_asset else (),
        )
    if source_id in {"funsd", "cord", "sroie"}:
        target = ""
        if source_id == "cord":
            raw = row.get("ground_truth")
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    target = normalize_target_text(raw)
                else:
                    payload = parsed.get("gt_parse", parsed)
                    blocks: list[str] = []
                    if isinstance(payload, dict):
                        blocks.append(_mapping_to_markdown("Receipt Fields", payload))
                    ocr_block = _cord_ocr_markdown(parsed)
                    if ocr_block:
                        blocks.append(ocr_block)
                    target = normalize_target_text("\n\n".join(block for block in blocks if block))
        elif source_id == "sroie":
            entities = row.get("entities")
            words = row.get("words")
            bboxes = row.get("bboxes")
            blocks: list[str] = []
            if isinstance(entities, dict):
                blocks.append(_mapping_to_markdown("Receipt Fields", entities))
            if isinstance(words, list) and isinstance(bboxes, list):
                blocks.append(_ocr_lines_markdown("Receipt OCR", words, bboxes))
            target = normalize_target_text("\n\n".join(block for block in blocks if block))
        else:
            words = row.get("words")
            bboxes = row.get("bboxes")
            ner_tags = row.get("ner_tags")
            blocks: list[str] = []
            if isinstance(words, list) and isinstance(ner_tags, list):
                fields = _funsd_fields_markdown(words, ner_tags)
                if fields:
                    blocks.append(fields)
            if isinstance(words, list) and isinstance(bboxes, list):
                blocks.append(_ocr_lines_markdown("Form OCR", words, bboxes))
            target = normalize_target_text("\n\n".join(block for block in blocks if block))
        if not target:
            target = _json_target(row)
        return CanonicalSample(
            sample_id=f"{source_id}-{doc_id}",
            source_id=source_id,
            bundle_id=bundle_id,
            doc_id=doc_id,
            data_class="trusted_aux",
            task_family="forms",
            target_type="page_markdown_projection",
            canonical_target=target,
            slice_tags=("forms", "business_docs", "receipts"),
            metadata={key: value for key, value in row.items() if key != "image"},
            assets=(image_asset,) if image_asset else (),
        )
    if source_id == "fintabnet_family":
        return CanonicalSample(
            sample_id=f"{source_id}-{doc_id}",
            source_id=source_id,
            bundle_id=bundle_id,
            doc_id=doc_id,
            data_class="trusted_aux",
            task_family="tables",
            target_type="table_markdown",
            canonical_target=_fintabnet_table_markdown(row),
            slice_tags=("table", "financial", "aligned_benchmark"),
            metadata={key: value for key, value in row.items() if key != "image"},
            assets=(image_asset,) if image_asset else (),
        )
    return CanonicalSample(
        sample_id=f"{source_id}-{doc_id}",
        source_id=source_id,
        bundle_id=bundle_id,
        doc_id=doc_id,
        data_class="trusted_aux",
        task_family="layout",
        target_type="aux_annotation",
        canonical_target=_json_target(row),
        slice_tags=("aux", source_id),
        metadata={key: value for key, value in row.items() if key != "image"},
        assets=(image_asset,) if image_asset else (),
    )


class HFParquetWorker(SourceWorker):
    def __init__(self, source_id: str, repo_id: str) -> None:
        self.source_id = source_id
        self.repo_id = repo_id

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 20000)
        cache_dir = context.raw_cache_dir / self.source_id / "hf-cache"
        for row_index, row in enumerate(
            _iter_hf_parquet_rows(self.repo_id, source_id=self.source_id, cache_dir=cache_dir)
        ):
            if row_index >= limit:
                break
            yield _hf_row_to_sample(
                source_id=self.source_id,
                bundle_id=context.bundle_id,
                row=row,
                row_index=row_index,
            )


class HFSplitAwareParquetWorker(SourceWorker):
    def __init__(self, source_id: str, repo_id: str) -> None:
        self.source_id = source_id
        self.repo_id = repo_id

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 20000)
        cache_dir = context.raw_cache_dir / self.source_id / "hf-cache"
        for row_index, (split_assignment, row) in enumerate(
            _iter_hf_parquet_rows_with_split(
                self.repo_id,
                source_id=self.source_id,
                cache_dir=cache_dir,
            )
        ):
            if row_index >= limit:
                break
            sample = _hf_row_to_sample(
                source_id=self.source_id,
                bundle_id=context.bundle_id,
                row=row,
                row_index=row_index,
            )
            sample.split_assignment = split_assignment
            yield sample


class UniMERArchiveWorker(SourceWorker):
    source_id = "unimer_1m"
    repo_id = "wanderkid/UniMER_Dataset"
    archive_name = "UniMER-1M.zip"

    def _archive_path(self, context: SourceBuildContext) -> Path:
        from huggingface_hub import hf_hub_download

        cache_dir = context.raw_cache_dir / self.source_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=self.archive_name,
                cache_dir=str(cache_dir),
            )
        )

    @staticmethod
    def _iter_manifest_pairs(archive: zipfile.ZipFile) -> Iterator[tuple[str, str, str]]:
        for member_name in archive.namelist():
            lowered = member_name.lower()
            if not lowered.endswith((".jsonl", ".json", ".csv", ".tsv")):
                continue
            if "readme" in lowered or lowered.endswith("/"):
                continue
            with archive.open(member_name) as handle:
                if lowered.endswith(".jsonl"):
                    for line in handle.read().decode("utf-8", "ignore").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        image_path = next((row.get(key) for key in ("image", "image_path", "img", "path", "file_name") if row.get(key)), None)
                        label = next((row.get(key) for key in ("latex", "label", "formula", "gt", "text") if row.get(key)), None)
                        sample_id = str(row.get("id") or row.get("uid") or Path(str(image_path or "sample")).stem)
                        if image_path and isinstance(label, str):
                            yield sample_id, str(image_path), label
                else:
                    rows = handle.read().decode("utf-8", "ignore").splitlines()
                    delimiter = "\t" if lowered.endswith(".tsv") else ","
                    reader = csv.DictReader(rows, delimiter=delimiter)
                    for row in reader:
                        image_path = next((row.get(key) for key in ("image", "image_path", "img", "path", "file_name") if row.get(key)), None)
                        label = next((row.get(key) for key in ("latex", "label", "formula", "gt", "text") if row.get(key)), None)
                        sample_id = str(row.get("id") or row.get("uid") or Path(str(image_path or "sample")).stem)
                        if image_path and isinstance(label, str):
                            yield sample_id, str(image_path), label

    @staticmethod
    def _fallback_stem_pairs(archive: zipfile.ZipFile) -> Iterator[tuple[str, str, str]]:
        image_members: dict[str, str] = {}
        label_members: dict[str, str] = {}
        for member_name in archive.namelist():
            lowered = member_name.lower()
            if lowered.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                image_members[Path(member_name).stem] = member_name
            elif lowered.endswith((".txt", ".tex", ".latex")):
                label_members[Path(member_name).stem] = member_name
        for stem in sorted(set(image_members) & set(label_members)):
            with archive.open(label_members[stem]) as handle:
                label = handle.read().decode("utf-8", "ignore").strip()
            if label:
                yield stem, image_members[stem], label

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 200000)
        archive_path = self._archive_path(context)
        yielded = 0
        with zipfile.ZipFile(archive_path) as archive:
            yielded_ids: set[str] = set()
            for sample_id, image_member, label in self._iter_manifest_pairs(archive):
                if yielded >= limit:
                    return
                try:
                    image_bytes = archive.read(image_member)
                except KeyError:
                    continue
                suffix = Path(image_member).suffix.lstrip(".") or "png"
                yielded_ids.add(sample_id)
                yielded += 1
                yield CanonicalSample(
                    sample_id=f"{self.source_id}-{sample_id}",
                    source_id=self.source_id,
                    bundle_id=context.bundle_id,
                    doc_id=sample_id,
                    data_class="trusted_aux",
                    task_family="formula",
                    target_type="latex_formula",
                    canonical_target=_standalone_formula_markdown(label),
                    slice_tags=("formula", "printed", "large_scale"),
                    metadata={"image_member": image_member, "archive": self.archive_name},
                    assets=(
                        CanonicalAsset(
                            name=f"image.{suffix}",
                            media_type=f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else 'png'}",
                            payload_bytes=image_bytes,
                        ),
                    ),
                    source_license="apache-2.0",
                )
            for sample_id, image_member, label in self._fallback_stem_pairs(archive):
                if yielded >= limit:
                    return
                if sample_id in yielded_ids:
                    continue
                image_bytes = archive.read(image_member)
                suffix = Path(image_member).suffix.lstrip(".") or "png"
                yielded += 1
                yield CanonicalSample(
                    sample_id=f"{self.source_id}-{sample_id}",
                    source_id=self.source_id,
                    bundle_id=context.bundle_id,
                    doc_id=sample_id,
                    data_class="trusted_aux",
                    task_family="formula",
                    target_type="latex_formula",
                    canonical_target=_standalone_formula_markdown(label),
                    slice_tags=("formula", "printed", "large_scale"),
                    metadata={"image_member": image_member, "archive": self.archive_name},
                    assets=(
                        CanonicalAsset(
                            name=f"image.{suffix}",
                            media_type=f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else 'png'}",
                            payload_bytes=image_bytes,
                        ),
                    ),
                    source_license="apache-2.0",
                )


class SynthDoGEuropeanWorker(SourceWorker):
    source_id = "synthdog_european"

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 100000)
        for sample in iter_synthdog_european_samples(total_limit=limit):
            yield CanonicalSample(
                sample_id=sample.sample_id,
                source_id=self.source_id,
                bundle_id=context.bundle_id,
                doc_id=sample.doc_id,
                page_id=f"{sample.doc_id}:1",
                data_class="synthetic_exact",
                task_family="document_parsing",
                target_type="page_markdown_projection",
                canonical_target=sample.canonical_target,
                slice_tags=("synthetic", "multilingual", "european", sample.language),
                metadata={
                    "language": sample.language,
                    "source": "wikimedia/wikipedia",
                    "title": sample.title,
                },
                assets=(
                    CanonicalAsset(
                        name="image.png",
                        media_type="image/png",
                        payload_bytes=sample.image_png,
                    ),
                ),
                source_license="cc-by-sa",
            )


class DocLayNetWorker(SourceWorker):
    source_id = "doclaynet"
    zip_url = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 15000)
        zip_path = context.raw_cache_dir / self.source_id / "DocLayNet_core.zip"
        if not _is_valid_cached_download(zip_path):
            zip_path.unlink(missing_ok=True)
            _download_to_path(self.zip_url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            annotation_files = [name for name in archive.namelist() if name.endswith(".json")]
            image_names = set(archive.namelist())
            yielded = 0
            for annotation_name in sorted(annotation_files):
                data = json.loads(archive.read(annotation_name))
                images = data.get("images", [])
                annotations = data.get("annotations", [])
                by_image: dict[int, list[dict[str, Any]]] = {}
                for annotation in annotations:
                    by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
                for image_info in images:
                    if yielded >= limit:
                        return
                    image_name = image_info.get("file_name") or ""
                    full_image_name = next(
                        (item for item in image_names if item.endswith(image_name)),
                        None,
                    )
                    assets: tuple[CanonicalAsset, ...] = ()
                    if full_image_name is not None:
                        suffix = Path(full_image_name).suffix.lstrip(".") or "png"
                        assets = (
                            CanonicalAsset(
                                name=f"image.{suffix}",
                                media_type=f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else 'png'}",
                                payload_bytes=archive.read(full_image_name),
                            ),
                        )
                    sample = CanonicalSample(
                        sample_id=f"{self.source_id}-{image_info['id']}",
                        source_id=self.source_id,
                        bundle_id=context.bundle_id,
                        doc_id=str(image_info.get("doc_name") or image_info["id"]),
                        page_id=str(image_info.get("page_no") or image_info["id"]),
                        data_class="trusted_aux",
                        task_family="layout",
                        target_type="aux_annotation",
                        canonical_target=normalize_target_text(
                            json.dumps(by_image.get(int(image_info["id"]), []), sort_keys=True)
                        ),
                        slice_tags=("layout", "human_annotated"),
                        metadata=image_info,
                        assets=assets,
                    )
                    yielded += 1
                    yield sample


def _pascal_xml_to_json(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    objects = []
    for obj in root.findall(".//object"):
        box = obj.find("./bndbox")
        objects.append(
            {
                "name": obj.findtext("./name", default=""),
                "bbox": {
                    "xmin": int(float(box.findtext("./xmin", default="0"))) if box is not None else 0,
                    "ymin": int(float(box.findtext("./ymin", default="0"))) if box is not None else 0,
                    "xmax": int(float(box.findtext("./xmax", default="0"))) if box is not None else 0,
                    "ymax": int(float(box.findtext("./ymax", default="0"))) if box is not None else 0,
                },
            }
        )
    return normalize_target_text(json.dumps(objects, sort_keys=True))


def _archive_stem(path: str) -> str:
    name = Path(path).name
    lowered = name.lower()
    for suffix in (
        ".page.xml",
        ".xml",
        ".txt",
        ".gt",
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    ):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _member_rank(path: str, *, prefer_page_dir: bool = False, prefer_xml: bool = False) -> tuple[int, int]:
    lowered = path.lower()
    page_bonus = 0 if prefer_page_dir and "/page/" in lowered else 1
    xml_bonus = 0 if prefer_xml and lowered.endswith(".xml") else 1
    return (page_bonus, xml_bonus)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _pagexml_to_markdown(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return normalize_target_text(xml_text)

    blocks: list[str] = []
    for region in root.iter():
        if _xml_local_name(region.tag) != "TextRegion":
            continue
        lines: list[str] = []
        for line in region:
            if _xml_local_name(line.tag) != "TextLine":
                continue
            unicode_nodes = [
                node.text.strip()
                for node in line.iter()
                if _xml_local_name(node.tag) == "Unicode" and isinstance(node.text, str) and node.text.strip()
            ]
            if unicode_nodes:
                lines.append(normalize_target_text(" ".join(unicode_nodes)))
        if lines:
            blocks.append("\n".join(lines))
            continue
        region_unicode = [
            node.text.strip()
            for node in region.iter()
            if _xml_local_name(node.tag) == "Unicode" and isinstance(node.text, str) and node.text.strip()
        ]
        if region_unicode:
            blocks.append(normalize_target_text(" ".join(region_unicode)))
    if not blocks:
        fallback = [
            node.text.strip()
            for node in root.iter()
            if _xml_local_name(node.tag) == "Unicode" and isinstance(node.text, str) and node.text.strip()
        ]
        if fallback:
            blocks = [normalize_target_text("\n".join(fallback))]
    return normalize_target_text("\n\n".join(blocks))


def _textlike_target_from_bytes(member_name: str, payload: bytes) -> str:
    lowered = member_name.lower()
    if lowered.endswith(".xml"):
        return _pagexml_to_markdown(payload.decode("utf-8", "ignore"))
    return normalize_target_text(payload.decode("utf-8", "ignore"))


class ArchivePairWorker(SourceWorker):
    def __init__(
        self,
        *,
        source_id: str,
        archives: tuple[tuple[str, str], ...],
        task_family: str,
        target_type: str,
        slice_tags: tuple[str, ...],
        split_from_path: bool = False,
        prefer_page_dir: bool = False,
    ) -> None:
        self.source_id = source_id
        self.archives = archives
        self.task_family = task_family
        self.target_type = target_type
        self.slice_tags = slice_tags
        self.split_from_path = split_from_path
        self.prefer_page_dir = prefer_page_dir

    def _archive_path(self, context: SourceBuildContext, filename: str) -> Path:
        archive_path = context.raw_cache_dir / self.source_id / filename
        if not _is_valid_cached_download(archive_path):
            archive_path.unlink(missing_ok=True)
            url = dict(self.archives)[filename]
            _download_to_path(url, archive_path)
        return archive_path

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 5000)
        archive_paths = [self._archive_path(context, filename) for filename, _ in self.archives]
        archive_handles = [tarfile.open(path, "r:*") for path in archive_paths]
        try:
            image_members: dict[str, tuple[int, tarfile.TarInfo]] = {}
            target_members: dict[str, tuple[int, tarfile.TarInfo]] = {}
            for archive_index, archive in enumerate(archive_handles):
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    lowered = member.name.lower()
                    stem = _archive_stem(member.name)
                    if lowered.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                        candidate = (archive_index, member)
                        existing = image_members.get(stem)
                        if existing is None or _member_rank(
                            member.name, prefer_page_dir=self.prefer_page_dir
                        ) < _member_rank(existing[1].name, prefer_page_dir=self.prefer_page_dir):
                            image_members[stem] = candidate
                    elif lowered.endswith((".xml", ".txt", ".gt")):
                        candidate = (archive_index, member)
                        existing = target_members.get(stem)
                        if existing is None or _member_rank(
                            member.name, prefer_page_dir=self.prefer_page_dir, prefer_xml=True
                        ) < _member_rank(
                            existing[1].name, prefer_page_dir=self.prefer_page_dir, prefer_xml=True
                        ):
                            target_members[stem] = candidate

            yielded = 0
            for stem in sorted(set(image_members) & set(target_members)):
                if yielded >= limit:
                    return
                image_archive_index, image_member = image_members[stem]
                target_archive_index, target_member = target_members[stem]
                image_handle = archive_handles[image_archive_index].extractfile(image_member)
                target_handle = archive_handles[target_archive_index].extractfile(target_member)
                if image_handle is None or target_handle is None:
                    continue
                image_bytes = image_handle.read()
                target = _textlike_target_from_bytes(target_member.name, target_handle.read())
                if not target:
                    continue
                split_assignment = "train"
                if self.split_from_path:
                    lowered = target_member.name.lower()
                    if "/validation/" in lowered or "/val/" in lowered:
                        split_assignment = "validation"
                    elif "/test/" in lowered:
                        split_assignment = "test"
                suffix = Path(image_member.name).suffix.lstrip(".") or "jpg"
                yielded += 1
                yield CanonicalSample(
                    sample_id=f"{self.source_id}-{stem}",
                    source_id=self.source_id,
                    bundle_id=context.bundle_id,
                    doc_id=stem,
                    page_id=f"{stem}:1",
                    data_class="trusted_aux",
                    task_family=self.task_family,
                    target_type=self.target_type,
                    canonical_target=target,
                    split_assignment=split_assignment,
                    slice_tags=self.slice_tags,
                    metadata={
                        "image_member": image_member.name,
                        "target_member": target_member.name,
                    },
                    assets=(
                        CanonicalAsset(
                            name=f"image.{suffix}",
                            media_type=f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else 'png'}",
                            payload_bytes=image_bytes,
                        ),
                    ),
                )
        finally:
            for archive in archive_handles:
                archive.close()


class PubTablesWorker(SourceWorker):
    source_id = "pubtables_1m"

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        from huggingface_hub import hf_hub_download

        limit = context.source_limit(self.source_id, 40000)
        filenames = {
            "images": "PubTables-1M-Structure_Images_Train.tar.gz",
            "annotations": "PubTables-1M-Structure_Annotations_Train.tar.gz",
        }
        cache_dir = context.raw_cache_dir / self.source_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        image_tar = Path(
            hf_hub_download(
                repo_id="bsmock/pubtables-1m",
                repo_type="dataset",
                filename=filenames["images"],
                cache_dir=str(cache_dir),
            )
        )
        annotation_tar = Path(
            hf_hub_download(
                repo_id="bsmock/pubtables-1m",
                repo_type="dataset",
                filename=filenames["annotations"],
                cache_dir=str(cache_dir),
            )
        )
        with tarfile.open(image_tar, "r:gz") as images_archive, tarfile.open(
            annotation_tar, "r:gz"
        ) as annotations_archive:
            image_members = {
                Path(member.name).stem: member
                for member in images_archive.getmembers()
                if member.isfile() and member.name.lower().endswith((".png", ".jpg", ".jpeg"))
            }
            yielded = 0
            for member in annotations_archive:
                if yielded >= limit:
                    return
                if not member.isfile() or not member.name.endswith(".xml"):
                    continue
                stem = Path(member.name).stem
                image_member = image_members.get(stem)
                if image_member is None:
                    continue
                xml_bytes = annotations_archive.extractfile(member).read()
                image_bytes = images_archive.extractfile(image_member).read()
                suffix = Path(image_member.name).suffix.lstrip(".") or "png"
                yielded += 1
                yield CanonicalSample(
                    sample_id=f"{self.source_id}-{stem}",
                    source_id=self.source_id,
                    bundle_id=context.bundle_id,
                    doc_id=stem,
                    data_class="trusted_aux",
                    task_family="tables",
                    target_type="aux_annotation",
                    canonical_target=_pascal_xml_to_json(xml_bytes),
                    slice_tags=("table", "structure"),
                    metadata={"archive_member": member.name},
                    assets=(
                        CanonicalAsset(
                            name=f"image.{suffix}",
                            media_type=f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else 'png'}",
                            payload_bytes=image_bytes,
                        ),
                    ),
                )


class SciTSRWorker(SourceWorker):
    source_id = "scitsr"
    file_id = "1qXaJblBg9sbPN0xknWsYls1aGGtlp4ZN"

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 15000)
        archive_path = context.raw_cache_dir / self.source_id / "SciTSR.tar.gz"
        extract_root = context.work_cache_dir / self.source_id / "SciTSR"
        if not _is_valid_cached_download(archive_path):
            archive_path.unlink(missing_ok=True)
            _download_google_drive_file(self.file_id, archive_path)
        if not extract_root.exists():
            _safe_extract_tar(archive_path, extract_root.parent)
        image_dir = next((item for item in extract_root.rglob("img") if item.is_dir()), None)
        structure_dir = next(
            (item for item in extract_root.rglob("structure") if item.is_dir()),
            None,
        )
        if image_dir is None or structure_dir is None:
            return
        yielded = 0
        for structure_file in sorted(structure_dir.glob("*.json")):
            if yielded >= limit:
                return
            stem = structure_file.stem
            image_file = next(
                (item for item in image_dir.glob(f"{stem}.*") if item.suffix.lower() in {".png", ".jpg", ".jpeg"}),
                None,
            )
            if image_file is None:
                continue
            yielded += 1
            suffix = image_file.suffix.lstrip(".")
            yield CanonicalSample(
                sample_id=f"{self.source_id}-{stem}",
                source_id=self.source_id,
                bundle_id=context.bundle_id,
                doc_id=stem,
                data_class="trusted_aux",
                task_family="tables",
                target_type="aux_annotation",
                canonical_target=normalize_target_text(structure_file.read_text(encoding="utf-8")),
                slice_tags=("table", "topology"),
                metadata={"structure_path": str(structure_file.relative_to(extract_root))},
                assets=(
                    CanonicalAsset(
                        name=f"image.{suffix}",
                        media_type=f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else 'png'}",
                        local_path=str(image_file),
                    ),
                ),
            )


class ManualManifestWorker(SourceWorker):
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        if context.manual_source_root is None:
            raise RuntimeError(f"Manual source root is required for {self.source_id}")
        source_root = context.manual_source_root / self.source_id
        manifest_candidates = [
            source_root / "samples.jsonl",
            source_root / "samples.parquet",
        ]
        manifest_path = next((item for item in manifest_candidates if item.exists()), None)
        if manifest_path is None:
            raise RuntimeError(
                f"Missing manual manifest for {self.source_id}; expected one of "
                f"{', '.join(str(item) for item in manifest_candidates)}"
            )
        if manifest_path.suffix == ".jsonl":
            with manifest_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        else:
            import pyarrow.parquet as pq

            rows = pq.read_table(manifest_path).to_pylist()
        for row in rows:
            assets: list[CanonicalAsset] = []
            for asset_path in row.get("asset_paths", []):
                asset_file = source_root / asset_path
                suffix = asset_file.suffix.lstrip(".") or "bin"
                media_type = (
                    "image/png"
                    if suffix == "png"
                    else "image/jpeg"
                    if suffix in {"jpg", "jpeg"}
                    else "application/octet-stream"
                )
                assets.append(
                    CanonicalAsset(
                        name=asset_file.name,
                        media_type=media_type,
                        local_path=str(asset_file),
                    )
                )
            yield CanonicalSample(
                sample_id=row["sample_id"],
                source_id=self.source_id,
                bundle_id=context.bundle_id,
                doc_id=row.get("doc_id") or row["sample_id"],
                page_id=row.get("page_id"),
                data_class=row.get("data_class", "trusted_aux"),
                task_family=row.get("task_family", "document_parsing"),
                target_type=row.get("target_type", "aux_annotation"),
                canonical_target=row["canonical_target"],
                split_assignment=row.get("split_assignment", "train"),
                difficulty_tier=row.get("difficulty_tier", "medium"),
                slice_tags=tuple(row.get("slice_tags", [])),
                metadata=row.get("metadata", {}),
                assets=tuple(assets),
                source_license=row.get("source_license"),
            )


def build_worker_registry() -> dict[str, SourceWorker]:
    registry: dict[str, SourceWorker] = {
        "arxiv_source_pdf": ArxivSourceWorker(),
        "pmc_oa_pdf_xml": PMCSourceWorker(),
        "publaynet": HFParquetWorker("publaynet", "jordanparker6/publaynet"),
        "doclaynet": DocLayNetWorker(),
        "pubtables_1m": PubTablesWorker(),
        "scitsr": SciTSRWorker(),
        "mathwriting": HFParquetWorker("mathwriting", "deepcopy/MathWriting-human"),
        "im2latex_100k": HFParquetWorker("im2latex_100k", "yuntian-deng/im2latex-100k"),
        "crohme": HFSplitAwareParquetWorker("crohme", "Neeze/CROHME-full"),
        "bentham": ArchivePairWorker(
            source_id="bentham",
            archives=(
                (
                    "BenthamDatasetR0-Images.tbz",
                    "https://zenodo.org/api/records/44519/files/BenthamDatasetR0-Images.tbz/content",
                ),
                (
                    "BenthamDatasetR0-GT.tbz",
                    "https://zenodo.org/api/records/44519/files/BenthamDatasetR0-GT.tbz/content",
                ),
            ),
            task_family="handwriting",
            target_type="page_markdown_projection",
            slice_tags=("handwriting", "historical", "manuscript"),
        ),
        "read_2016": ArchivePairWorker(
            source_id="read_2016",
            archives=(
                (
                    "PublicData.tgz",
                    "https://zenodo.org/api/records/218236/files/PublicData.tgz/content",
                ),
            ),
            task_family="handwriting",
            target_type="page_markdown_projection",
            slice_tags=("handwriting", "layout", "historical"),
            split_from_path=True,
            prefer_page_dir=True,
        ),
        "funsd": HFParquetWorker("funsd", "nielsr/funsd"),
        "cord": HFParquetWorker("cord", "naver-clova-ix/cord-v2"),
        "sroie": HFParquetWorker("sroie", "jsdnrs/ICDAR2019-SROIE"),
        "chartqa": HFParquetWorker("chartqa", "HuggingFaceM4/ChartQA"),
        "plotqa": HFParquetWorker("plotqa", "achang/plot_qa"),
        "fintabnet_family": HFParquetWorker(
            "fintabnet_family", "docling-project/FinTabNet_OTSL"
        ),
        "iam": HFSplitAwareParquetWorker("iam", "Teklia/IAM-line"),
        "unimer_1m": UniMERArchiveWorker(),
        "synthdog_european": SynthDoGEuropeanWorker(),
        "approved_exact_full_page_targets": ManualManifestWorker("approved_exact_full_page_targets"),
        "synthetic_from_exact": ManualManifestWorker("synthetic_from_exact"),
        "model_failures_plus_exact_truth": ManualManifestWorker("model_failures_plus_exact_truth"),
        "sft_and_repair_prompt_packs": ManualManifestWorker("sft_and_repair_prompt_packs"),
        "curated_non_overlap_holdouts": ManualManifestWorker("curated_non_overlap_holdouts"),
    }
    return registry
