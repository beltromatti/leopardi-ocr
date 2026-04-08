from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import shutil
import ssl
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


USER_AGENT = "leopardi-ocr-data-pipeline/0.1"


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
    with opener.open(_request(url), timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
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
        limit = context.source_limit(self.source_id, 2500)
        max_pages = context.page_limit(self.source_id, 8)
        from_date = "2018-01-01" if context.stage.target_param_budget_m <= 100 else "2012-01-01"
        source_root = context.raw_cache_dir / self.source_id
        source_root.mkdir(parents=True, exist_ok=True)
        for record in _iter_arxiv_oai_records(limit=limit, from_date=from_date):
            doc_id = record["id"].replace("/", "_")
            doc_root = source_root / doc_id
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


def _latest_pmc_filelists() -> list[str]:
    opener = _build_opener()
    url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
    with opener.open(_request(url), timeout=120) as response:
        html = response.read().decode("utf-8", "replace")
    baselines = re.findall(r'href="([^"]+baseline[^"]+filelist\.csv)"', html)
    incrementals = re.findall(r'href="([^"]+incr\.[^"]+filelist\.csv)"', html)
    return [f"{url}{name}" for name in sorted(baselines)] + [
        f"{url}{name}" for name in sorted(incrementals)[-30:]
    ]


def _iter_pmc_records(limit: int) -> Iterator[dict[str, str]]:
    yielded = 0
    opener = _build_opener()
    for filelist_url in _latest_pmc_filelists():
        with opener.open(_request(filelist_url), timeout=120) as response:
            payload = response.read().decode("utf-8", "replace").splitlines()
        reader = csv.DictReader(payload)
        for row in reader:
            if yielded >= limit:
                return
            pmcid_path = row["Article File"]
            pmcid = Path(pmcid_path).stem
            yielded += 1
            yield {
                "pmcid": pmcid,
                "citation": row.get("Article Citation", ""),
                "license": row.get("License", ""),
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
            package_path = doc_root / f"{pmcid}.tar.gz"
            extract_root = doc_root / "package"
            if not package_path.exists():
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
            xml_files = list(extract_root.rglob("*.xml"))
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


def _iter_hf_parquet_rows(repo_id: str) -> Iterator[dict[str, Any]]:
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    files = [item for item in list_repo_files(repo_id, repo_type="dataset") if item.endswith(".parquet")]
    for filename in sorted(files):
        parquet_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            for row in table.to_pylist():
                yield row


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
    return normalize_target_text(json.dumps(row, ensure_ascii=True, sort_keys=True))


def _hf_row_to_sample(
    *,
    source_id: str,
    bundle_id: str,
    row: dict[str, Any],
    row_index: int,
) -> CanonicalSample:
    image_asset = _extract_image_asset(row)
    doc_id = str(row.get("id") or row.get("image_id") or row.get("uid") or row_index)
    if source_id in {"mathwriting", "im2latex_100k"}:
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
            canonical_target=normalize_target_text(target),
            slice_tags=("formula", "aux"),
            metadata={key: value for key, value in row.items() if key != "image"},
            assets=(image_asset,) if image_asset else (),
        )
    if source_id in {"chartqa", "plotqa"}:
        question = str(row.get("question") or row.get("query") or "").strip()
        answer = str(row.get("answer") or row.get("label") or "").strip()
        target = normalize_target_text(f"Question: {question}\nAnswer: {answer}")
        return CanonicalSample(
            sample_id=f"{source_id}-{doc_id}",
            source_id=source_id,
            bundle_id=bundle_id,
            doc_id=doc_id,
            data_class="trusted_aux",
            task_family="graphics",
            target_type="chart_qa",
            canonical_target=target or _json_target(row),
            slice_tags=("graphics", "qa"),
            metadata={key: value for key, value in row.items() if key != "image"},
            assets=(image_asset,) if image_asset else (),
        )
    if source_id in {"funsd", "cord", "sroie"}:
        words = row.get("words")
        target = ""
        if isinstance(words, list):
            target = normalize_target_text(" ".join(str(item) for item in words))
        if not target:
            target = _json_target(row)
        return CanonicalSample(
            sample_id=f"{source_id}-{doc_id}",
            source_id=source_id,
            bundle_id=bundle_id,
            doc_id=doc_id,
            data_class="trusted_aux",
            task_family="forms",
            target_type="aux_annotation",
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
            target_type="aux_annotation",
            canonical_target=_json_target(row),
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
        for row_index, row in enumerate(_iter_hf_parquet_rows(self.repo_id)):
            if row_index >= limit:
                break
            yield _hf_row_to_sample(
                source_id=self.source_id,
                bundle_id=context.bundle_id,
                row=row,
                row_index=row_index,
            )


class DocLayNetWorker(SourceWorker):
    source_id = "doclaynet"
    zip_url = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"

    def iter_samples(self, context: SourceBuildContext) -> Iterator[CanonicalSample]:
        limit = context.source_limit(self.source_id, 15000)
        zip_path = context.raw_cache_dir / self.source_id / "DocLayNet_core.zip"
        if not zip_path.exists():
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
            )
        )
        annotation_tar = Path(
            hf_hub_download(
                repo_id="bsmock/pubtables-1m",
                repo_type="dataset",
                filename=filenames["annotations"],
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
        if not archive_path.exists():
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
        "funsd": HFParquetWorker("funsd", "nielsr/funsd"),
        "cord": HFParquetWorker("cord", "naver-clova-ix/cord-v2"),
        "sroie": HFParquetWorker("sroie", "jsdnrs/ICDAR2019-SROIE"),
        "chartqa": HFParquetWorker("chartqa", "HuggingFaceM4/ChartQA"),
        "plotqa": HFParquetWorker("plotqa", "achang/plot_qa"),
        "fintabnet_family": HFParquetWorker(
            "fintabnet_family", "docling-project/FinTabNet_OTSL"
        ),
        "crohme": ManualManifestWorker("crohme"),
        "iam": ManualManifestWorker("iam"),
        "bentham": ManualManifestWorker("bentham"),
        "read_2016": ManualManifestWorker("read_2016"),
        "approved_exact_full_page_targets": ManualManifestWorker("approved_exact_full_page_targets"),
        "synthetic_from_exact": ManualManifestWorker("synthetic_from_exact"),
        "model_failures_plus_exact_truth": ManualManifestWorker("model_failures_plus_exact_truth"),
        "sft_and_repair_prompt_packs": ManualManifestWorker("sft_and_repair_prompt_packs"),
        "curated_non_overlap_holdouts": ManualManifestWorker("curated_non_overlap_holdouts"),
    }
    return registry
