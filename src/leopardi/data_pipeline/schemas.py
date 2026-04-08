from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def _infer_canonical_features(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    has_display_math = "$$" in text
    has_inline_math = bool(re.search(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", text, flags=re.DOTALL))
    has_gfm_table = any(line.strip().startswith("|") and line.strip().endswith("|") for line in lines)
    has_table_block = "```table" in text
    has_figure_caption = bool(re.search(r"(?im)^figure:\s+", text))
    has_heading = bool(re.search(r"(?m)^#{1,6}\s", text))
    has_list = bool(re.search(r"(?m)^\s*[-*]\s", text))
    return {
        "target_char_count": len(text),
        "has_display_math": has_display_math,
        "has_inline_math": has_inline_math,
        "has_gfm_table": has_gfm_table,
        "has_table_block": has_table_block,
        "has_figure_caption": has_figure_caption,
        "has_heading": has_heading,
        "has_list": has_list,
    }


@dataclass(slots=True)
class CanonicalAsset:
    name: str
    media_type: str
    payload_bytes: bytes | None = None
    text_payload: str | None = None
    local_path: str | None = None

    def materialize_bytes(self) -> bytes:
        if self.payload_bytes is not None:
            return self.payload_bytes
        if self.text_payload is not None:
            return self.text_payload.encode("utf-8")
        if self.local_path is not None:
            return Path(self.local_path).read_bytes()
        raise ValueError(f"Asset {self.name} has no payload")

    def size_bytes(self) -> int:
        if self.payload_bytes is not None:
            return len(self.payload_bytes)
        if self.text_payload is not None:
            return len(self.text_payload.encode("utf-8"))
        if self.local_path is not None:
            return Path(self.local_path).stat().st_size
        return 0


@dataclass(slots=True)
class CanonicalSample:
    sample_id: str
    source_id: str
    bundle_id: str
    data_class: str
    task_family: str
    target_type: str
    canonical_target: str
    doc_id: str
    page_id: str | None = None
    split_assignment: str = "train"
    difficulty_tier: str = "medium"
    slice_tags: tuple[str, ...] = ()
    transform_recipe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    assets: tuple[CanonicalAsset, ...] = ()
    source_license: str | None = None

    @property
    def canonical_target_hash(self) -> str:
        return hashlib.sha256(self.canonical_target.encode("utf-8")).hexdigest()

    def manifest_record(self) -> dict[str, Any]:
        features = _infer_canonical_features(self.canonical_target)
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "bundle_id": self.bundle_id,
            "doc_id": self.doc_id,
            "page_id": self.page_id,
            "data_class": self.data_class,
            "task_family": self.task_family,
            "target_type": self.target_type,
            "canonical_target": self.canonical_target,
            "canonical_target_hash": self.canonical_target_hash,
            "difficulty_tier": self.difficulty_tier,
            "slice_tags": list(self.slice_tags),
            "split_assignment": self.split_assignment,
            "transform_recipe": self.transform_recipe,
            "source_license": self.source_license,
            **features,
            "asset_names": [asset.name for asset in self.assets],
            "asset_media_types": [asset.media_type for asset in self.assets],
            "metadata_json": json.dumps(self.metadata, sort_keys=True),
        }


@dataclass(slots=True)
class BuildSourceSelection:
    source_id: str
    max_documents: int
    max_pages_per_document: int
    max_raw_bytes: int
    estimated_documents: int | None = None


@dataclass(slots=True)
class BundleBuildStats:
    bundle_id: str
    sample_count: int = 0
    document_count: int = 0
    page_count: int = 0
    asset_bytes: int = 0
    shard_count: int = 0
    published: bool = False
    published_bundle_uri: str | None = None
    published_manifest_uri: str | None = None


@dataclass(slots=True)
class PublishResult:
    uri: str
    repo_id: str
    verified_file_count: int
