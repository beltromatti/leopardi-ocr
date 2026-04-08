from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Iterable

from leopardi.data_pipeline.schemas import CanonicalSample


def target_extension_for_type(target_type: str) -> str:
    return "md" if "markdown" in target_type else "txt"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class TarShardWriter:
    def __init__(self, output_dir: str | Path, shard_target_size_mb: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_target_bytes = max(shard_target_size_mb, 1) * 1024 * 1024
        self._current_shard_index = -1
        self._current_size = 0
        self._tar: tarfile.TarFile | None = None
        self.shard_paths: list[Path] = []

    def _open_next_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()
        self._current_shard_index += 1
        self._current_size = 0
        shard_path = self.output_dir / f"shard-{self._current_shard_index:06d}.tar"
        self.shard_paths.append(shard_path)
        self._tar = tarfile.open(shard_path, mode="w")

    def _write_member(self, arcname: str, payload: bytes) -> None:
        if self._tar is None:
            self._open_next_shard()
        assert self._tar is not None
        info = tarfile.TarInfo(name=arcname)
        info.size = len(payload)
        self._tar.addfile(info, io.BytesIO(payload))
        self._current_size += len(payload)

    def write_sample(self, sample: CanonicalSample) -> None:
        sample_payload = json.dumps(sample.manifest_record(), indent=2, sort_keys=True).encode("utf-8")
        estimated_size = len(sample_payload) + sum(asset.size_bytes() for asset in sample.assets)
        estimated_size += len(sample.canonical_target.encode("utf-8"))
        if self._tar is None or self._current_size + estimated_size > self.shard_target_bytes:
            self._open_next_shard()

        target_extension = target_extension_for_type(sample.target_type)
        self._write_member(f"{sample.sample_id}.sample.json", sample_payload)
        self._write_member(
            f"{sample.sample_id}.target.{target_extension}",
            sample.canonical_target.encode("utf-8"),
        )
        for asset in sample.assets:
            self._write_member(f"{sample.sample_id}.{asset.name}", asset.materialize_bytes())

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None


def write_manifest_parquet(records: Iterable[dict[str, object]], path: str | Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = list(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, target)
    return target


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_bundle_card(
    *,
    path: str | Path,
    bundle_id: str,
    source_ids: tuple[str, ...],
    sample_count: int,
    shard_paths: list[Path],
    manifest_path: Path,
    notes: list[str],
) -> Path:
    payload = {
        "bundle_id": bundle_id,
        "source_ids": list(source_ids),
        "sample_count": sample_count,
        "shards": [str(item) for item in shard_paths],
        "manifest_path": str(manifest_path),
        "notes": notes,
    }
    return write_json(path, payload)
