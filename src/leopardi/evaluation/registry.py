from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetRegistryEntry:
    dataset_family: str
    scope: str
    primary_role: str
    unit: str
    public_status: str
    core_strength: str
    main_weakness: str
    default_split_policy: str
    covered_protocols: tuple[str, ...]


@dataclass(slots=True)
class BaselineRegistryEntry:
    baseline_id: str
    model_name: str
    group: str
    open_status: str
    primary_role: str
    size_band: str
    main_protocols: tuple[str, ...]
    evidence_policy: str
    notes: str


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dataset_registry(
    path: str | Path = "evaluation/datasets/registry.csv",
) -> list[DatasetRegistryEntry]:
    rows = _read_csv(path)
    return [
        DatasetRegistryEntry(
            dataset_family=row["dataset_family"],
            scope=row["scope"],
            primary_role=row["primary_role"],
            unit=row["unit"],
            public_status=row["public_status"],
            core_strength=row["core_strength"],
            main_weakness=row["main_weakness"],
            default_split_policy=row["default_split_policy"],
            covered_protocols=tuple(item for item in row["covered_protocols"].split("|") if item),
        )
        for row in rows
    ]


def load_baseline_registry(
    path: str | Path = "evaluation/baselines/registry.csv",
) -> list[BaselineRegistryEntry]:
    rows = _read_csv(path)
    return [
        BaselineRegistryEntry(
            baseline_id=row["baseline_id"],
            model_name=row["model_name"],
            group=row["group"],
            open_status=row["open_status"],
            primary_role=row["primary_role"],
            size_band=row["size_band"],
            main_protocols=tuple(item for item in row["main_protocols"].split("|") if item),
            evidence_policy=row["evidence_policy"],
            notes=row["notes"],
        )
        for row in rows
    ]


def registry_summary() -> dict[str, object]:
    datasets = load_dataset_registry()
    baselines = load_baseline_registry()
    return {
        "dataset_count": len(datasets),
        "baseline_count": len(baselines),
        "public_dataset_families": sorted(item.dataset_family for item in datasets if item.public_status == "public"),
        "frontier_baselines": sorted(item.model_name for item in baselines if item.group.startswith("frontier")),
    }
