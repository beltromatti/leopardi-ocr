from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import shutil

from leopardi.data_pipeline.config import DataBuildStageConfig
from leopardi.data_pipeline.planner import build_data_build_execution_plan, plan_dict
from leopardi.data_pipeline.publish import publish_folder_to_hf
from leopardi.data_pipeline.schemas import BundleBuildStats, CanonicalSample
from leopardi.data_pipeline.storage import TarShardWriter, write_bundle_card, write_json, write_manifest_parquet
from leopardi.data_pipeline.workers import SourceBuildContext, build_worker_registry
from leopardi.ops import (
    ArtifactPointer,
    RunHeartbeat,
    RunManifest,
    RunSummary,
    append_event,
    ensure_run_layout,
    write_heartbeat,
    write_manifest,
    write_summary,
)


DEFAULT_SOURCE_LIMITS_S0 = {
    # arXiv / PMC are capped in source documents and yield page-level samples.
    # With the current page-projection rules this targets roughly:
    # - arXiv: ~2.0M projected pages
    # - PMC OA: ~1.2M projected pages
    "arxiv_source_pdf": 250000,
    "pmc_oa_pdf_xml": 150000,
    "publaynet": 300000,
    "doclaynet": 80863,
    "pubtables_1m": 250000,
    "scitsr": 15000,
    "fintabnet_family": 100000,
    "crohme": 10000,
    "mathwriting": 200000,
    "im2latex_100k": 100000,
    "unimer_1m": 1000000,
    "iam": 10373,
    "bentham": 5000,
    "read_2016": 5000,
    "funsd": 1000,
    "cord": 2000,
    "sroie": 2000,
    "chartqa": 15000,
    "plotqa": 15000,
    "synthdog_european": 500000,
}

DEFAULT_SOURCE_LIMITS_S1 = {
    # S1 keeps the same exact-first logic but opens the aperture materially.
    "arxiv_source_pdf": 800000,
    "pmc_oa_pdf_xml": 500000,
    "publaynet": 360000,
    "doclaynet": 80863,
    "pubtables_1m": 500000,
    "scitsr": 15000,
    "fintabnet_family": 112000,
    "crohme": 10000,
    "mathwriting": 230000,
    "im2latex_100k": 100000,
    "unimer_1m": 1000000,
    "iam": 10373,
    "bentham": 5000,
    "read_2016": 5000,
    "funsd": 1000,
    "cord": 2000,
    "sroie": 2000,
    "chartqa": 32719,
    "plotqa": 100000,
    "synthdog_european": 1500000,
}


DEFAULT_MAX_PAGES_S0 = {
    "arxiv_source_pdf": 12,
    "pmc_oa_pdf_xml": 12,
}


@dataclass(slots=True)
class DataBuildResult:
    experiment_id: str
    stage: str
    bundle_stats: tuple[BundleBuildStats, ...]
    plan_path: str
    publish_ledger_path: str


@dataclass(slots=True)
class _BundleRuntimeState:
    bundle_id: str
    source_ids: tuple[str, ...]
    bundle_class: str
    retention_mode: str
    bundle_dir: Path
    bundle_repo_root: Path
    repo_manifests_dir: Path
    manifests_dir: Path
    cards_dir: Path
    shard_writer: TarShardWriter
    manifest_rows: list[dict[str, object]]
    unique_docs: set[str]
    page_count: int = 0
    asset_bytes: int = 0
    remaining_sources: int = 0


def _write_publish_ledger(ledger_path: Path, bundle_stats: list[BundleBuildStats]) -> None:
    write_json(
        ledger_path,
        {
            "bundles": [asdict(item) for item in bundle_stats],
        },
    )


def _purge_source_cache(*, source_id: str, raw_cache_dir: Path, work_cache_dir: Path) -> None:
    for base_dir in (raw_cache_dir / source_id, work_cache_dir / source_id):
        if base_dir.exists():
            shutil.rmtree(base_dir)


def build_data_pipeline_stage(
    *,
    experiment_id: str,
    stage: DataBuildStageConfig,
    stage_config_path: str,
    runtime_config_path: str,
    root: str | Path = "runs",
    publish: bool = False,
    keep_raw: bool = False,
    manual_source_root: str | Path | None = None,
    source_limits: dict[str, int] | None = None,
) -> DataBuildResult:
    layout = ensure_run_layout(experiment_id, root=root)
    plan = build_data_build_execution_plan(
        experiment_id=experiment_id,
        stage=stage,
        stage_config_path=stage_config_path,
        runtime_config_path=runtime_config_path,
        root=root,
    )
    write_json(plan.plan_path, plan_dict(plan))
    write_manifest(
        RunManifest(
            experiment_id=experiment_id,
            phase="data_pipeline",
            stage=stage.stage,
            track=stage.profile_id,
            hardware_tag=stage.runtime.hardware_tag,
            config_paths=[stage_config_path, runtime_config_path],
            data_bundle_ids=list(plan.bundle_ids),
            protocol_version="data_pipeline_v1",
            local_run_root=str(layout.experiment_root),
            persistent_targets={
                "bundles": stage.runtime.persistence.bundle_target,
                "metadata": stage.runtime.persistence.metadata_target,
            },
        ),
        layout=layout,
    )
    worker_registry = build_worker_registry()
    default_source_limits = DEFAULT_SOURCE_LIMITS_BY_FAMILY.get(stage.target_model_family, {})
    effective_limits = {**default_source_limits, **(source_limits or {})}
    effective_max_pages = DEFAULT_MAX_PAGES_BY_FAMILY.get(stage.target_model_family, {})
    bundle_stats: list[BundleBuildStats] = []
    raw_cache_dir = Path(plan.local_paths.raw_cache_dir)
    work_cache_dir = Path(plan.local_paths.work_cache_dir)

    bundle_states: dict[str, _BundleRuntimeState] = {}
    source_to_bundle_ids: dict[str, list[str]] = {}
    for bundle_spec in plan.bundle_specs:
        bundle_dir = Path(bundle_spec.local_staging_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_repo_root = bundle_dir
        shards_dir = bundle_repo_root / "shards"
        repo_manifests_dir = bundle_repo_root / "manifests"
        manifests_dir = Path(bundle_spec.local_manifest_dir)
        cards_dir = bundle_repo_root / "cards"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        repo_manifests_dir.mkdir(parents=True, exist_ok=True)
        cards_dir.mkdir(parents=True, exist_ok=True)
        bundle_states[bundle_spec.bundle_id] = _BundleRuntimeState(
            bundle_id=bundle_spec.bundle_id,
            source_ids=bundle_spec.source_ids,
            bundle_class=bundle_spec.bundle_class,
            retention_mode=bundle_spec.retention_mode,
            bundle_dir=bundle_dir,
            bundle_repo_root=bundle_repo_root,
            repo_manifests_dir=repo_manifests_dir,
            manifests_dir=manifests_dir,
            cards_dir=cards_dir,
            shard_writer=TarShardWriter(shards_dir, stage.shard_target_size_mb),
            manifest_rows=[],
            unique_docs=set(),
            remaining_sources=len(bundle_spec.source_ids),
        )
        for source_id in bundle_spec.source_ids:
            source_to_bundle_ids.setdefault(source_id, []).append(bundle_spec.bundle_id)

    write_heartbeat(
        RunHeartbeat(
            experiment_id=experiment_id,
            phase="data_pipeline",
            stage=stage.stage,
            state="running",
            current_step=0,
            latest_metrics={},
        ),
        layout=layout,
    )

    def finalize_bundle(bundle_id: str, *, completion_index: int) -> None:
        bundle_spec = next(item for item in plan.bundle_specs if item.bundle_id == bundle_id)
        state = bundle_states[bundle_id]
        state.shard_writer.close()
        manifest_path = write_manifest_parquet(
            state.manifest_rows,
            state.manifests_dir / "samples.parquet",
        )
        shutil.copy2(manifest_path, state.repo_manifests_dir / "samples.parquet")
        stats = BundleBuildStats(
            bundle_id=bundle_id,
            sample_count=len(state.manifest_rows),
            document_count=len(state.unique_docs),
            page_count=state.page_count,
            asset_bytes=state.asset_bytes,
            shard_count=len(state.shard_writer.shard_paths),
        )
        write_bundle_card(
            path=state.cards_dir / "bundle-card.json",
            bundle_id=bundle_id,
            source_ids=state.source_ids,
            sample_count=stats.sample_count,
            shard_paths=state.shard_writer.shard_paths,
            manifest_path=manifest_path,
            notes=[
                f"bundle_class={state.bundle_class}",
                f"retention_mode={state.retention_mode}",
            ],
        )
        shutil.copy2(state.cards_dir / "bundle-card.json", state.manifests_dir / "bundle-card.json")
        if publish:
            bundle_publish = publish_folder_to_hf(
                local_folder=state.bundle_repo_root,
                hf_uri=bundle_spec.bundle_uri,
                num_workers=min(stage.runtime.io_workers, 8),
            )
            metadata_publish = publish_folder_to_hf(
                local_folder=state.manifests_dir,
                hf_uri=bundle_spec.manifest_uri,
                num_workers=min(stage.runtime.io_workers, 4),
            )
            stats.published = True
            stats.published_bundle_uri = bundle_publish.uri
            stats.published_manifest_uri = metadata_publish.uri

        bundle_stats.append(stats)
        _write_publish_ledger(Path(plan.local_paths.publish_ledger_path), bundle_stats)
        append_event(
            layout=layout,
            event_type="data_bundle_completed",
            phase="data_pipeline",
            stage=stage.stage,
            payload={"bundle_id": bundle_id, "sample_count": stats.sample_count},
        )
        write_heartbeat(
            RunHeartbeat(
                experiment_id=experiment_id,
                phase="data_pipeline",
                stage=stage.stage,
                state="running",
                current_step=completion_index,
                latest_metrics={
                    "bundle_index": float(completion_index),
                    "sample_count": float(stats.sample_count),
                },
            ),
            layout=layout,
        )
        if publish and not keep_raw and state.bundle_dir.exists():
            shutil.rmtree(state.bundle_dir)

    for source_index, source_id in enumerate(plan.source_ids, start=1):
        bundle_ids = tuple(source_to_bundle_ids.get(source_id, ()))
        if not bundle_ids:
            continue
        append_event(
            layout=layout,
            event_type="data_source_started",
            phase="data_pipeline",
            stage=stage.stage,
            payload={"source_id": source_id, "bundle_ids": list(bundle_ids)},
        )
        worker = worker_registry.get(source_id)
        if worker is None:
            raise RuntimeError(f"No data worker is registered for source {source_id}")
        primary_bundle_id = bundle_ids[0]
        primary_bundle = bundle_states[primary_bundle_id]
        source_context = SourceBuildContext(
            stage=stage,
            experiment_id=experiment_id,
            bundle_id=primary_bundle_id,
            bundle_class=primary_bundle.bundle_class,
            raw_cache_dir=raw_cache_dir,
            work_cache_dir=work_cache_dir,
            manual_source_root=Path(manual_source_root) if manual_source_root else None,
            render_pdf_pages=True,
            publish_enabled=publish,
            keep_raw=keep_raw,
                source_limits=effective_limits,
                max_pages_per_document=effective_max_pages,
            )
        for sample in worker.iter_samples(source_context):
            assert isinstance(sample, CanonicalSample)
            for target_bundle_id in bundle_ids:
                state = bundle_states[target_bundle_id]
                bundle_sample = replace(sample, bundle_id=target_bundle_id)
                state.shard_writer.write_sample(bundle_sample)
                state.manifest_rows.append(bundle_sample.manifest_record())
                state.unique_docs.add(bundle_sample.doc_id)
                if bundle_sample.page_id is not None:
                    state.page_count += 1
                state.asset_bytes += sum(asset.size_bytes() for asset in bundle_sample.assets)

        append_event(
            layout=layout,
            event_type="data_source_completed",
            phase="data_pipeline",
            stage=stage.stage,
            payload={"source_id": source_id, "bundle_ids": list(bundle_ids)},
        )
        if stage.raw_retention_mode == "publish_canonical_then_purge_raw" and not keep_raw:
            _purge_source_cache(
                source_id=source_id,
                raw_cache_dir=raw_cache_dir,
                work_cache_dir=work_cache_dir,
            )
        for target_bundle_id in bundle_ids:
            state = bundle_states[target_bundle_id]
            state.remaining_sources -= 1
            if state.remaining_sources == 0:
                finalize_bundle(target_bundle_id, completion_index=len(bundle_stats) + 1)

    summary = RunSummary(
        experiment_id=experiment_id,
        phase="data_pipeline",
        stage=stage.stage,
        outcome="completed",
        key_metrics={
            "bundle_count": float(len(bundle_stats)),
            "sample_count": float(sum(item.sample_count for item in bundle_stats)),
            "document_count": float(sum(item.document_count for item in bundle_stats)),
        },
        artifacts=[
            ArtifactPointer(
                artifact_kind="runtime_plan",
                uri=f"local://{plan.plan_path}",
                local_path=plan.plan_path,
                persistence_status="local_only",
            ),
            ArtifactPointer(
                artifact_kind="summary_table",
                uri=f"local://{plan.local_paths.publish_ledger_path}",
                local_path=plan.local_paths.publish_ledger_path,
                persistence_status="verified" if publish else "local_only",
            ),
        ],
        notes=[
            "Real data-pipeline build completed.",
            "Manual-only sources still require curated local manifests before the corresponding bundles can be built.",
        ],
    )
    write_summary(summary, layout=layout)
    write_heartbeat(
        RunHeartbeat(
            experiment_id=experiment_id,
            phase="data_pipeline",
            stage=stage.stage,
            state="completed",
            current_step=len(bundle_stats),
            latest_metrics=summary.key_metrics,
            last_sync_status="ok" if publish else "unknown",
        ),
        layout=layout,
    )
    return DataBuildResult(
        experiment_id=experiment_id,
        stage=stage.stage,
        bundle_stats=tuple(bundle_stats),
        plan_path=plan.plan_path,
        publish_ledger_path=plan.local_paths.publish_ledger_path,
    )
DEFAULT_SOURCE_LIMITS_BY_FAMILY = {
    "leopardi_s0": DEFAULT_SOURCE_LIMITS_S0,
    "leopardi_s1": DEFAULT_SOURCE_LIMITS_S1,
}

DEFAULT_MAX_PAGES_BY_FAMILY = {
    "leopardi_s0": DEFAULT_MAX_PAGES_S0,
    "leopardi_s1": DEFAULT_MAX_PAGES_S0,
}
