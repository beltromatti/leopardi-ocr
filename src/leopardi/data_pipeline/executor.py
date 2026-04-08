from __future__ import annotations

from dataclasses import asdict, dataclass
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
    "arxiv_source_pdf": 2500,
    "pmc_oa_pdf_xml": 2000,
    "publaynet": 50000,
    "doclaynet": 15000,
    "pubtables_1m": 40000,
    "scitsr": 15000,
    "mathwriting": 30000,
    "im2latex_100k": 30000,
    "funsd": 1000,
    "cord": 2000,
    "chartqa": 12000,
    "plotqa": 12000,
}


DEFAULT_MAX_PAGES_S0 = {
    "arxiv_source_pdf": 8,
    "pmc_oa_pdf_xml": 8,
}


@dataclass(slots=True)
class DataBuildResult:
    experiment_id: str
    stage: str
    bundle_stats: tuple[BundleBuildStats, ...]
    plan_path: str
    publish_ledger_path: str


def _bundle_upload_root(bundle_dir: Path, bundle_id: str) -> Path:
    root = bundle_dir / "publish-staging"
    target = root / bundle_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return root


def _write_publish_ledger(ledger_path: Path, bundle_stats: list[BundleBuildStats]) -> None:
    write_json(
        ledger_path,
        {
            "bundles": [asdict(item) for item in bundle_stats],
        },
    )


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
    default_source_limits = DEFAULT_SOURCE_LIMITS_S0 if stage.target_param_budget_m <= 100 else {}
    effective_limits = {**default_source_limits, **(source_limits or {})}
    bundle_stats: list[BundleBuildStats] = []

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

    for bundle_index, bundle_spec in enumerate(plan.bundle_specs, start=1):
        append_event(
            layout=layout,
            event_type="data_bundle_started",
            phase="data_pipeline",
            stage=stage.stage,
            payload={"bundle_id": bundle_spec.bundle_id, "source_ids": list(bundle_spec.source_ids)},
        )
        bundle_dir = Path(bundle_spec.local_staging_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = bundle_dir / "shards"
        manifests_dir = Path(bundle_spec.local_manifest_dir)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        shard_writer = TarShardWriter(shards_dir, stage.shard_target_size_mb)
        manifest_rows: list[dict[str, object]] = []
        unique_docs: set[str] = set()
        page_count = 0
        asset_bytes = 0

        for source_id in bundle_spec.source_ids:
            worker = worker_registry.get(source_id)
            if worker is None:
                raise RuntimeError(f"No data worker is registered for source {source_id}")
            source_context = SourceBuildContext(
                stage=stage,
                experiment_id=experiment_id,
                bundle_id=bundle_spec.bundle_id,
                bundle_class=bundle_spec.bundle_class,
                raw_cache_dir=Path(plan.local_paths.raw_cache_dir),
                work_cache_dir=Path(plan.local_paths.work_cache_dir),
                manual_source_root=Path(manual_source_root) if manual_source_root else None,
                render_pdf_pages=True,
                publish_enabled=publish,
                keep_raw=keep_raw,
                source_limits=effective_limits,
                max_pages_per_document=DEFAULT_MAX_PAGES_S0,
            )
            for sample in worker.iter_samples(source_context):
                assert isinstance(sample, CanonicalSample)
                shard_writer.write_sample(sample)
                manifest_rows.append(sample.manifest_record())
                unique_docs.add(sample.doc_id)
                if sample.page_id is not None:
                    page_count += 1
                asset_bytes += sum(asset.size_bytes() for asset in sample.assets)

        shard_writer.close()
        manifest_path = write_manifest_parquet(
            manifest_rows,
            manifests_dir / "samples.parquet",
        )
        stats = BundleBuildStats(
            bundle_id=bundle_spec.bundle_id,
            sample_count=len(manifest_rows),
            document_count=len(unique_docs),
            page_count=page_count,
            asset_bytes=asset_bytes,
            shard_count=len(shard_writer.shard_paths),
        )
        write_bundle_card(
            path=manifests_dir / "bundle-card.json",
            bundle_id=bundle_spec.bundle_id,
            source_ids=bundle_spec.source_ids,
            sample_count=stats.sample_count,
            shard_paths=shard_writer.shard_paths,
            manifest_path=manifest_path,
            notes=[
                f"bundle_class={bundle_spec.bundle_class}",
                f"retention_mode={bundle_spec.retention_mode}",
            ],
        )
        if publish:
            upload_root = _bundle_upload_root(bundle_dir, bundle_spec.bundle_id)
            target_dir = upload_root / bundle_spec.bundle_id
            shutil.copytree(shards_dir, target_dir / "shards", dirs_exist_ok=True)
            shutil.copy2(manifest_path, target_dir / "manifests" / "samples.parquet")
            shutil.copy2(manifests_dir / "bundle-card.json", target_dir / "cards" / "bundle-card.json")

            bundle_publish = publish_folder_to_hf(
                local_folder=upload_root,
                hf_uri=bundle_spec.bundle_uri,
                num_workers=min(stage.runtime.io_workers, 8),
            )
            metadata_publish = publish_folder_to_hf(
                local_folder=manifests_dir,
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
            payload={"bundle_id": bundle_spec.bundle_id, "sample_count": stats.sample_count},
        )
        write_heartbeat(
            RunHeartbeat(
                experiment_id=experiment_id,
                phase="data_pipeline",
                stage=stage.stage,
                state="running",
                current_step=bundle_index,
                latest_metrics={
                    "bundle_index": float(bundle_index),
                    "sample_count": float(stats.sample_count),
                },
            ),
            layout=layout,
        )

    if stage.raw_retention_mode == "publish_canonical_then_purge_raw" and publish and not keep_raw:
        raw_cache = Path(plan.local_paths.raw_cache_dir)
        if raw_cache.exists():
            shutil.rmtree(raw_cache)

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
