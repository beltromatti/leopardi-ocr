from __future__ import annotations

import base64
import json
from pathlib import Path
import tarfile

import pyarrow.parquet as pq

from leopardi.data_pipeline.canonicalize import (
    jats_to_markdown,
    project_markdown_to_pages,
    tex_to_markdown,
)
from leopardi.data_pipeline.config import DataBuildStageConfig
from leopardi.data_pipeline.executor import build_data_pipeline_stage
from leopardi.data_pipeline.publish import parse_hf_uri
from leopardi.data_pipeline.schemas import CanonicalSample
from leopardi.data_pipeline.workers import HFParquetWorker, ManualManifestWorker, build_worker_registry


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2p6H8AAAAASUVORK5CYII="
)


def _manual_stage() -> DataBuildStageConfig:
    return DataBuildStageConfig.from_dict(
        {
            "data_build": {
                "stage": "manual_handwriting_build",
                "profile_id": "handwriting_only",
                "bundle_ids": ["p3_hardcases_v1"],
                "source_ids": ["iam"],
                "target_model_family": "leopardi_s0",
                "target_param_budget_m": 100,
                "strict_disk_guard": True,
                "shard_target_size_mb": 1,
                "raw_retention_mode": "publish_canonical_then_purge_raw",
            }
        },
        {
            "runtime": {
                "hardware_tag": "cpu-dev",
                "cpu_workers": 2,
                "io_workers": 2,
                "local_disk_budget_gb": 10,
                "max_active_sources": 1,
            }
        },
    )


def test_tex_and_jats_canonicalizers() -> None:
    tex = r"""
    \documentclass{article}
    \begin{document}
    \section{Intro}
    Text with \textbf{bold} and $x+y$.
    \begin{equation}
    a=b
    \end{equation}
    \end{document}
    """
    markdown = tex_to_markdown(tex)
    assert "## Intro" in markdown
    assert "Text with bold" in markdown
    assert "$$" in markdown

    jats = """
    <article>
      <front><article-meta><title-group><article-title>Title</article-title></title-group></article-meta></front>
      <body><sec><title>Methods</title><p>Body text.</p></sec></body>
    </article>
    """
    assert "# Title" in jats_to_markdown(jats)
    assert "## Methods" in jats_to_markdown(jats)


def test_tex_table_and_figure_are_preserved() -> None:
    tex = r"""
    \documentclass{article}
    \begin{document}
    \begin{figure}
    \caption{Architecture overview}
    \end{figure}
    \begin{table}
    \caption{Metrics}
    \begin{tabular}{ll}
    Name & Value \\
    Accuracy & 99 \\
    Speed & Fast \\
    \end{tabular}
    \end{table}
    \end{document}
    """
    markdown = tex_to_markdown(tex)
    assert "Figure: Architecture overview" in markdown
    assert "Table: Metrics" in markdown
    assert "| Name | Value |" in markdown
    assert "| Accuracy | 99 |" in markdown


def test_jats_table_and_inline_formula_are_preserved() -> None:
    jats = """
    <article>
      <front>
        <article-meta>
          <title-group><article-title>Doc</article-title></title-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Results</title>
          <p>We solve <inline-formula><tex-math>x+y</tex-math></inline-formula> exactly.</p>
          <table-wrap>
            <caption><title>Scores</title></caption>
            <table>
              <tbody>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>99</td></tr>
              </tbody>
            </table>
          </table-wrap>
        </sec>
      </body>
    </article>
    """
    markdown = jats_to_markdown(jats)
    assert "$x+y$" in markdown
    assert "Table: Scores" in markdown
    assert "| Metric | Value |" in markdown


def test_page_projection_preserves_order() -> None:
    markdown = "## A\n\nalpha beta gamma\n\n## B\n\ndelta epsilon zeta"
    pages = ["alpha beta gamma", "delta epsilon zeta"]
    projected = project_markdown_to_pages(markdown, pages)
    assert len(projected) == 2
    assert "alpha beta gamma" in projected[0]
    assert "delta epsilon zeta" in projected[1]


def test_parse_hf_uri() -> None:
    assert parse_hf_uri("hf://leopardi-ocr-data-bundles/p2_exact_core_v1") == (
        "leopardi-ocr-data-bundles",
        "p2_exact_core_v1",
    )


def test_data_pipeline_build_manual_source(tmp_path: Path) -> None:
    manual_root = tmp_path / "manual"
    source_root = manual_root / "iam"
    source_root.mkdir(parents=True)
    image_path = source_root / "line.png"
    image_path.write_bytes(_PNG_1X1)
    manifest_path = source_root / "samples.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "iam-0001",
                "doc_id": "iam-doc-1",
                "page_id": "iam-doc-1:1",
                "data_class": "trusted_aux",
                "task_family": "handwriting",
                "target_type": "text_line",
                "canonical_target": "sample handwriting target",
                "asset_paths": ["line.png"],
                "slice_tags": ["handwriting", "manual"],
                "metadata": {"writer_id": "w1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_data_pipeline_stage(
        experiment_id="leo-data-manual-test",
        stage=_manual_stage(),
        stage_config_path="generated::manual-stage",
        runtime_config_path="generated::manual-runtime",
        root=tmp_path / "runs",
        publish=False,
        manual_source_root=manual_root,
    )

    assert result.stage == "manual_handwriting_build"
    assert result.bundle_stats[0].sample_count == 1

    manifest = (
        tmp_path
        / "runs"
        / "leo-data-manual-test"
        / "artifacts"
        / "data_pipeline"
        / "manual_handwriting_build"
        / "manifests"
        / "p3_hardcases_v1"
        / "samples.parquet"
    )
    assert manifest.exists()
    rows = pq.read_table(manifest).to_pylist()
    assert rows[0]["sample_id"] == "iam-0001"

    shard_path = (
        tmp_path
        / "runs"
        / "leo-data-manual-test"
        / "scratch"
        / "data_pipeline"
        / "manual_handwriting_build"
        / "bundles"
        / "p3_hardcases_v1"
        / "shards"
        / "shard-000000.tar"
    )
    assert shard_path.exists()
    with tarfile.open(shard_path, "r") as archive:
        names = archive.getnames()
    assert "iam-0001.sample.json" in names
    assert "iam-0001.target.txt" in names


def test_worker_registry_promotions_for_verified_sources() -> None:
    registry = build_worker_registry()
    assert isinstance(registry["sroie"], HFParquetWorker)
    assert isinstance(registry["fintabnet_family"], HFParquetWorker)
    assert isinstance(registry["iam"], ManualManifestWorker)
    assert isinstance(registry["bentham"], ManualManifestWorker)
    assert isinstance(registry["read_2016"], ManualManifestWorker)
    assert isinstance(registry["crohme"], ManualManifestWorker)


def test_manifest_record_infers_canonical_features() -> None:
    sample = CanonicalSample(
        sample_id="sample-1",
        source_id="unit",
        bundle_id="bundle",
        data_class="gold_exact",
        task_family="document_parsing",
        target_type="page_markdown_projection",
        canonical_target="# Title\n\nFigure: Example\n\n| A | B |\n| :--- | :--- |\n| 1 | 2 |\n\n$x+y$\n\n```table\ncolumns: 2\ncells:\n  - [0, 0, 0, 1, \"X\"]\n```",
        doc_id="doc-1",
    )
    record = sample.manifest_record()
    assert record["has_heading"] is True
    assert record["has_figure_caption"] is True
    assert record["has_gfm_table"] is True
    assert record["has_table_block"] is True
    assert record["has_inline_math"] is True
