from __future__ import annotations

import base64
import io
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
from leopardi.data_pipeline.storage import TarShardWriter
from leopardi.data_pipeline.workers import (
    ArchivePairWorker,
    HFParquetWorker,
    HFSplitAwareParquetWorker,
    SynthDoGEuropeanWorker,
    SourceBuildContext,
    UniMERArchiveWorker,
    _is_valid_cached_download,
    _infer_hf_split_from_filename,
    _hf_row_to_sample,
    _pagexml_to_markdown,
    build_worker_registry,
)


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2p6H8AAAAASUVORK5CYII="
)


def _manual_stage() -> DataBuildStageConfig:
    return DataBuildStageConfig.from_dict(
        {
            "data_build": {
                "stage": "manual_handwriting_build",
                "profile_id": "full_frontier",
                "bundle_ids": ["sft_core_v1"],
                "source_ids": ["approved_exact_full_page_targets"],
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


def _write_tar_archive(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:bz2" if path.suffix == ".tbz" else "w:gz"
    with tarfile.open(path, mode) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, fileobj=io.BytesIO(payload))


def _write_zip_archive(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


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


def test_tex_front_matter_is_cleaned_and_preserved() -> None:
    tex = r"""
    \documentclass{article}
    \usepackage{hyperref}
    \definecolor{pastelBlue}{rgb}{0.0,0.4,0.7}
    \newcommand{\algname}{RandGraph}
    \begin{document}
    \TITLE{Generating Random Networks Without Short Cycles}
    \ARTICLEAUTHORS{
      \AUTHOR{Mohsen Bayati}
      \AFF{Stanford University}
    }
    \ABSTRACT{We propose $\algname$.}
    \section{Intro}
    Our method is called \algname.
    \end{document}
    """
    markdown = tex_to_markdown(tex)
    assert markdown.startswith("# Generating Random Networks Without Short Cycles")
    assert "Mohsen Bayati" in markdown
    assert "## Abstract" in markdown
    assert "RandGraph" in markdown
    assert "definecolor" not in markdown


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


def test_page_projection_skips_noisy_page_headers() -> None:
    markdown = (
        "# Title\n\n"
        "intro words alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n\n"
        "## Section\n\n"
        "continued discussion with more content and a final theorem statement here"
    )
    pages = [
        "1 University header and author block\nintro words alpha beta gamma delta epsilon zeta eta theta iota",
        "2 footer noise and page number\ncontinued discussion with more content and a final theorem statement here",
    ]
    projected = project_markdown_to_pages(markdown, pages)
    assert "intro words alpha beta gamma" in projected[0]
    assert "discussion with more content and a final theorem statement here" in projected[1]


def test_parse_hf_uri() -> None:
    assert parse_hf_uri("hf://leopardi-ocr-data-bundles/p2_exact_core_v1") == (
        "leopardi-ocr-data-bundles",
        "p2_exact_core_v1",
    )


def test_data_pipeline_build_manual_source(tmp_path: Path) -> None:
    manual_root = tmp_path / "manual"
    source_root = manual_root / "approved_exact_full_page_targets"
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
        / "sft_core_v1"
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
        / "sft_core_v1"
        / "shards"
        / "shard-000000.tar"
    )
    assert shard_path.exists()
    with tarfile.open(shard_path, "r") as archive:
        names = archive.getnames()
    assert "iam-0001.sample.json" in names
    assert "iam-0001.target.txt" in names


def test_tar_shard_writer_uses_markdown_extension_for_page_markdown_projection(tmp_path: Path) -> None:
    writer = TarShardWriter(tmp_path / "shards", shard_target_size_mb=1)
    writer.write_sample(
        CanonicalSample(
            sample_id="page-1",
            source_id="unit",
            bundle_id="bundle",
            data_class="gold_exact",
            task_family="document_parsing",
            target_type="page_markdown_projection",
            canonical_target="# Title",
            doc_id="doc-1",
        )
    )
    writer.close()
    with tarfile.open(tmp_path / "shards" / "shard-000000.tar", "r") as archive:
        names = archive.getnames()
    assert "page-1.target.md" in names


def test_worker_registry_promotions_for_verified_sources() -> None:
    registry = build_worker_registry()
    assert isinstance(registry["sroie"], HFParquetWorker)
    assert isinstance(registry["fintabnet_family"], HFParquetWorker)
    assert isinstance(registry["crohme"], HFSplitAwareParquetWorker)
    assert isinstance(registry["bentham"], ArchivePairWorker)
    assert isinstance(registry["read_2016"], ArchivePairWorker)
    assert isinstance(registry["iam"], HFSplitAwareParquetWorker)
    assert isinstance(registry["unimer_1m"], UniMERArchiveWorker)
    assert isinstance(registry["synthdog_european"], SynthDoGEuropeanWorker)


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


def test_hf_row_to_sample_sanitizes_bytes_and_emits_structured_targets() -> None:
    fintabnet_row = {
        "filename": "demo.png",
        "cells": [
            [
                {"tokens": list("Metric")},
                {"tokens": list("Value")},
                {"tokens": list("Accuracy")},
                {"tokens": list("99")},
            ]
        ],
        "cols": 2,
        "image": {"bytes": _PNG_1X1, "path": "demo.png"},
    }
    fintabnet = _hf_row_to_sample(
        source_id="fintabnet_family",
        bundle_id="preview",
        row=fintabnet_row,
        row_index=0,
    )
    assert fintabnet.target_type == "table_markdown"
    assert "| Metric | Value |" in fintabnet.canonical_target

    sroie_row = {
        "key": "receipt-1",
        "entities": {"company": "ACME", "total": "12.00"},
        "words": ["ACME", "TOTAL", "12.00"],
        "bboxes": [[0, 0, 10, 10], [0, 20, 20, 30], [25, 20, 40, 30]],
        "image": {"bytes": _PNG_1X1, "path": "receipt.jpg"},
    }
    sroie = _hf_row_to_sample(
        source_id="sroie",
        bundle_id="preview",
        row=sroie_row,
        row_index=0,
    )
    assert sroie.target_type == "page_markdown_projection"
    assert "## Receipt Fields" in sroie.canonical_target
    assert "## Receipt OCR" in sroie.canonical_target

    funsd_row = {
        "words": ["Total", "Amount", "42.00"],
        "bboxes": [[0, 0, 10, 10], [12, 0, 30, 10], [35, 0, 50, 10]],
        "ner_tags": [3, 4, 6],
        "image": {"bytes": _PNG_1X1, "path": "form.png"},
    }
    funsd = _hf_row_to_sample(
        source_id="funsd",
        bundle_id="preview",
        row=funsd_row,
        row_index=0,
    )
    assert funsd.target_type == "page_markdown_projection"
    assert "## Form Fields" in funsd.canonical_target
    assert "- question: Total Amount" in funsd.canonical_target
    assert "- answer: 42.00" in funsd.canonical_target
    assert "## Form OCR" in funsd.canonical_target

    plotqa_row = {
        "text": "<s_y>1<sep/>2</s_y><s_x>a<sep/>b</s_x><s_bboxes><s_y>10</s_y><s_x>11</s_x><s_w>12</s_w><s_h>13</s_h><sep/><s_y>20</s_y><s_x>21</s_x><s_w>22</s_w><s_h>23</s_h></s_bboxes>",
        "image": {"bytes": _PNG_1X1, "path": "plot.png"},
    }
    plotqa = _hf_row_to_sample(
        source_id="plotqa",
        bundle_id="preview",
        row=plotqa_row,
        row_index=0,
    )
    assert plotqa.target_type == "chart_serialization"
    assert "```chart" in plotqa.canonical_target
    assert "(x=11, y=10, w=12, h=13)" in plotqa.canonical_target

    cord_row = {
        "ground_truth": json.dumps(
            {
                "gt_parse": {"total": {"total_price": "60.000"}},
                "valid_line": [
                    {
                        "category": "menu.num",
                        "group_id": 2,
                        "sub_group_id": 0,
                        "words": [{"text": "901016"}],
                    },
                    {
                        "category": "menu.nm",
                        "group_id": 1,
                        "sub_group_id": 0,
                        "words": [{"text": "-TICKET"}, {"text": "CP"}],
                    },
                ],
            }
        ),
        "image": {"bytes": _PNG_1X1, "path": "receipt.png"},
    }
    cord = _hf_row_to_sample(
        source_id="cord",
        bundle_id="preview",
        row=cord_row,
        row_index=0,
    )
    assert cord.target_type == "page_markdown_projection"
    assert "## Receipt Fields" in cord.canonical_target
    assert "## Receipt OCR" in cord.canonical_target
    assert "-TICKET CP" in cord.canonical_target
    assert "901016" in cord.canonical_target

    crohme_row = {
        "label": r"\frac{a}{b}",
        "image": {"bytes": _PNG_1X1, "path": "formula.png"},
    }
    crohme = _hf_row_to_sample(
        source_id="crohme",
        bundle_id="preview",
        row=crohme_row,
        row_index=0,
    )
    assert crohme.target_type == "latex_formula"
    assert crohme.canonical_target == "$$\n\\frac{a}{b}\n$$"

    unimer_row = {
        "label": r"\int_0^1 x^2 \\, dx",
        "image": {"bytes": _PNG_1X1, "path": "unimer.png"},
    }
    unimer = _hf_row_to_sample(
        source_id="unimer_1m",
        bundle_id="preview",
        row=unimer_row,
        row_index=0,
    )
    assert unimer.target_type == "latex_formula"
    assert "int_0^1" in unimer.canonical_target
    assert "large_scale" in unimer.slice_tags

    iam_row = {
        "text": "put down a resolution on the subject",
        "image": {"bytes": _PNG_1X1, "path": "line.jpg"},
    }
    iam = _hf_row_to_sample(
        source_id="iam",
        bundle_id="preview",
        row=iam_row,
        row_index=0,
    )
    assert iam.target_type == "text_line"
    assert iam.task_family == "handwriting"
    assert iam.canonical_target == "put down a resolution on the subject"


def test_unimer_archive_worker_pairs_zip_samples(tmp_path: Path) -> None:
    archive_path = tmp_path / "UniMER-1M.zip"
    _write_zip_archive(
        archive_path,
        {
            "images/sample-1.png": _PNG_1X1,
            "labels/sample-1.txt": b"\\frac{a}{b}",
        },
    )
    worker = UniMERArchiveWorker()
    worker._archive_path = lambda context: archive_path  # type: ignore[method-assign]
    context = SourceBuildContext(
        stage=_manual_stage(),
        experiment_id="unit",
        bundle_id="preview",
        bundle_class="structural_aux",
        raw_cache_dir=tmp_path / "raw",
        work_cache_dir=tmp_path / "work",
    )
    sample = next(worker.iter_samples(context))
    assert sample.source_id == "unimer_1m"
    assert sample.target_type == "latex_formula"
    assert sample.assets[0].materialize_bytes() == _PNG_1X1


def test_synthdog_worker_emits_page_markdown(monkeypatch) -> None:
    from leopardi.data_pipeline import workers as worker_module
    from leopardi.data_pipeline.synthdog import SynthDoGSample

    monkeypatch.setattr(
        worker_module,
        "iter_synthdog_european_samples",
        lambda total_limit: [
            SynthDoGSample(
                sample_id="synthdog-eu-000001",
                doc_id="de-wiki-000001",
                language="de",
                title="Titel",
                canonical_target="# Titel\n\nAbsatz",
                image_png=_PNG_1X1,
            )
        ],
    )
    worker = SynthDoGEuropeanWorker()
    context = SourceBuildContext(
        stage=_manual_stage(),
        experiment_id="unit",
        bundle_id="preview",
        bundle_class="multimodal_exact",
        raw_cache_dir=Path("tmp/raw"),
        work_cache_dir=Path("tmp/work"),
    )
    sample = next(worker.iter_samples(context))
    assert sample.target_type == "page_markdown_projection"
    assert sample.task_family == "document_parsing"
    assert "multilingual" in sample.slice_tags


def test_tex_canonicalizer_preserves_spaces_around_inline_math() -> None:
    tex = r"""
    \documentclass{article}
    \def\algname{{\sf RandGraph}}
    \begin{document}
    \ABSTRACT{For any constant $k$, $\algname$ generates a graph with $n$ vertices, $m$ edges, and no cycle of length at most $k$.}
    \end{document}
    """
    markdown = tex_to_markdown(tex)
    assert "$RandGraph$ generates" in markdown
    assert "$n$ vertices, $m$ edges" in markdown
    assert "at most $k$." in markdown


def test_tex_canonicalizer_trims_inline_math_punctuation_spacing() -> None:
    tex = r"""
    \documentclass{article}
    \begin{document}
    \ABSTRACT{We have $m=O(n^{1+1/[2k(k+3)]}\, )$.}
    \end{document}
    """
    markdown = tex_to_markdown(tex)
    assert r"$m=O(n^{1+1/[2k(k+3)]}\,)$" in markdown


def test_tex_canonicalizer_preserves_complex_math_commands() -> None:
    tex = r"""
    \documentclass{article}
    \begin{document}
    \section{Core}
    Here is inline math $\P_{\alg}(G)=\frac{1}{Z(G_t)}$ and
    \[
    E_k(G_t,ij)\equiv\sum_{r=3}^k\sum_{\ell=0}^{r-2}N_{r,\ell}^{G_t,ij}q_t^{r-1-\ell}\,.
    \]
    \end{document}
    """
    markdown = tex_to_markdown(tex)
    assert r"$\P_{\alg}(G)=\frac{1}{Z(G_t)}$" in markdown
    assert r"\sum_{r=3}^k" in markdown
    assert r"N_{r,\ell}^{G_t,ij}" in markdown


def test_clean_tex_text_preserves_nxml_payloads(tmp_path: Path) -> None:
    import fitz

    stage = _manual_stage()
    context = SourceBuildContext(
        stage=stage,
        experiment_id="pmc-test",
        bundle_id="p2_exact_core_v1",
        bundle_class="exact_pair",
        raw_cache_dir=tmp_path / "raw",
        work_cache_dir=tmp_path / "work",
        render_pdf_pages=True,
        keep_raw=True,
        source_limits={"pmc_oa_pdf_xml": 1},
        max_pages_per_document={"pmc_oa_pdf_xml": 1},
    )

    pmcid = "PMCUNIT0001"
    doc_root = context.raw_cache_dir / "pmc_oa_pdf_xml" / pmcid
    package_path = doc_root / f"{pmcid}.tar.gz"
    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Synthetic abstract text for PMC exact-core.")
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    nxml = """
    <article>
      <front>
        <article-meta>
          <title-group><article-title>Unit Test Paper</article-title></title-group>
          <abstract><p>Synthetic abstract text for PMC exact-core.</p></abstract>
        </article-meta>
      </front>
      <body>
        <sec><title>Body</title><p>Body text.</p></sec>
      </body>
    </article>
    """.encode("utf-8")
    _write_tar_archive(
        package_path,
        {
            f"{pmcid}/{pmcid}.nxml": nxml,
            f"{pmcid}/{pmcid}.pdf": pdf_bytes,
        },
    )

    from leopardi.data_pipeline import workers as worker_module
    original_iter = worker_module._iter_pmc_records
    worker_module._iter_pmc_records = lambda limit: iter(
        [{"pmcid": pmcid, "citation": "Unit citation", "license": "CC-BY"}]
    )
    try:
        sample = next(iter(build_worker_registry()["pmc_oa_pdf_xml"].iter_samples(context)))
    finally:
        worker_module._iter_pmc_records = original_iter

    assert sample.sample_id.startswith("pmc_oa_pdf_xml-")
    assert sample.target_type == "page_markdown_projection"
    assert "## Abstract" in sample.canonical_target


def test_pagexml_to_markdown_preserves_reading_order() -> None:
    xml = """
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
      <Page imageFilename="demo.JPG">
        <TextRegion id="r1">
          <TextLine id="l1"><TextEquiv><Unicode>Line one</Unicode></TextEquiv></TextLine>
          <TextLine id="l2"><TextEquiv><Unicode>Line two</Unicode></TextEquiv></TextLine>
        </TextRegion>
        <TextRegion id="r2">
          <TextLine id="l3"><TextEquiv><Unicode>Line three</Unicode></TextEquiv></TextLine>
        </TextRegion>
      </Page>
    </PcGts>
    """
    markdown = _pagexml_to_markdown(xml)
    assert "Line one\nLine two" in markdown
    assert "\n\nLine three" in markdown


def test_hf_split_inference_marks_crohme_benchmark_years_as_test() -> None:
    assert _infer_hf_split_from_filename("data/train-00000-of-00002.parquet", source_id="crohme") == "train"
    assert _infer_hf_split_from_filename("data/2016-00000-of-00001.parquet", source_id="crohme") == "test"
    assert _infer_hf_split_from_filename("data/validation.parquet", source_id="iam") == "validation"


def test_archive_pair_workers_read_and_bentham_from_local_archives(tmp_path: Path) -> None:
    registry = build_worker_registry()
    page_xml = b"""
    <PcGts xmlns=\"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15\">
      <Page imageFilename=\"demo.JPG\">
        <TextRegion id=\"r1\">
          <TextLine id=\"l1\"><TextEquiv><Unicode>Archivio uno</Unicode></TextEquiv></TextLine>
          <TextLine id=\"l2\"><TextEquiv><Unicode>Archivio due</Unicode></TextEquiv></TextLine>
        </TextRegion>
      </Page>
    </PcGts>
    """.strip()

    read_path = tmp_path / "raw" / "read_2016" / "PublicData.tgz"
    _write_tar_archive(
        read_path,
        {
            "PublicData/Training/page/Seite0001.JPG": _PNG_1X1,
            "PublicData/Training/page/Seite0001.xml": page_xml,
        },
    )

    bentham_images = tmp_path / "raw" / "bentham" / "BenthamDatasetR0-Images.tbz"
    bentham_gt = tmp_path / "raw" / "bentham" / "BenthamDatasetR0-GT.tbz"
    _write_tar_archive(
        bentham_images,
        {"BenthamDatasetR0-Images/Pages/Page0001.jpg": _PNG_1X1},
    )
    _write_tar_archive(
        bentham_gt,
        {"BenthamDatasetR0-GT/Pages/Page0001.xml": page_xml},
    )

    context = SourceBuildContext(
        stage=_manual_stage(),
        experiment_id="archive-test",
        bundle_id="p3_hardcases_v1",
        bundle_class="trusted_aux",
        raw_cache_dir=tmp_path / "raw",
        work_cache_dir=tmp_path / "work",
    )

    read_sample = next(iter(registry["read_2016"].iter_samples(context)))
    assert read_sample.split_assignment == "train"
    assert "Archivio uno" in read_sample.canonical_target

    bentham_sample = next(iter(registry["bentham"].iter_samples(context)))
    assert bentham_sample.source_id == "bentham"
    assert "Archivio due" in bentham_sample.canonical_target


def test_cached_download_validator_rejects_truncated_archives(tmp_path: Path) -> None:
    good_zip = tmp_path / "good.zip"
    _write_zip_archive(good_zip, {"demo.txt": b"ok"})
    assert _is_valid_cached_download(good_zip) is True

    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(good_zip.read_bytes()[:8])
    assert _is_valid_cached_download(bad_zip) is False

    good_tar = tmp_path / "good.tgz"
    _write_tar_archive(good_tar, {"demo.txt": b"ok"})
    assert _is_valid_cached_download(good_tar) is True

    bad_tar = tmp_path / "bad.tgz"
    bad_tar.write_bytes(b"not-a-real-tar")
    assert _is_valid_cached_download(bad_tar) is False
