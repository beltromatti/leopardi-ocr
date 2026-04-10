#!/usr/bin/env python3
"""Preview/export European multilingual synthetic samples using the production generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leopardi.data_pipeline.european_multilingual_generator import (
    EUROPEAN_WIKIPEDIA_CONFIGS,
    iter_european_multilingual_synthetic_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate European multilingual synthetic data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--per-language", type=int, default=2000, help="Samples per language")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "samples.jsonl"

    total_limit = args.per_language * len(EUROPEAN_WIKIPEDIA_CONFIGS)
    samples = iter_european_multilingual_synthetic_samples(total_limit=total_limit)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for sample in samples:
            image_rel = Path("images") / f"{sample.sample_id}.png"
            (output_dir / image_rel).write_bytes(sample.image_png)
            manifest.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "source_id": "european_multilingual_synthetic",
                        "doc_id": sample.doc_id,
                        "page_id": f"{sample.doc_id}:1",
                        "data_class": "synthetic_exact",
                        "task_family": "document_parsing",
                        "target_type": "page_markdown_projection",
                        "canonical_target": sample.canonical_target,
                        "slice_tags": ["synthetic", "multilingual", "european", sample.language],
                        "metadata": {
                            "language": sample.language,
                            "source": "wikimedia/wikipedia",
                            "title": sample.title,
                        },
                        "asset_paths": [str(image_rel)],
                        "source_license": "cc-by-sa",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(samples)} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
