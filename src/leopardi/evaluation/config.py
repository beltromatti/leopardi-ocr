from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EvaluationRuntimeConfig:
    hardware_tag: str = "rtx5090"
    primary_runtime: str = "vllm"
    alternate_runtime: str = "sglang"
    structured_output_backend: str = "auto"
    max_concurrent_requests: int = 1
    warmup_pages: int = 8
    log_every: int = 10
    eval_every: int = 1
    save_every: int = 1


@dataclass(slots=True)
class EvaluationStageConfig:
    protocol: str
    bundle_id: str
    decode_modes: tuple[str, ...] = ("standard",)
    public_benchmarks: tuple[str, ...] = ()
    difficulty_tiers: tuple[str, ...] = ()
    required_slices: tuple[str, ...] = ()
    reporting: tuple[str, ...] = ()
    runtime: EvaluationRuntimeConfig = field(default_factory=EvaluationRuntimeConfig)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        runtime_payload: dict[str, Any] | None = None,
    ) -> "EvaluationStageConfig":
        eval_payload = payload["eval"]
        runtime_root = (runtime_payload or {}).get("runtime", runtime_payload or {})
        protocol_name = eval_payload["protocol"]
        if protocol_name == "public_frontier":
            protocol = "public_frontier_v1"
            bundle_id = "public_frontier_v1"
            decode_modes = ("fast", "standard", "hard")
        elif protocol_name == "internal_holdout":
            protocol = "internal_holdout_v1"
            bundle_id = "Leopardi_Internal_Holdout"
            decode_modes = ("standard", "hard")
        else:
            protocol = protocol_name
            bundle_id = protocol
            decode_modes = tuple(eval_payload.get("decode_modes", ("standard",)))
        return cls(
            protocol=protocol,
            bundle_id=bundle_id,
            decode_modes=tuple(eval_payload.get("decode_modes", decode_modes)),
            public_benchmarks=tuple(eval_payload.get("public_benchmarks", ())),
            difficulty_tiers=tuple(eval_payload.get("difficulty_tiers", ())),
            required_slices=tuple(eval_payload.get("required_slices", ())),
            reporting=tuple(eval_payload.get("reporting", ())),
            runtime=EvaluationRuntimeConfig(
                hardware_tag=runtime_root.get("hardware_tag", "rtx5090"),
                primary_runtime=runtime_root.get("primary_runtime", "vllm"),
                alternate_runtime=runtime_root.get("alternate_runtime", "sglang"),
                structured_output_backend=runtime_root.get("structured_output_backend", "auto"),
                max_concurrent_requests=runtime_root.get("max_concurrent_requests", 1),
                warmup_pages=runtime_root.get("warmup_pages", 8),
                log_every=runtime_root.get("log_every", 10),
                eval_every=runtime_root.get("eval_every", 1),
                save_every=runtime_root.get("save_every", 1),
            ),
        )

    @classmethod
    def from_yaml(
        cls,
        stage_path: str | Path,
        runtime_path: str | Path | None = None,
    ) -> "EvaluationStageConfig":
        with Path(stage_path).open("r", encoding="utf-8") as handle:
            stage_payload = yaml.safe_load(handle)
        runtime_payload: dict[str, Any] | None = None
        if runtime_path is not None:
            with Path(runtime_path).open("r", encoding="utf-8") as handle:
                runtime_payload = yaml.safe_load(handle)
        return cls.from_dict(stage_payload, runtime_payload=runtime_payload)
