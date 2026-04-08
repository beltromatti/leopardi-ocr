from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leopardi.ops import ensure_run_layout  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/create_experiment.py <name>")
        return 1

    name = sys.argv[1].strip().replace(" ", "-")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    experiment_id = f"draft-{stamp}-{name}"
    layout = ensure_run_layout(experiment_id, root="runs")
    spec_dir = Path("experiments") / "drafts"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{experiment_id}.md"
    spec_path.write_text(
        "\n".join(
            [
                "# Experiment Draft",
                "",
                f"- experiment id: {experiment_id}",
                f"- created at: {stamp} UTC",
                "- track: ",
                "- stage: ",
                "- hypothesis: ",
                "- model config: ",
                "- data config: ",
                "- optimization config: ",
                "- runtime config: ",
                f"- run root: {layout.experiment_root}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(spec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
