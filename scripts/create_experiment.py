from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/create_experiment.py <name>")
        return 1

    name = sys.argv[1].strip().replace(" ", "-")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    root = Path("runs") / f"{stamp}-{name}"
    for child in ("configs", "notes", "artifacts"):
        (root / child).mkdir(parents=True, exist_ok=True)
    (root / "notes" / "README.md").write_text(
        f"# Experiment {name}\n\nCreated at {stamp} UTC.\n",
        encoding="utf-8",
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

