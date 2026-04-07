# Data Pipeline

The data stack is split so large-scale experiments remain manageable:

- `ingestion/`: raw source acquisition, metadata extraction, deduplication, and page manifests
- `synthesis/`: synthetic perturbation, rendering, and target generation
- `curation/`: quality filters, heuristics, and scorecards before training

Keep raw data outside the repository. Store only configs, manifests, and reproducible scripts.

