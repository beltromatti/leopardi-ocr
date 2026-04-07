# Data Pipeline

The data stack is split so large-scale experiments remain manageable:

- `ingestion/`: raw source acquisition, metadata extraction, deduplication, and page manifests
- `synthesis/`: synthetic perturbation, rendering, and target generation
- `curation/`: quality filters, heuristics, and scorecards before training
- `manifests/`: versioned dataset and source manifests
- `splits/`: explicit split definitions
- `registry/`: lightweight dataset status and provenance registry

Keep raw data outside the repository. Store only configs, manifests, and reproducible scripts.
