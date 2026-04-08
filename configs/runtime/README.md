# Runtime Configs

Store phase-specific runtime presets here.

Examples:

- single-GPU RTX 5090 data builds
- single-GPU RTX 5090 training
- single-GPU RTX 5090 finetuning
- single-GPU RTX 5090 optimization
- single-GPU RTX 5090 evaluation
- single-GPU serving
- later multi-GPU or datacenter presets

Every serious preset should declare:

- logging and heartbeat paths
- control files
- persistent targets
- the main optimization priority for that phase
