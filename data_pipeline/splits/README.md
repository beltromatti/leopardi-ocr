# Data Splits

Keep split definitions versioned and explicit.

Recommended split families:

- `pretrain_exact`
- `pretrain_synthetic`
- `sft_exact`
- `eval_holdout`

Every split version should document:

- source pools
- exclusions
- benchmark leakage checks
