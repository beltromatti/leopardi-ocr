# Config Stack

Configs in Leopardi should be layered, not monolithic.

The stack should answer a simple question:

What exact combination of data, model, training stage, evaluation protocol, and runtime produced a result?

## Layer Order

Recommended config stack order:

1. `model/`
2. `data/`
3. `pretraining/`, `finetune/`, or `optimization/`
4. `eval/`
5. `runtime/`
6. `overrides/`

## Directory Roles

- `model/`: model-family definitions such as `Leopardi-S0` and `Leopardi-S1`
- `data/`: dataset mixtures, split versions, and canonical target settings
- `pretraining/`: stage configs `p0` to `p3`
- `finetune/`: stage configs `f0` to `f3`
- `optimization/`: post-finetune optimization stages `o0` to `o4`
- `eval/`: benchmark protocol presets
- `runtime/`: training and serving environment presets
- `overrides/`: temporary ablation overrides, never long-lived defaults
- `templates/`: example stack compositions

## Best Practices

1. One file should express one concern.
2. Promotion candidates should not depend on ad hoc overrides hidden in shell history.
3. If an override survives more than a few runs, it should be promoted into the main stack.
4. Config names should reflect track and stage.
