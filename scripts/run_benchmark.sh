#!/usr/bin/env bash
set -euo pipefail

python -m leopardi.cli evaluation-materialize "$@"
