#!/usr/bin/env bash
set -euo pipefail
LPI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
cd "$LPI_ROOT"
exec /home/troy/anaconda3/envs/Difdet/bin/python tools/lpi_train.py "$@"
