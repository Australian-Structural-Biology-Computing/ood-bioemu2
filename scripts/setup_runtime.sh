#!/usr/bin/env bash
# Administrator-only: prefetch the BioEmu2 ported weights used by the KoD app.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${BIOEMU2_IMAGE:-/srv/scratch/sbf-pipelines/containers/ghcr.io-jscgh-proteinfold_colabfold-af3-preview-e9e8bd0.img}
WEIGHTS_ROOT=${BIOEMU2_WEIGHTS_ROOT:-/srv/scratch/sbf-pipelines/bioemu2/weights}
PRECISION=${1:-int8}
GROUP=${BIOEMU2_RUNTIME_GROUP:-sbf-pipelines}

case "${PRECISION}" in int8|fp16|fp32) ;; *) echo "Usage: $0 [int8|fp16|fp32]" >&2; exit 2;; esac
[[ -r "${IMAGE}" ]] || { echo "Container not readable: ${IMAGE}" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "apptainer is required" >&2; exit 1; }
sha256sum -c <(printf '70a1102908353d75bccfccfa85738f72d053a8fa819554c5e02e6230cc9ba86b  %s\n' "${ROOT}/template/run_alphafold.py")

mkdir -p "${WEIGHTS_ROOT}/${PRECISION}"
exec 9>"${WEIGHTS_ROOT}/.setup.lock"
flock 9
apptainer exec --cleanenv --bind "${WEIGHTS_ROOT}:${WEIGHTS_ROOT}" "${IMAGE}" \
  python - "${WEIGHTS_ROOT}/${PRECISION}" "${PRECISION}" <<'PY'
import sys
from pathlib import Path
from alphafold3.model import weights

destination, precision = sys.argv[1:]
resolved = weights.ensure_weights("bioemu2", destination, download=True, precision=precision)
# Assert the exact published filename, not just "some big blob": the loader
# accepts any *.bin.zst in the directory and never re-checks the precision, so a
# directory holding the wrong representation would otherwise pass as ready.
from alphafold3.model import model_registry

wanted = model_registry.get("bioemu2").weights_file_for(precision)
path = Path(resolved) / wanted
if not path.is_file() or path.stat().st_size < 100_000_000:
    raise SystemExit(f"BioEmu2 {precision} weights are missing or incomplete: expected {path}")
print(f"BioEmu2 {precision} weights ready: {path} ({path.stat().st_size} bytes)")
PY
chgrp -R "${GROUP}" "${WEIGHTS_ROOT}"
find "${WEIGHTS_ROOT}" -type d -exec chmod 2775 {} +
find "${WEIGHTS_ROOT}" -type f -exec chmod 664 {} +
echo "Runtime setup complete. Keep BIOEMU2_ALLOW_WEIGHT_DOWNLOAD=false in production."
