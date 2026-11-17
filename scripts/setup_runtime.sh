#!/usr/bin/env bash
# Administrator-only: prefetch the BioEmu2 ported weights used by the KoD app.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../template/bioemu2_weights_lib.sh
. "${ROOT}/template/bioemu2_weights_lib.sh"
IMAGE=${BIOEMU2_IMAGE:-/srv/scratch/sbf-pipelines/containers/ghcr.io-jscgh-proteinfold_colabfold-af3-preview-e9e8bd0.img}
WEIGHTS_ROOT=${BIOEMU2_WEIGHTS_ROOT:-/srv/scratch/sbf-pipelines/bioemu2/weights}
PRECISION=${1:-int8}
GROUP=${BIOEMU2_RUNTIME_GROUP:-sbf-pipelines}

bioemu2_valid_precision "${PRECISION}" || { echo "Usage: $0 [${BIOEMU2_PRECISIONS// /|}]" >&2; exit 2; }
[[ -r "${IMAGE}" ]] || { echo "Container not readable: ${IMAGE}" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "apptainer is required" >&2; exit 1; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 1; }
sha256sum -c <(printf '70a1102908353d75bccfccfa85738f72d053a8fa819554c5e02e6230cc9ba86b  %s\n' "${ROOT}/template/run_alphafold.py")

TARGET_DIR=$(bioemu2_weights_dir "${WEIGHTS_ROOT}" "${PRECISION}")
mkdir -p "${TARGET_DIR}"

# The gate below is the SAME one template/script.sh.erb applies to a queued job,
# so this script can never declare a layout that user jobs then refuse to load.
# It runs before the download to skip completed work, and again afterwards so a
# blob that arrived truncated or under the wrong name is still caught.
gate_and_report() {
  local reason='' file=''
  bioemu2_gate_weights "${WEIGHTS_ROOT}" "${PRECISION}" reason file
  # Status on its own line: a bare word `case` right after a command is parsed
  # as a new pattern-clause header, not as `case $? in`, which swallows this
  # whole block and still passes `bash -n`.
  local status=$?
  case "${status}" in
    0) echo "BioEmu2 ${PRECISION} weights ready: ${file}"; return 0;;
    1) return 1;;
    *) echo "ERROR: ${reason}" >&2; return 1;;
  esac
}
if gate_and_report; then
  echo "Nothing to do."
  exit 0
fi

# One lock for the whole root, held across the download: a second administrator
# fetching the same precision concurrently would otherwise leave two
# *.bin.zst files in one directory, which the upstream loader resolves by
# picking any blob it finds. The lock lives here in the administrator-owned
# weight root rather than /tmp, where a predictable name would be a symlink
# target for a script that runs with elevated rights.
exec 8>"${WEIGHTS_ROOT}/.setup.lock"
flock 8
apptainer exec --cleanenv --bind "${WEIGHTS_ROOT}:${WEIGHTS_ROOT}" "${IMAGE}" \
  python - "${TARGET_DIR}" "${PRECISION}" <<'PY'
import sys
from pathlib import Path
from alphafold3.model import weights

destination, precision = sys.argv[1:]
resolved = weights.ensure_weights("bioemu2", destination, download=True, precision=precision)
from alphafold3.model import model_registry

wanted = model_registry.get("bioemu2").weights_file_for(precision)
path = Path(resolved) / wanted
if not path.is_file():
    raise SystemExit(f"BioEmu2 {precision} weights were not staged at {path}")
print(f"Downloaded {path} ({path.stat().st_size} bytes)")
PY
chgrp -R "${GROUP}" "${WEIGHTS_ROOT}"
find "${WEIGHTS_ROOT}" -type d -exec chmod 2775 {} +
find "${WEIGHTS_ROOT}" -type f -exec chmod 664 {} +

gate_and_report
echo "Runtime setup complete. Keep BIOEMU2_ALLOW_WEIGHT_DOWNLOAD=false in production."
