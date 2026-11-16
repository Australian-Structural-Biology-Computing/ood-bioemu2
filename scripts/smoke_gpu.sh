#!/usr/bin/env bash
# Run inside a one-GPU PBS allocation after scripts/setup_runtime.sh int8.
set -euo pipefail
[[ -n "${PBS_JOBID:-}" ]] || { echo "Run this script inside a PBS GPU allocation." >&2; exit 1; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${BIOEMU2_IMAGE:-/srv/scratch/sbf-pipelines/containers/ghcr.io-jscgh-proteinfold_colabfold-af3-preview-e9e8bd0.img}
WEIGHTS=${BIOEMU2_WEIGHTS_ROOT:-/srv/scratch/sbf-pipelines/bioemu2/weights}
OUT=${1:-/srv/scratch/${USER}/bioemu2-smoke-${PBS_JOBID%%.*}}
CACHE=${BIOEMU2_CACHE_ROOT:-/srv/scratch/${USER}/.bioemu2/cache}
mkdir -p "${OUT}" "${OUT}/work" "${CACHE}"
apptainer exec --nv --cleanenv \
  --bind "${ROOT}:${ROOT}:ro" --bind "${OUT}:${OUT}" --bind "${CACHE}:${CACHE}" --bind "${WEIGHTS}:${WEIGHTS}" \
  "${IMAGE}" python "${ROOT}/template/bioemu2_colabfold.py" \
  --runner "${ROOT}/template/run_alphafold.py" --output-dir "${OUT}/results" --work-dir "${OUT}/work" \
  --cache-dir "${CACHE}" --weights-dir "${WEIGHTS}" --job-name smoke \
  --protein ACDEFGHIKLMNPQRSTVWY --msa-mode single_sequence --seeds 1 \
  --num-recycles 1 --num-diffusion-samples 1 --mode structure --sampler edm \
  --physicality high_quality --weights-precision int8
python3 "${ROOT}/template/build_report.py" "${OUT}/results"
echo "GPU smoke test passed: ${OUT}/results/report.html"
