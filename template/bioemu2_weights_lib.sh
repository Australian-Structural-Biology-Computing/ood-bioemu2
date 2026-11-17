# shellcheck shell=bash
# Single source of truth for the BioEmu2 stored-weight gate.
#
# Sourced (never executed) by scripts/setup_runtime.sh and template/script.sh.erb.
# Both callers MUST apply the same rule. Upstream's
# alphafold3.model.weights.ensure_weights returns any *.bin.zst already sitting
# in --model_dir and deliberately does NOT re-check the requested precision, so a
# directory holding the wrong representation looks "ready" to a caller that only
# asks "is some blob here?", and is then loaded as the weights the user asked
# for. The gate therefore accepts exactly one complete, correctly named blob and
# refuses everything else.
#
# Keep the precision list in step with template/bioemu2_colabfold.py
# (BIOEMU2_PRECISIONS), which cannot source this file because it runs inside the
# container; tests/test_bioemu2_colabfold.py asserts the two agree.

# The stored-weight representations this app offers, in form order.
BIOEMU2_PRECISIONS='int8 fp16 fp32'

# Smallest size still considered a complete download. The int8 blob is ~490 MB,
# so this only rejects truncation or an empty placeholder.
BIOEMU2_MIN_WEIGHT_SIZE=${BIOEMU2_MIN_WEIGHT_SIZE:-100000000}

# The published basename of the blob at a precision. Mirrors
# alphafold3.model.model_registry.ModelSpec.weights_file_for: 'fp32' is the
# original release and carries no infix, every other precision gains
# '.<precision>'.
bioemu2_weights_filename() {
  local precision=${1:-}
  case "${precision}" in
    fp32) printf 'bioemu2.bin.zst\n' ;;
    '') return 1 ;;
    *) printf 'bioemu2.%s.bin.zst\n' "${precision}" ;;
  esac
}

# The per-precision directory under a shared weight root. Every precision, fp32
# included, gets its own directory: upstream's weights.default_dir puts all of
# them in one place, and 'bioemu2.bin.zst' beside 'bioemu2.int8.bin.zst' is two
# candidates for one slot in the loader's directory-wide glob.
# Usage: bioemu2_weights_dir <root> <precision>
bioemu2_weights_dir() {
  local root=${1:-} precision=${2:-}
  [[ -n "${root}" && -n "${precision}" ]] || return 1
  printf '%s/%s\n' "${root%/}" "${precision}"
}

# Is $1 a precision this app supports?
bioemu2_valid_precision() {
  local candidate=${1:-} precision
  for precision in ${BIOEMU2_PRECISIONS}; do
    [[ "${candidate}" == "${precision}" ]] && return 0
  done
  return 1
}

# The path a caller must find, or failure when the precision is unknown.
# Usage: bioemu2_expected_weights <root> <precision>
bioemu2_expected_weights() {
  local root=${1:-} precision=${2:-} filename
  bioemu2_valid_precision "${precision}" || return 1
  filename=$(bioemu2_weights_filename "${precision}") || return 1
  printf '%s/%s\n' "$(bioemu2_weights_dir "${root}" "${precision}")" "${filename}"
}

# Is a blob large enough to be a complete download?
bioemu2_weights_complete() {
  local file=${1:-} size
  [[ -f "${file}" ]] || return 1
  size=$(stat -c %s -- "${file}" 2>/dev/null) || return 1
  [[ "${size}" =~ ^[0-9]+$ ]] && ((size >= BIOEMU2_MIN_WEIGHT_SIZE))
}

# Decide what a precision directory provides. Sets <file_var> to the resolved
# path (or the directory itself when unusable) and <reason_var> to the diagnosis.
#
# Usage: bioemu2_resolve_weights <dir> <wanted_basename> file_var reason_var
#   -> 0 ready | 1 empty (a download may fill it) | 2 unusable (always fatal)
bioemu2_resolve_weights() {
  local dir=${1:-} wanted=${2:-} file_var=${3:-} reason_var=${4:-}
  [[ -n "${file_var}" && -n "${reason_var}" ]] || return 2
  local -a blobs=()
  local candidate
  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] && blobs+=("${candidate##*/}")
  done < <(find "${dir}" -maxdepth 1 -type f -name '*.bin.zst' 2>/dev/null | sort)

  case "${#blobs[@]}" in
    0)
      printf -v "${file_var}" '%s' ''
      printf -v "${reason_var}" '%s holds no *.bin.zst' "${dir}"
      return 1
      ;;
    1)
      if [[ "${blobs[0]}" != "${wanted}" ]]; then
        printf -v "${file_var}" '%s' "${dir}/${blobs[0]}"
        printf -v "${reason_var}" '%s holds %s, which is not the published name %s' \
          "${dir}" "${blobs[0]}" "${wanted}"
        return 2
      fi
      if ! bioemu2_weights_complete "${dir}/${blobs[0]}"; then
        printf -v "${file_var}" '%s' "${dir}/${blobs[0]}"
        printf -v "${reason_var}" '%s/%s is present but incomplete (need at least %s bytes)' \
          "${dir}" "${blobs[0]}" "${BIOEMU2_MIN_WEIGHT_SIZE}"
        return 2
      fi
      printf -v "${file_var}" '%s' "${dir}/${blobs[0]}"
      printf -v "${reason_var}" 'ok'
      return 0
      ;;
    *)
      printf -v "${file_var}" '%s' "${dir}"
      printf -v "${reason_var}" '%s holds %s *.bin.zst files (%s); the runner would load any one of them without checking the precision' \
        "${dir}" "${#blobs[@]}" "$(IFS=', '; echo "${blobs[*]}")"
      return 2
      ;;
  esac
}

# Gate result for both callers. Sets the named variables rather than printing so
# each caller can render its own wording (the job script must name the
# administrator recovery step).
#
# Usage: bioemu2_gate_weights <root> <precision> reason_var file_var
#   -> 0 ready | 1 missing (may download) | 2 unusable (always fatal)
bioemu2_gate_weights() {
  local root=${1:-} precision=${2:-} reason_var=${3:-} file_var=${4:-}
  [[ -n "${reason_var}" && -n "${file_var}" ]] || return 2
  local dir wanted filename resolved='' reason=''
  if ! wanted=$(bioemu2_expected_weights "${root}" "${precision}"); then
    printf -v "${reason_var}" 'unsupported precision %s (want one of: %s)' \
      "${precision:-<empty>}" "${BIOEMU2_PRECISIONS// /, }"
    printf -v "${file_var}" '%s' ''
    return 2
  fi
  dir=$(bioemu2_weights_dir "${root}" "${precision}")
  filename=${wanted##*/}
  bioemu2_resolve_weights "${dir}" "${filename}" resolved reason
  local status=$?
  printf -v "${reason_var}" '%s' "${reason}"
  printf -v "${file_var}" '%s' "${resolved}"
  return "${status}"
}
