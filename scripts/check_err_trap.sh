#!/usr/bin/env bash
# Regression check for the ERR trap in template/script.sh.erb.
#
# The trap body once ran `set +e` at handler scope, which LEAKS: errexit stays
# disabled for the rest of the job script, so a failed fold still ran the
# report step, printed "BioEmu2 completed successfully", wrote run_dir.txt,
# and exited 0 (PBS job 9800577 did exactly that). The body now runs in a
# subshell; this check extracts the block verbatim from the template -- it
# contains no ERB -- and proves that a failure both populates the diagnostics
# directory and still kills the script with a non-zero status.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

awk '/^collect_debug_files\(\) \{/,/^trap collect_debug_files ERR$/' \
  "$ROOT/template/script.sh.erb" > "$T/trap.sh"
if [[ ! -s "$T/trap.sh" ]]; then
  echo "FAIL - could not extract the ERR trap block from the template"
  exit 1
fi
if ! grep -q '^  ($' "$T/trap.sh"; then
  echo "FAIL - trap body is not wrapped in a subshell; the set +e leak is back"
  exit 1
fi

export DEBUGDIR="$T/debug" LIVE_LOG="$T/live.log" OUT_DIR="$T/out"
export RUN_ID="check_1" BIOEMU2_IMAGE="/nonexistent.img" PBS_JOBID="1"
export USER="$(id -un)" DEBUGGROUP="$(id -gn)"
mkdir -p "$OUT_DIR"
echo "log line before the failure" > "$LIVE_LOG"

bash -c '
  set -eEuo pipefail
  source "$DEBUGDIR/../../trap.sh" 2>/dev/null || source "'"$T"'/trap.sh"
  false
  echo "SCRIPT CONTINUED AFTER FAILURE"
  exit 0
' > "$T/out.log" 2>&1
rc=$?

fail=0
if [[ $rc -eq 0 ]]; then
  echo "FAIL - script exited 0 after a trapped failure (status=$rc)"
  fail=1
else
  echo "ok   - failure keeps a non-zero exit status ($rc)"
fi
if grep -q "SCRIPT CONTINUED AFTER FAILURE" "$T/out.log"; then
  echo "FAIL - execution continued past the failure"
  sed 's/^/       /' "$T/out.log"
  fail=1
else
  echo "ok   - errexit holds after the trap runs"
fi
if [[ -s "$DEBUGDIR/debug-metadata.txt" ]] && grep -q "exit_code=1" "$DEBUGDIR/debug-metadata.txt"; then
  echo "ok   - diagnostics recorded the failing exit code"
else
  echo "FAIL - debug-metadata.txt missing or wrong"
  cat "$DEBUGDIR/debug-metadata.txt" 2>/dev/null
  fail=1
fi
if [[ -s "$DEBUGDIR/live.log" ]]; then
  echo "ok   - live log copied into the diagnostics directory"
else
  echo "FAIL - live log was not copied"
  fail=1
fi
if grep -q "BioEmu2 failed (exit 1)" "$T/out.log"; then
  echo "ok   - the user-visible failure message is still emitted"
else
  echo "FAIL - failure banner missing"
  sed 's/^/       /' "$T/out.log"
  fail=1
fi

[[ $fail -eq 0 ]]
