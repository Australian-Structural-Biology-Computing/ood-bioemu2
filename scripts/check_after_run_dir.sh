#!/usr/bin/env bash
# Regression check for template/after.sh.erb's run-directory resolution.
#
# The Open OnDemand wrapper sources after.sh IMMEDIATELY AFTER launching the
# job script in the background -- not after the job completes -- and then
# writes connection.yml, which is what the completed-session view reads. The
# job publishes run_dir.txt within its first seconds, so after.sh must wait
# for it; a job that dies instantly falls back to the live log and finally to
# the sanitized run name plus the PBS job id (which is defined here, because
# the whole wrapper is the PBS job).
#
# This check renders after.sh.erb the way ERB would and replays the four
# staged states against the real wrapper order: script launched in the
# background, after.sh sourced immediately.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T=$(mktemp -d); export T; trap 'rm -rf "$T"' EXIT

python3 - "$ROOT" "$T" <<'PY' || exit 1
import pathlib, re, sys
root, tmp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
src = (root / "template" / "after.sh.erb").read_text()
# The single ERB expression: run_name gsub. Render a name that exercises the
# sanitiser (space -> underscore).
rendered = re.sub(
    r"<%=\s*context\.run_name\.to_s\.gsub\(\"'\", \"\"\)\s*%>",
    "bioemu2 testing", src)
assert "<%" not in rendered, "unhandled ERB tag in after.sh.erb"
(tmp / "after_render.sh").write_text(rendered)
print("after.sh rendered")
PY

pass=0; fail=0
run_case() { # desc, background-setup-command, expected run_dir
  local desc=$1 setup=$2 want=$3 d rc
  d="$T/$1"; mkdir -p "$d"
  ( cd "$d" && eval "$setup" ) &   # the "job", launched in background
  local bg=$!
  ( cd "$d" && PBS_JOBID=9809999.tman bash -c "source '$T/after_render.sh'; printf '%s' \"\$run_dir\"" ) > "$T/out" 2>&1
  rc=$?
  wait "$bg" 2>/dev/null
  local got; got=$(cat "$T/out")
  if [[ "$got" == "$want" ]]; then
    echo "ok   - $desc -> $got"; pass=$((pass + 1))
  else
    echo "FAIL - $desc: want [$want] got [$got] (rc=$rc)"; fail=$((fail + 1))
  fi
}

run_case "run_dir.txt appears within a second" \
  'sleep 0.8; echo bioemu2_testing_9801111 > run_dir.txt' \
  "bioemu2_testing_9801111"
run_case "job dies before publishing anything" \
  'true' \
  "bioemu2_testing_9809999"
run_case "job echoes its output dir, then dies" \
  'echo "BioEmu2 output: /srv/scratch/x/bioemu2_testing_9802222" > live_log.log' \
  "bioemu2_testing_9802222"
run_case "run_dir.txt already present" \
  'echo bioemu2_testing_9803333 > run_dir.txt' \
  "bioemu2_testing_9803333"

echo "PASS=$pass FAIL=$fail"
[[ $fail -eq 0 ]]
