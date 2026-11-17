#!/usr/bin/env bash
# Exercise the weight gate inside the rendered job script across every scenario.
# Not part of the app: a development check for template/script.sh.erb, which the
# OOD validator only syntax-checks. Run: bash scripts/check_gate_branches.sh
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T=$(mktemp -d); export T; trap 'rm -rf "$T"' EXIT

python3 - "$ROOT" "$T" <<'PY' || exit 1
import pathlib, re, sys
root, tmp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
CTX = {"weights_precision": "int8", "protein_input": "ACDE", "dna_input": "", "rna_input": "",
       "ligand_ccd": "", "ligand_smiles": "", "bioemu2_mode": "structure", "msa_mode": "single_sequence",
       "seeds": "1", "num_recycles": "1", "num_diffusion_samples": "1", "sampler": "edm",
       "physicality": "high_quality", "propagate_from": "", "propagation_steps": "1",
       "lag_ns": "1", "run_name": "gate_probe"}
def esc(v): return "'" + str(v).replace("'", "'\\''") + "'"
def expr(m):
    inner = m.group(1).strip()
    if inner.startswith("require"): return ""
    found = re.fullmatch(r"Shellwords\.escape\(context\.(\w+)\.to_s\)", inner)
    if found: return esc(CTX[found.group(1)])
    sys.exit("unhandled ERB expression: " + repr(inner))
src = (root / "template" / "script.sh.erb").read_text()
rendered = re.sub(r"<%=(.*?)%>", expr, src, flags=re.S)
rendered = re.sub(r"<%(?!=).*?%>", "", rendered, flags=re.S)
assert "<%" not in rendered and "%>" not in rendered, "leftover ERB tag"
lines = rendered.splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.startswith("WEIGHTS_PRECISION="))
wdir = next(i for i, l in enumerate(lines) if l.startswith('WEIGHTS_DIR="$('))
metrics = next(i for i, l in enumerate(lines[wdir:], wdir) if l.startswith("if [[ \"${ENABLE_METRICS}\""))
end = max(i for i, l in enumerate(lines[wdir:metrics], wdir) if l.rstrip() == "esac")
apx = next(i for i, l in enumerate(lines) if l.startswith("ALLOW_DOWNLOAD=()"))
tail = next(i for i, l in enumerate(lines[apx:], apx) if l.startswith("apptainer exec"))
head = lines[:wdir]
assigns = [l for l in head if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", l)]
sourcelib = [l for l in head if "bioemu2_weights_lib.sh" in l and l.strip().startswith(".")]
out = ["#!/bin/bash\n", "set -eEuo pipefail\n"]
# Environment overrides FIRST: some staged assignment lines reference
# OOD_SESSION_DIR, which is unbound under set -u until it is defined.
out += [f'OOD_SESSION_DIR="$T"\n', 'ENV_FILE=.env\n', f'APP_ROOT="{root}"\n']
out += [l for l in assigns if not l.startswith(("OOD_SESSION_DIR", "ENV_FILE", "APP_ROOT"))]
out += sourcelib + lines[start:metrics] + lines[apx:tail]
out += ['echo "GATE_RESOLVED=${WEIGHTS_GATE_FILE}"\n',
        'echo "BIND=${WEIGHTS_BIND}${WEIGHTS_BIND_SUFFIX}"\n',
        'echo "FLAG_WEIGHTS_DIR=${WEIGHTS_GATE_ROOT}/${WEIGHTS_PRECISION}"\n']
(tmp / "harness.sh").write_text("".join(out))
# The rendered script sources the gate library next to itself (that is how the
# staged session finds it), so the harness needs a copy beside harness.sh.
(tmp / "bioemu2_weights_lib.sh").write_bytes((root / "template" / "bioemu2_weights_lib.sh").read_bytes())
print("harness built from current source")
PY

W="$T/weights/int8"; mkdir -p "$W"
go() { # desc, downloads, expected_rc, expected_substring
  local desc=$1 dl=$2 want_rc=$3 want=$4 o e rc
  o="$T/o"; e="$T/e"
  PBS_JOBID=99.x BASE_WORK_DIR="$T/work" BIOEMU2_WEIGHTS_ROOT="$T/weights" \
    BIOEMU2_ALLOW_WEIGHT_DOWNLOAD=$dl bash "$T/harness.sh" >"$o" 2>"$e"; rc=$?
  if [[ "$rc" == "$want_rc" ]] && grep -qF -- "$want" "$o" "$e"; then
    echo "ok   - $desc (rc=$rc)"
  else
    echo "FAIL - $desc: want rc=$want_rc containing '$want'; got rc=$rc"
    sed 's/^/         out| /' "$o"; sed 's/^/         err| /' "$e"
  fi
}
blank() { rm -f "$W"/*.bin.zst; }
blank
go "missing weights, downloads OFF  -> fatal" false 1 "administrator must run scripts/setup_runtime.sh"
go "missing weights, downloads ON   -> warn+continue" true 0 "GATE_RESOLVED="
head -c 200000000 /dev/zero > "$W/bioemu2.int8.bin.zst"
go "complete int8 blob              -> accepted" false 0 "weights: $W/bioemu2.int8.bin.zst"
blank; head -c 200000000 /dev/zero > "$W/bioemu2.fp16.bin.zst"
go "foreign blob name               -> fatal" false 1 "not the published name"
blank; head -c 100 /dev/zero > "$W/bioemu2.int8.bin.zst"
go "truncated blob                  -> fatal" false 1 "incomplete"
head -c 200000000 /dev/zero > "$W/bioemu2.fp16.bin.zst"
go "two blobs in one directory      -> fatal" false 1 "2 *.bin.zst files"
go "corrupt shared dir, downloads ON -> fall back to private download" true 0 "falling back to a private download"
blank; head -c 200000000 /dev/zero > "$W/bioemu2.int8.bin.zst"
go "asked precision already staged, downloads ON -> adopt shared" true 0 "already installed"

blank; head -c 200000000 /dev/zero > "$W/bioemu2.fp16.bin.zst"
go "only fp16 staged, int8 asked, downloads ON -> private download" true 0 "falling back to a private download"
blank; head -c 200000000 /dev/zero > "$W/bioemu2.int8.bin.zst"
go "int8 staged, int8 asked, downloads ON -> adopt shared, read-only bind" true 0 "BIND=$T/weights:ro"
blank
echo "--- private download dir is used, shared root untouched ---"
find "$T/work" -mindepth 1 -type d | sed 's/^/   work: /'
