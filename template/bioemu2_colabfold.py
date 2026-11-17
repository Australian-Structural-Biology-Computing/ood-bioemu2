#!/usr/bin/env python3
"""Prepare and run a pinned ColabFold2-preview BioEmu2 job.

This is deliberately a thin, headless translation of ColabFold2_preview.ipynb.
It validates user input, writes an AlphaFold-3-dialect JSON input, selects the
GPU-specific XLA settings used by the notebook, and invokes the vendored
run_alphafold.py. Retrieved text is data and is never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROTEIN = frozenset("ACDEFGHIKLMNPQRSTVWYX")
DNA = frozenset("ACGTN")
RNA = frozenset("ACGUN")
CHAIN_IDS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
CCD_RE = re.compile(r"[A-Za-z0-9]{1,8}\Z")
JOB_RE = re.compile(r"[^A-Za-z0-9]+")
# Mirror of template/bioemu2_weights_lib.sh, which this module cannot source
# because it runs inside the container. Both gate the same published blobs, so
# keep the two in step.
BIOEMU2_PRECISIONS = ("int8", "fp16", "fp32")


class InputError(ValueError):
    """Input cannot be represented safely as a fold request."""


def _clean_sequence(value: str) -> str:
    return "".join(ch for ch in value if not ch.isspace() and ch not in {"\ufeff", "\u200b"}).upper()


def _read_fasta(path: Path) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    pending: list[str] = []  # lines before the first header
    saw_header = False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if pending:
                # Sequence data before the first header is garbled input, and
                # silently merging it into the first record would fold junk
                # into a chain. A file with no headers at all is the one
                # legitimate shape for leading data: a bare sequence.
                raise InputError(f"{path} has sequence data before the first '>' header")
            if saw_header:
                if not current:
                    raise InputError(f"empty FASTA record in {path}")
                records.append(_clean_sequence("".join(current)))
                current = []
            saw_header = True
            continue
        if saw_header:
            current.append(line)
        else:
            pending.append(line)
    if pending:
        records.append(_clean_sequence("".join(pending)))
    elif saw_header and not current:
        # A file ending right after a header used to drop that record
        # silently, shifting every later chain id.
        raise InputError(f"empty FASTA record in {path}")
    elif current:
        records.append(_clean_sequence("".join(current)))
    if not records:
        raise InputError(f"no sequence records found in {path}")
    return records


def _split_polymer(value: str, alphabet: frozenset[str], label: str, allow_path: bool = False) -> list[str]:
    value = value.strip()
    if not value:
        return []
    candidate = Path(value).expanduser()
    if allow_path and candidate.is_file():
        entries = _read_fasta(candidate)
    else:
        entries = [_clean_sequence(part) for part in re.split(r"[:\n]+", value) if _clean_sequence(part)]
    for entry in entries:
        invalid = sorted(set(entry) - alphabet)
        if invalid:
            raise InputError(f"{label} contains unsupported characters: {''.join(invalid)}")
        if not entry:
            raise InputError(f"{label} contains an empty chain")
    return entries


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for token in re.findall(r"\d+", value):
        seed = int(token)
        if seed > 2**31 - 1:
            raise InputError("seeds must be at most 2147483647")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise InputError("provide at least one non-negative integer seed")
    if len(seeds) > 32:
        raise InputError("at most 32 unique seeds are allowed")
    return seeds


def parse_ccd(value: str) -> list[str]:
    codes = [part.upper() for part in re.split(r"[\s,:]+", value.strip()) if part]
    for code in codes:
        if not CCD_RE.fullmatch(code):
            raise InputError(f"invalid CCD code: {code!r}")
    return codes


def parse_smiles(value: str) -> list[str]:
    smiles = [line.strip() for line in value.splitlines() if line.strip()]
    for item in smiles:
        if len(item) > 4096 or any(ord(ch) < 32 for ch in item):
            raise InputError("a ligand SMILES is invalid or too long")
    return smiles


def _chain_ids(chain: dict) -> list[str]:
    """Every chain identifier carried by one entity of the AF3-dialect payload."""
    entity = next(iter(chain.values()))
    ident = entity.get("id")
    if isinstance(ident, str):
        return [ident]
    if isinstance(ident, (list, tuple)):
        return [str(part) for part in ident]
    raise InputError("a chain is missing an identifier")


def _distinct_chain_ids(chains: list[dict]) -> None:
    """Fail loudly rather than emit chain ids the runner cannot resolve.

    AlphaFold 3's Input validator assigns one asym_id per entity and then
    requires every id in an entity's chain list to be present, so repeating an
    id across entities raises deep inside inference. Merging identical protein
    sequences into one entity is intended, but it must not consume an id that
    another molecule type has already been given.
    """
    ids = [ident for chain in chains for ident in _chain_ids(chain)]
    duplicates = sorted({ident for ident in ids if ids.count(ident) > 1})
    if duplicates:
        raise InputError(
            "chain identifiers collided: " + ", ".join(duplicates) + "; reduce the number of chains"
        )


def weights_dir_for(precision: str) -> Path:
    """The per-precision directory under the shared weight root.

    Derived here rather than in the job script so the directory a run loads from
    is decided in exactly one place. A foreign representation must never share a
    directory with the wanted one: upstream's loader takes any *.bin.zst it
    finds and does not re-check the precision.
    """
    if precision not in BIOEMU2_PRECISIONS:
        raise InputError(f"weights precision must be one of {', '.join(BIOEMU2_PRECISIONS)}")
    return Path(".") / precision


def build_fold_input(args: argparse.Namespace) -> tuple[dict, str]:
    proteins = _split_polymer(args.protein, PROTEIN, "protein", allow_path=True)
    dna = _split_polymer(args.dna, DNA, "DNA")
    rna = _split_polymer(args.rna, RNA, "RNA")
    ccd = parse_ccd(args.ligand_ccd)
    smiles = parse_smiles(args.ligand_smiles)
    seeds = parse_seeds(args.seeds)
    if not any((proteins, dna, rna, ccd, smiles)):
        raise InputError("provide at least one protein, DNA, RNA, or ligand input")
    entity_count = len(proteins) + len(dna) + len(rna) + len(ccd) + len(smiles)
    if entity_count > len(CHAIN_IDS):
        raise InputError(f"at most {len(CHAIN_IDS)} molecular chains are supported")

    chains: list[dict] = []
    protein_groups: dict[str, dict] = {}
    index = 0
    for sequence in proteins:
        chain_id = CHAIN_IDS[index]
        index += 1
        if sequence in protein_groups:
            entity = protein_groups[sequence]
            ids = entity["id"] if isinstance(entity["id"], list) else [entity["id"]]
            entity["id"] = [*ids, chain_id]
            continue
        # Featurisation validates that every protein chain carries all three
        # fields, and an absent one is a hard error, not a default ("Protein
        # chain 1 is missing paired MSA" -- the first GPU smoke run, PBS job
        # 9804946, failed exactly there). `templates: []` is the no-template
        # representation BioEmu2 samples with, in every mode. In single-sequence
        # mode the MSA fields must also be present: an empty `pairedMsa`
        # featurises identically to a query-only A3M (Msa.from_a3m('') yields
        # [query]), which is the single-sequence form, and it is what AF3's own
        # precompile dummy input uses. In server MSA mode the fields are left
        # absent so the runner's fill_missing_msas populates them.
        entity: dict = {"id": chain_id, "sequence": sequence, "templates": []}
        if args.msa_mode == "single_sequence":
            entity["unpairedMsa"] = f">query\n{sequence}\n"
            entity["pairedMsa"] = ""
        protein_groups[sequence] = entity
        chains.append({"protein": entity})
    for sequence in rna:
        entity = {"id": CHAIN_IDS[index], "sequence": sequence}
        if args.msa_mode == "single_sequence":
            entity["unpairedMsa"] = f">query\n{sequence}\n"
        chains.append({"rna": entity})
        index += 1
    for sequence in dna:
        chains.append({"dna": {"id": CHAIN_IDS[index], "sequence": sequence}})
        index += 1
    for code in ccd:
        chains.append({"ligand": {"id": CHAIN_IDS[index], "ccdCodes": [code]}})
        index += 1
    for item in smiles:
        chains.append({"ligand": {"id": CHAIN_IDS[index], "smiles": item}})
        index += 1
    _distinct_chain_ids(chains)

    flattened = json.dumps({"sequences": chains, "seeds": seeds}, sort_keys=True, separators=(",", ":"))
    base = JOB_RE.sub("", args.job_name).lower() or "bioemu2"
    name = f"{base}_{hashlib.sha1(flattened.encode()).hexdigest()[:7]}"
    return {
        "name": name,
        "sequences": chains,
        "modelSeeds": seeds,
        "dialect": "alphafold3",
        "version": 1,
    }, name


def detect_accelerator() -> tuple[str, list[str]]:
    """Notebook-equivalent attention selection, delegated to the pinned fork.

    The notebook picks the attention implementation and its XLA flags via
    alphafold3.model.components.platform.attention_config, and that policy is
    subtler than a compute-capability threshold: pre-Ampere cards take the
    legacy Volta kernels when they are installed, A100 (exactly 8.0) runs
    Triton, but 8.6/8.9 consumer cards must NOT (tokamax refuses to launch
    there and the fork falls back to Pallas). This wrapper used to
    reimplement that table and had already drifted from it. The wrapper runs
    inside the pinned container, so call the fork's own selector instead of
    keeping a second copy of the policy.
    """
    from alphafold3.model.components import platform

    device, _ = platform.detect_device()
    if device != "gpu":
        raise RuntimeError("BioEmu2 requires an NVIDIA GPU, but none was detected")
    config = platform.attention_config(device=device)
    if config.get("nojit"):
        # Only the CPU row sets nojit; a GPU should never land here.
        raise RuntimeError("no usable GPU accelerator was detected")
    return config["attention"], list(config["xla_flags"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--runner", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--work-dir", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--weights-dir", type=Path, required=True)
    result.add_argument("--job-name", required=True)
    result.add_argument("--protein", default="")
    result.add_argument("--dna", default="")
    result.add_argument("--rna", default="")
    result.add_argument("--ligand-ccd", default="")
    result.add_argument("--ligand-smiles", default="")
    result.add_argument("--msa-mode", choices=("mmseqs2_server", "single_sequence"), default="mmseqs2_server")
    result.add_argument("--seeds", default="1")
    result.add_argument("--num-recycles", type=int, default=10)
    result.add_argument("--num-diffusion-samples", type=int, default=5)
    result.add_argument("--mode", choices=("structure", "equilibrium"), default="structure")
    result.add_argument("--sampler", choices=("edm", "edm_gd", "edm_fk", "edm_fkc"), default="edm")
    result.add_argument("--physicality", choices=("high_quality", "no_violations", "high_quality_per_residue", "no_filter"), default="high_quality")
    result.add_argument("--propagate-from", default="")
    result.add_argument("--propagation-steps", type=int, default=None)
    result.add_argument("--lag-ns", type=float, default=None)
    result.add_argument("--weights-precision", choices=BIOEMU2_PRECISIONS, default="int8")
    result.add_argument("--allow-weight-download", action="store_true")
    return result


def effective_mode(args: argparse.Namespace) -> str:
    """The sampling mode a run gets, given the propagation request.

    Upstream defines propagation as equilibrium sampling by construction:
    with a starting frame set, BIOEMU2_MODE must be unset or 'equilibrium',
    and 'structure' is rejected (alphafold3/model/pipeline/bioemu2_features.py,
    mode()). The app mirrors that contract with a message that names the form
    control, instead of the deep-in-inference error the raw contract produces.
    """
    if not args.propagate_from:
        return args.mode
    if args.mode != "equilibrium":
        raise InputError(
            "trajectory propagation samples the equilibrium ensemble; set the "
            "task to 'Equilibrium ensemble sampling' or clear the starting structure"
        )
    return args.mode


def propagation_flags(args: argparse.Namespace) -> list[str]:
    """Runner flags for trajectory propagation, or [] when not propagating.

    steps is bounded on the app side because the runtime scales with it and
    the allocation is capped at 48 h; lag must be finite and positive, which
    is exactly the contract upstream enforces later (add_propagation).
    """
    if not args.propagate_from:
        return []
    start = Path(args.propagate_from)
    if not start.is_file():
        raise RuntimeError(f"the propagation start structure is unavailable: {start}")
    if args.propagation_steps is None or not 1 <= args.propagation_steps <= 100:
        raise InputError("propagation steps must be between 1 and 100")
    if args.lag_ns is None or not math.isfinite(args.lag_ns) or args.lag_ns <= 0:
        raise InputError("the lag time must be a positive, finite number of nanoseconds")
    return [
        f"--bioemu2_propagate_from={start}",
        f"--bioemu2_propagation_steps={args.propagation_steps}",
        f"--bioemu2_lag_ns={args.lag_ns}",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.num_recycles <= 30:
        raise InputError("recycles must be between 1 and 30")
    if not 1 <= args.num_diffusion_samples <= 100:
        raise InputError("samples per seed must be between 1 and 100")
    if not args.runner.is_file():
        raise RuntimeError(f"pinned ColabFold runner is unavailable: {args.runner}")

    fold_input, job_name = build_fold_input(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    precision_weights = args.weights_dir / weights_dir_for(args.weights_precision)
    precision_weights.mkdir(parents=True, exist_ok=True)
    input_path = args.work_dir / f"{job_name}.json"
    input_path.write_text(json.dumps(fold_input, indent=2) + "\n", encoding="utf-8")

    attention, xla_additions = detect_accelerator()
    mode = effective_mode(args)
    propagation = propagation_flags(args)
    env = os.environ.copy()
    current_xla = env.get("XLA_FLAGS", "")
    for addition in xla_additions:
        if addition not in current_xla:
            current_xla = f"{current_xla} {addition}".strip()
    if current_xla:
        env["XLA_FLAGS"] = current_xla

    command = [
        sys.executable,
        str(args.runner),
        f"--json_path={input_path}",
        "--model=bioemu2",
        "--norun_data_pipeline",
        f"--output_dir={args.output_dir}",
        f"--cache_dir={args.cache_dir}",
        f"--model_dir={precision_weights}",
        f"--weights_precision={args.weights_precision}",
        f"--download_weights={'true' if args.allow_weight_download else 'false'}",
        "--force_output_dir",
        f"--flash_attention_implementation={attention}",
        f"--num_recycles={args.num_recycles}",
        f"--num_diffusion_samples={args.num_diffusion_samples}",
        f"--bioemu2_mode={mode}",
        f"--bioemu2_sampler={args.sampler}",
        f"--bioemu2_physicality={args.physicality}",
        *propagation,
    ]
    if args.msa_mode == "mmseqs2_server":
        command.append("--use_msa_server")

    metadata = {
        "model": "bioemu2",
        "colabfold_runner": "alphafold3-colabfold 3.1.18",
        "job_name": job_name,
        "mode": mode,
        "sampler": args.sampler,
        "physicality": args.physicality,
        "weights_precision": args.weights_precision,
        "msa_mode": args.msa_mode,
        "num_recycles": args.num_recycles,
        "num_diffusion_samples": args.num_diffusion_samples,
        **({
            "propagate_from": args.propagate_from,
            "propagation_steps": args.propagation_steps,
            "lag_ns": args.lag_ns,
        } if propagation else {}),
        "attention": attention,
        "xla_flags": current_xla,
        "command": command,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "fold_input.json").write_text(json.dumps(fold_input, indent=2) + "\n", encoding="utf-8")
    print("Running:", " ".join(command))
    result = subprocess.run(command, env=env, check=False)
    if result.returncode:
        return result.returncode
    job_dir = args.output_dir / job_name
    if not job_dir.is_dir() or not list(job_dir.rglob("*.cif")):
        raise RuntimeError("ColabFold completed without producing a CIF structure")
    (args.output_dir / "job_name.txt").write_text(job_name + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InputError, OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
