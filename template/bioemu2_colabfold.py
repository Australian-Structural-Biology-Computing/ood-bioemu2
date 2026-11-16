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


class InputError(ValueError):
    """Input cannot be represented safely as a fold request."""


def _clean_sequence(value: str) -> str:
    return "".join(ch for ch in value if not ch.isspace() and ch not in {"\ufeff", "\u200b"}).upper()


def _read_fasta(path: Path) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    saw_header = False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
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
            current.append(line)
    if current:
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
        entity: dict = {"id": chain_id, "sequence": sequence, "templates": []}
        if args.msa_mode == "single_sequence":
            entity.update({"unpairedMsa": f">query\n{sequence}\n", "pairedMsa": ""})
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
    """Return notebook-equivalent attention implementation and XLA additions."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        caps = [float(value.strip()) for value in result.stdout.splitlines() if value.strip()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        caps = []
    if not caps:
        raise RuntimeError("BioEmu2 requires an NVIDIA GPU, but none was detected")
    cap = min(caps)
    if cap < 8.0:
        return "xla", ["--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"]
    if 8.0 < cap < 9.0:
        return "xla", ["--xla_gpu_enable_triton_gemm=false"]
    return "triton", ["--xla_gpu_enable_triton_gemm=false"]


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
    result.add_argument("--weights-precision", choices=("int8", "fp16", "fp32"), default="int8")
    result.add_argument("--allow-weight-download", action="store_true")
    return result


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
    precision_weights = args.weights_dir / args.weights_precision
    precision_weights.mkdir(parents=True, exist_ok=True)
    input_path = args.work_dir / f"{job_name}.json"
    input_path.write_text(json.dumps(fold_input, indent=2) + "\n", encoding="utf-8")

    attention, xla_additions = detect_accelerator()
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
        f"--bioemu2_mode={args.mode}",
        f"--bioemu2_sampler={args.sampler}",
        f"--bioemu2_physicality={args.physicality}",
    ]
    if args.msa_mode == "mmseqs2_server":
        command.append("--use_msa_server")

    metadata = {
        "model": "bioemu2",
        "colabfold_runner": "alphafold3-colabfold 3.1.18",
        "job_name": job_name,
        "mode": args.mode,
        "sampler": args.sampler,
        "physicality": args.physicality,
        "weights_precision": args.weights_precision,
        "msa_mode": args.msa_mode,
        "num_recycles": args.num_recycles,
        "num_diffusion_samples": args.num_diffusion_samples,
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
