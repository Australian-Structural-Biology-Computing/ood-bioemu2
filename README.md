# bioemu2

Open OnDemand (KoD) batch application for the `bioemu2` model in the
[ColabFold2 preview notebook](https://github.com/sokrypton/ColabFold/blob/main/ColabFold2_preview.ipynb).
It follows the deployment and user experience of `kod-proteinfold`: the form queues a
GPU PBS job, the job writes live logs and durable scratch results, and the completed
session links to a static HTML/JSON report.

## What runs

This app deliberately implements the **ColabFold2-preview port**, not Microsoft's native
PyTorch BioEmu2 CLI:

- runtime image: `ghcr.io-jscgh/proteinfold_colabfold-af3-preview-e9e8bd0.img`;
- `alphafold3-colabfold==3.1.18`;
- model: `bioemu2`;
- vendored upstream runner: `template/run_alphafold.py` from the
  `sokrypton/alphafold3` `v3.1.18` tag, pinned by SHA-256;
- converted weights: `sokrypton/af3-any-model`, downloaded by the pinned package;
- optional protein MSA: the public ColabFold MMseqs2 server.

The container already carries the AlphaFold/ColabFold Python stack and Chemical Component
Dictionary tables. Jobs do not install software. Administrators prefetch model weights once;
production jobs run with weight downloads disabled.

BioEmu2 has no confidence head in this port. Zero or absent pLDDT, predicted TM-score
(pTM), interface predicted TM-score (ipTM), or predicted aligned error (PAE) values are
reported as **unavailable**, not as low confidence. Physicality checks are retained, and
failed structures remain in the output.

## User inputs

The form accepts:

- protein sequences directly, as colon-separated chains, or from a FASTA file;
- optional DNA and RNA chains;
- optional ligands by Protein Data Bank Chemical Component Dictionary code or SMILES;
- structure-prediction or equilibrium-ensemble mode;
- ColabFold server or single-sequence mode;
- seeds, recycles, samples per seed, sampler, physicality preset, and stored-weight
  representation.

Identical protein sequences are represented as copies of one entity, matching the notebook.
All input is parsed into an AlphaFold-3-dialect JSON file before inference. The application
never executes input text.

## Output layout

Each run writes to:

```text
/srv/scratch/$USER/bioemu2_output/<sanitised-run-name>_<PBS-job-number>/
```

Important files include:

- `report.html` and `report.json` — static summary and machine-readable index;
- `fold_input.json` — resolved molecular input and seeds;
- `run_metadata.json` — pinned runtime, options, and executed argument vector;
- `<job-name>_<hash>/` — ColabFold result tree with CIF structures, ranking files,
  physicality records, and any model-produced metadata.

The report links individual CIF files and explicitly marks unavailable confidence metrics.
Input sequence text is not written to central metrics or debug metadata. It remains in the
user-owned run directory and bounded failure diagnostics.

## Administrator deployment

### 1. Install the OOD app

Deploy an immutable release and link it into the KoD system-app directory using the same
release process as `ood-proteinfold`. The application tile is declared by `manifest.yml` as
a Batch Connect app under **Interactive Apps → Biology**.

The deployed tree must not be writable by ordinary users. The PBS job executes the scripts
copied by Open OnDemand into its private session directory.

### 2. Confirm the pinned image

The default image is:

```text
/srv/scratch/sbf-pipelines/containers/ghcr.io-jscgh-proteinfold_colabfold-af3-preview-e9e8bd0.img
```

It must be readable on the OOD host and compute nodes. Override `BIOEMU2_IMAGE` in
`template/.env` only with a validated replacement. Revalidate package, runner, GPU, and
output compatibility before changing it.

### 3. Prefetch weights

Run once as an administrator on a node with internet access:

```bash
scripts/setup_runtime.sh int8
```

The default destination is `/srv/scratch/sbf-pipelines/bioemu2/weights/int8`. Repeat with
`fp16` or `fp32` only if those form choices will be offered. Each representation has its own
directory so the upstream loader cannot silently reuse a different representation.

Keep this production setting:

```bash
BIOEMU2_ALLOW_WEIGHT_DOWNLOAD=false
```

This prevents simultaneous user jobs from modifying shared runtime state or independently
downloading large weights.

### 4. Validate on a GPU

No inference is run on a login node. Queue a one-GPU allocation equivalent to
`submit.yml.erb`, then run:

```bash
scripts/smoke_gpu.sh
```

The smoke fixture is an explicitly synthetic sequence containing each standard amino-acid
letter once; it is not a claimed biological protein. The script uses one seed, one recycle,
one sample, and no MSA search. Confirm it creates a CIF and `report.html`.

### 5. Canary through KoD

Submit one job from the deployed form and verify:

1. the job requests one `mwacgpu100=gpu-mwac` GPU under `kod_proteinfold`;
2. the live log names the pinned image and runner;
3. the result directory includes the resolved input, CIF output, report, and physicality
   information;
4. the completed-session link opens through the OOD file browser;
5. central metrics contain operational settings but no molecular input.

## Configuration

Production defaults live in `template/.env`:

| Variable | Purpose |
| --- | --- |
| `BIOEMU2_IMAGE` | Pinned Apptainer image |
| `BIOEMU2_WEIGHTS_ROOT` | Administrator-managed converted-weight root |
| `BIOEMU2_CACHE_ROOT` | Per-user XLA/MSA cache |
| `BIOEMU2_ALLOW_WEIGHT_DOWNLOAD` | Permit missing weight downloads; false in production |
| `BASE_OUT_DIR` | Durable user results |
| `BASE_WORK_DIR` | Per-user temporary run state |
| `BASE_DEBUGDIR` | Group-readable bounded failure diagnostics |
| `METRICS_DIRECTORY` | Operational usage metrics |
| `ENABLE_METRICS` | Enable metrics JSON |
| `EXPECTED_RUNNER_SHA256` | Integrity pin for the vendored upstream runner |

The submit template currently requests one GPU, eight CPUs, 124 GB RAM, and 48 hours. Site
administrators should adjust those requests only after representative GPU memory and runtime
testing.

## Validation and tests

Run the CPU-only checks on a login/development host:

```bash
ruby scripts/validate_ood_app.rb
python3 -m unittest discover -s tests -v
```

The validator renders the OOD templates, parses YAML, syntax-checks shell and Python files,
and verifies the upstream runner checksum. Unit tests cover FASTA/direct input, chain
construction, ligand parsing, seed bounds, validation failures, and report behaviour when
confidence values are unavailable.

These checks do not perform inference. Use `scripts/smoke_gpu.sh` in an allocated GPU job for
the end-to-end test.

## Provenance and licences

- ColabFold source: <https://github.com/sokrypton/ColabFold>
- ColabFold2 preview notebook: <https://github.com/sokrypton/ColabFold/blob/main/ColabFold2_preview.ipynb>
- BioEmu2 source: <https://github.com/microsoft/bioemu2>
- Ported model weights: <https://huggingface.co/sokrypton/af3-any-model>

The application code is separate from the licences governing the runtime image, upstream
code, dependencies, and model weights. Preserve their notices and the `TERMS_OF_USE.md`
written by the model runtime. Users are responsible for reviewing those terms and citing
BioEmu2 and ColabFold appropriately.
