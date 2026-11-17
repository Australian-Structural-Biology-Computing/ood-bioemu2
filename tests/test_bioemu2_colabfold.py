import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bioemu2_colabfold", ROOT / "template" / "bioemu2_colabfold.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
REPORT_SPEC = importlib.util.spec_from_file_location("build_report", ROOT / "template" / "build_report.py")
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
assert REPORT_SPEC.loader
REPORT_SPEC.loader.exec_module(REPORT)


def arguments(**overrides):
    values = dict(
        protein="ACDE:ACDE",
        dna="",
        rna="",
        ligand_ccd="ATP,MG",
        ligand_smiles="C[C@@H](O)C\n[Na+]",
        seeds="1,2,2",
        msa_mode="single_sequence",
        job_name="My run",
        mode="equilibrium",
        propagate_from="",
        propagation_steps=1,
        lag_ns=1.0,
    )
    values.update(overrides)
    return Namespace(**values)


class InputTests(unittest.TestCase):
    def test_builds_af3_input_and_merges_identical_proteins(self):
        payload, name = MODULE.build_fold_input(arguments())
        self.assertRegex(name, r"^myrun_[0-9a-f]{7}$")
        self.assertEqual(payload["modelSeeds"], [1, 2])
        protein = payload["sequences"][0]["protein"]
        self.assertEqual(protein["id"], ["A", "B"])
        # Single-sequence mode: every field featurisation validates must be
        # present. An empty pairedMsa featurises as a query-only MSA, which is
        # the single-sequence form (missing fields are hard errors upstream).
        self.assertIn("unpairedMsa", protein)
        self.assertEqual(protein["pairedMsa"], "")
        self.assertEqual(protein["templates"], [])
        self.assertEqual(payload["sequences"][-1]["ligand"]["smiles"], "[Na+]")

    def test_server_msa_mode_leaves_msa_fields_to_the_runner(self):
        payload, _ = MODULE.build_fold_input(arguments(msa_mode="mmseqs2_server"))
        protein = payload["sequences"][0]["protein"]
        self.assertNotIn("unpairedMsa", protein)
        self.assertNotIn("pairedMsa", protein)
        # Templates must still be present: validation rejects an absent field.
        self.assertEqual(protein["templates"], [])

    def test_fasta_records_become_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "input.fa"
            fasta.write_text(">one\nACDE\n>two\nFGHI\n", encoding="utf-8")
            payload, _ = MODULE.build_fold_input(arguments(protein=str(fasta), ligand_ccd="", ligand_smiles=""))
        proteins = [entity["protein"] for entity in payload["sequences"]]
        self.assertEqual([item["sequence"] for item in proteins], ["ACDE", "FGHI"])

    def test_fasta_headerless_file_is_a_bare_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "input.fa"
            fasta.write_text("ACDE\nFGHI\n", encoding="utf-8")
            payload, _ = MODULE.build_fold_input(arguments(protein=str(fasta), ligand_ccd="", ligand_smiles=""))
        proteins = [entity["protein"] for entity in payload["sequences"]]
        self.assertEqual([item["sequence"] for item in proteins], ["ACDEFGHI"])

    def test_fasta_data_before_the_first_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "input.fa"
            fasta.write_text("ACDE\n>one\nFGHI\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.InputError, "before the first '>' header"):
                MODULE.build_fold_input(arguments(protein=str(fasta)))

    def test_fasta_trailing_empty_record_is_rejected(self):
        # A record dropped here used to shift every later chain id silently.
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "input.fa"
            fasta.write_text(">one\nACDE\n>two\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.InputError, "empty FASTA record"):
                MODULE.build_fold_input(arguments(protein=str(fasta)))

    def test_rejects_invalid_polymer(self):
        with self.assertRaisesRegex(MODULE.InputError, "unsupported"):
            MODULE.build_fold_input(arguments(protein="ACD-J"))

    def test_requires_molecule(self):
        with self.assertRaisesRegex(MODULE.InputError, "at least one"):
            MODULE.build_fold_input(arguments(protein="", ligand_ccd="", ligand_smiles=""))

    def test_seed_limit(self):
        with self.assertRaisesRegex(MODULE.InputError, "at most"):
            MODULE.parse_seeds(",".join(map(str, range(33))))

    def test_atom_mapped_smiles_colon_is_preserved(self):
        value = "[CH3:1][OH:2]"
        self.assertEqual(MODULE.parse_smiles(value), [value])


class GateConsistencyTests(unittest.TestCase):
    """The shell gate and this module must describe the same published blobs."""

    LIB = ROOT / "template" / "bioemu2_weights_lib.sh"

    def shell(self, source: str, *, allow_failure: bool = False) -> str:
        result = subprocess.run(
            ["bash", "-c", f'set -uo pipefail; . "{self.LIB}"; {source}'],
            capture_output=True,
            text=True,
            check=not allow_failure,
        )
        return result.stdout.strip()

    def test_precisions_match_the_shell_library(self):
        listed = self.shell('printf "%s" "$BIOEMU2_PRECISIONS"')
        self.assertEqual(MODULE.BIOEMU2_PRECISIONS, tuple(listed.split()))

    def test_expected_filenames_match_the_shell_library(self):
        for precision in MODULE.BIOEMU2_PRECISIONS:
            with self.subTest(precision=precision):
                published = re.compile(r"^bioemu2(\.[a-z0-9]+)?\.bin\.zst$")
                self.assertRegex(self.shell(f'bioemu2_weights_filename {precision}'), published)
                self.assertEqual(
                    self.shell(f'bioemu2_expected_weights /weights {precision}'),
                    f"/weights/{precision}/{self.shell(f'bioemu2_weights_filename {precision}')}",
                )

    def test_rejects_precision_outside_the_form_choices(self):
        self.assertEqual(self.shell('bioemu2_valid_precision fp32; echo $?'), "0")
        self.assertEqual(self.shell('bioemu2_valid_precision bf16; echo $?'), "1")

    def gate(self, root, precision: str) -> tuple[str, str]:
        out = self.shell(
            'r=; f=; bioemu2_gate_weights %s %s r f; printf "%s\\n%s\\n" "$?" "$r"'
            % (shlex.quote(str(root)), shlex.quote(str(precision)), "%s", "%s")
        )
        code, reason = out.split("\n", 1)
        return code, reason

    def test_gate_classifies_shared_state_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = root / "int8" / "bioemu2.int8.bin.zst"
            wanted.parent.mkdir(parents=True)
            self.assertEqual(self.gate(root, "int8")[0], "1")
            wanted.write_bytes(b"x" * 128)
            self.assertIn("incomplete", self.gate(root, "int8")[1])
            wanted.unlink()
            (root / "int8" / "bioemu2.fp16.bin.zst").write_bytes(b"x" * 200_000_000)
            self.assertEqual(self.gate(root, "int8")[0], "2")
            self.assertIn("not the published name", self.gate(root, "int8")[1])
            self.assertTrue(self.gate(root, "not-a-precision")[1].startswith("unsupported precision"))
            wanted.write_bytes(b"x" * 200_000_000)
            self.assertEqual(self.gate(root, "int8")[0], "2")
            self.assertIn("2 *.bin.zst files", self.gate(root, "int8")[1])
            (root / "int8" / "bioemu2.fp16.bin.zst").unlink()
            self.assertEqual(self.gate(root, "int8"), ("0", "ok"))

    def test_fp32_isolates_its_uninfixed_blob(self):
        # Upstream names the fp32 release 'bioemu2.bin.zst' with no precision
        # infix, so it must not share a directory with a quantised blob: the
        # loader globs the directory and would pick either one.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fp32").mkdir()
            self.assertEqual(self.gate(root, "fp32")[0], "1")
            (root / "fp32" / "bioemu2.bin.zst").write_bytes(b"x" * 200_000_000)
            self.assertEqual(self.gate(root, "fp32"), ("0", "ok"))
            (root / "fp32" / "bioemu2.int8.bin.zst").write_bytes(b"x" * 200_000_000)
            self.assertEqual(self.gate(root, "fp32")[0], "2")

    def test_installed_int8_weights_pass_the_gate(self):
        root = Path("/srv/scratch/sbf-pipelines/bioemu2/weights")
        if not (root / "int8").is_dir():  # skip on hosts without the shared install
            self.skipTest(f"no installed weights at {root}")
        self.assertEqual(self.gate(root, "int8"), ("0", "ok"))

    def test_download_root_isolates_the_private_copy(self):
        # A job may never write into the shared root, so the permitted-download
        # path gates a private directory instead of the shared one.
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            private = Path(tmp) / "work" / "weights"
            (shared / "int8").mkdir(parents=True)
            (private / "int8").mkdir(parents=True)
            self.assertEqual(self.gate(shared, "int8")[0], "1")
            self.assertEqual(self.gate(private, "int8")[0], "1")
            (private / "int8" / "bioemu2.int8.bin.zst").write_bytes(b"x" * 200_000_000)
            self.assertEqual(self.gate(private, "int8")[0], "0")
            self.assertEqual(self.gate(shared, "int8")[0], "1")


class AcceleratorTests(unittest.TestCase):
    """Attention selection is delegated to the pinned fork's own policy."""

    def setUp(self):
        self._saved = {name: sys.modules.get(name) for name in (
            "alphafold3", "alphafold3.model", "alphafold3.model.components",
            "alphafold3.model.components.platform",
        )}

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def install_platform(self, device: str, attention: str, flags: list[str]):
        import types
        platform = types.ModuleType("alphafold3.model.components.platform")
        platform.detect_device = lambda: (device, 8.0 if device == "gpu" else None)
        platform.attention_config = lambda device=None, cap=None: {
            "attention": attention, "xla_flags": flags, "nojit": device == "cpu",
        }
        components = types.ModuleType("alphafold3.model.components")
        components.platform = platform
        model = types.ModuleType("alphafold3.model")
        model.components = components
        af3 = types.ModuleType("alphafold3")
        af3.model = model
        sys.modules.update({
            "alphafold3": af3,
            "alphafold3.model": model,
            "alphafold3.model.components": components,
            "alphafold3.model.components.platform": platform,
        })

    def test_delegates_to_the_fork_policy(self):
        self.install_platform("gpu", "volta", ["--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"])
        self.assertEqual(
            MODULE.detect_accelerator(),
            ("volta", ["--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"]),
        )

    def test_without_a_gpu_the_run_is_refused(self):
        self.install_platform("cpu", "xla", [])
        with self.assertRaisesRegex(RuntimeError, "NVIDIA GPU"):
            MODULE.detect_accelerator()


class PropagationTests(unittest.TestCase):
    """Trajectory propagation mirrors upstream's contract exactly."""

    def test_no_start_structure_emits_no_flags(self):
        self.assertEqual(MODULE.propagation_flags(arguments(propagate_from="")), [])

    def test_valid_request_builds_the_three_runner_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "start.cif"
            cif.write_text("data_x\n", encoding="utf-8")
            args = arguments(propagate_from=str(cif), propagation_steps=5, lag_ns=2.5)
            self.assertEqual(
                MODULE.propagation_flags(args),
                [
                    f"--bioemu2_propagate_from={cif}",
                    "--bioemu2_propagation_steps=5",
                    "--bioemu2_lag_ns=2.5",
                ],
            )
            self.assertEqual(MODULE.effective_mode(args), "equilibrium")

    def test_structure_mode_with_a_start_structure_is_rejected(self):
        # Upstream rejects BIOEMU2_MODE=structure with BIOEMU2_X0 set; the app
        # raises the same contract before a GPU allocation is spent.
        args = arguments(propagate_from="/tmp/whatever.cif", mode="structure")
        with self.assertRaisesRegex(MODULE.InputError, "equilibrium"):
            MODULE.effective_mode(args)

    def test_steps_and_lag_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "start.cif"
            cif.write_text("data_x\n", encoding="utf-8")
            for steps in (0, 101, None):
                with self.assertRaises(MODULE.InputError):
                    MODULE.propagation_flags(arguments(propagate_from=str(cif), propagation_steps=steps))
            for lag in (0.0, -1.0, float("nan")):
                with self.assertRaises(MODULE.InputError):
                    MODULE.propagation_flags(arguments(propagate_from=str(cif), lag_ns=lag))

    def test_missing_start_structure_fails_before_inference(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            MODULE.propagation_flags(arguments(propagate_from="/nonexistent/x.cif"))


class ReportTests(unittest.TestCase):
    def test_report_explains_missing_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / "job_abc"
            job.mkdir()
            (job / "sample.cif").write_text("data_test\n", encoding="utf-8")
            (job / "job_summary_confidences.json").write_text(
                json.dumps({"mean_plddt": 0, "ptm": 0}), encoding="utf-8"
            )
            (root / "run_metadata.json").write_text(
                json.dumps({"job_name": "job_abc", "mode": "equilibrium"}), encoding="utf-8"
            )
            document, report = REPORT.build(root)
        self.assertIn("Confidence head unavailable", document)
        self.assertFalse(report["confidence_available"])
        self.assertEqual(report["structures"], ["job_abc/sample.cif"])


if __name__ == "__main__":
    unittest.main()
