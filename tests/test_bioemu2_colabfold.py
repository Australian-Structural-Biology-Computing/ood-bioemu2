import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
import warnings
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
    """The report frames output the way the science actually behaves."""

    @staticmethod
    def stage(tmp: str, metadata: dict) -> Path:
        root = Path(tmp)
        job = root / str(metadata.get("job_name", "job"))
        job.mkdir(parents=True)
        (job / "job_summary_confidences.json").write_text(
            json.dumps({"mean_plddt": 0, "ptm": 0}), encoding="utf-8"
        )
        (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return root

    def test_nan_summary_does_not_fake_confidence(self):
        # json round-trips NaN happily; one NaN sample must not suppress the
        # "no confidence head" warning or set confidence_available.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            (job / "job_summary_confidences.json").write_text(
                json.dumps({"mean_plddt": float("nan"), "ptm": float("nan"),
                            "iptm": float("nan"), "ranking_score": float("nan")}),
                encoding="utf-8",
            )
            (job / "sample.cif").write_text("data_test\n", encoding="utf-8")
            document, report = REPORT.build(root)
        self.assertIn("Confidence head unavailable", document)
        self.assertFalse(report["confidence_available"])
        self.assertIsNone(report["summary_metrics"]["mean_plddt"])

    def test_ensemble_mode_shows_ranking_when_scores_are_real(self):
        # csv.DictReader yields strings; a finite score must still be detected.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            (job / "job_seed-1_sample-0_model.cif").write_text("data_test\n", encoding="utf-8")
            (job / "job_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n1,0,0.75\n", encoding="utf-8"
            )
            document, _ = REPORT.build(root)
        self.assertIn("<h2>Ranking</h2>", document)

    def test_ensemble_bundle_is_capped(self):
        try:
            from Bio.PDB import MMCIFIO, PDBParser
            from Bio.PDB.Structure import Structure
            from Bio.PDB.Model import Model
            from Bio.PDB.Chain import Chain
            from Bio.PDB.Residue import Residue
            from Bio.PDB.Atom import Atom
        except ImportError:
            self.skipTest("biopython unavailable on this host")

        def frame_cif(path: Path) -> None:
            structure = Structure("frame")
            model = Model(0)
            chain = Chain("A")
            residue = Residue((" ", 1, " "), "ALA", " ")
            for index, name in enumerate(("N", "CA", "C")):
                residue.add(Atom(name, (index * 1.5, 0.0, 0.0), 1.0, 0.0, " ", name, 1, "N"))
            chain.add(residue)
            model.add(chain)
            structure.add(model)
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(str(path))

        total = REPORT.MAX_BUNDLE_MODELS + 5
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            for step in range(total):
                frame_cif(job / f"job_seed-1_sample-{step}_model.cif")
            document, report = REPORT.build(root)
            self.assertEqual(report["ensemble_bundle_frames"], REPORT.MAX_BUNDLE_MODELS)
            self.assertTrue(report["ensemble_bundle_truncated"])
            self.assertIn(f"first {REPORT.MAX_BUNDLE_MODELS} of {total}", document)
            parser = PDBParser(QUIET=True)
            warnings.filterwarnings("ignore")
            structure = parser.get_structure("b", str(root / report["ensemble_bundle"]))
            self.assertEqual(len(list(structure)), REPORT.MAX_BUNDLE_MODELS)

    def test_report_explains_missing_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            (job / "sample.cif").write_text("data_test\n", encoding="utf-8")
            document, report = REPORT.build(root)
        self.assertIn("Confidence head unavailable", document)
        self.assertFalse(report["confidence_available"])
        self.assertEqual(report["structures"], ["job/sample.cif"])

    def test_structure_mode_keeps_the_ranking_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "structure"})
            job = root / "job"
            (job / "job_seed-1_sample-0_model.cif").write_text("data_test\n", encoding="utf-8")
            (job / "job_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n1,0,nan\n", encoding="utf-8"
            )
            document, report = REPORT.build(root)
        self.assertIn("<h2>Ranking</h2>", document)
        self.assertIn("<h2>Structures</h2>", document)
        self.assertEqual(report["kind"], "structure")

    def test_equilibrium_mode_frames_samples_as_an_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            (job / "job_seed-1_sample-0_model.cif").write_text("data_test\n", encoding="utf-8")
            (job / "job_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n1,0,nan\n", encoding="utf-8"
            )
            document, report = REPORT.build(root)
        self.assertIn("<h2>Ensemble samples</h2>", document)
        self.assertIn("not a time series, and are not ranked", document)
        self.assertNotIn("<h2>Ranking</h2>", document)
        self.assertEqual(report["kind"], "ensemble")

    def test_propagation_mode_labels_time_ordered_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {
                "job_name": "job", "mode": "equilibrium",
                "propagation_steps": 3, "lag_ns": 2.0, "propagate_from": "/x/start.cif",
            })
            job = root / "job"
            for step in (0, 1, 2):
                (job / f"job_seed-1_sample-{step}_model.cif").write_text("data_test\n", encoding="utf-8")
            document, report = REPORT.build(root)
        self.assertIn("<h2>Trajectory frames</h2>", document)
        self.assertIn("step 2, t = 4 ns", document)
        self.assertEqual(report["kind"], "trajectory")

    def test_frames_sort_numerically_past_sample_nine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            for step in (0, 2, 10):
                (job / f"job_seed-1_sample-{step}_model.cif").write_text("data_test\n", encoding="utf-8")
            _, report = REPORT.build(root)
        self.assertEqual(
            [Path(name).name for name in report["structures"]],
            ["job_seed-1_sample-0_model.cif", "job_seed-1_sample-2_model.cif",
             "job_seed-1_sample-10_model.cif"],
        )

    def test_metrics_average_across_sample_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            (job / "job_summary_confidences.json").write_text(
                json.dumps({"mean_plddt": 10}), encoding="utf-8"
            )
            for step, value in ((0, 10), (1, 20)):
                sub = job / f"seed-1_sample-{step}"
                sub.mkdir()
                (sub / f"job_seed-1_sample-{step}_summary_confidences.json").write_text(
                    json.dumps({"mean_plddt": value}), encoding="utf-8"
                )
            _, report = REPORT.build(root)
        # The job-level file repeats the frames; the mean must not double-count.
        self.assertEqual(report["summary_metrics"]["mean_plddt"], 15.0)
        self.assertEqual(report["summary_samples"], 2)

    def test_ensemble_bundle_written_when_biopython_present(self):
        try:
            from Bio.PDB import MMCIFIO, PDBParser
            from Bio.PDB.Structure import Structure
            from Bio.PDB.Model import Model
            from Bio.PDB.Chain import Chain
            from Bio.PDB.Residue import Residue
            from Bio.PDB.Atom import Atom
        except ImportError:
            self.skipTest("biopython unavailable on this host")

        def frame_cif(path: Path) -> None:
            """A valid single-chain mmCIF written by biopython itself."""
            structure = Structure("frame")
            model = Model(0)
            chain = Chain("A")
            residue = Residue((" ", 1, " "), "ALA", " ")
            for index, name in enumerate(("N", "CA", "C")):
                atom = Atom(name, (index * 1.5, 0.0, 0.0), 1.0, 0.0, " ", name, 1, "N")
                residue.add(atom)
            chain.add(residue)
            model.add(chain)
            structure.add(model)
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(str(path))

        with tempfile.TemporaryDirectory() as tmp:
            root = self.stage(tmp, {"job_name": "job", "mode": "equilibrium"})
            job = root / "job"
            for step in (0, 1):
                frame_cif(job / f"job_seed-1_sample-{step}_model.cif")
            document, report = REPORT.build(root)
            self.assertIsNotNone(report["ensemble_bundle"])
            self.assertIn("ensemble.pdb", document)
            bundle = root / report["ensemble_bundle"]
            self.assertEqual(bundle.read_text(encoding="utf-8").count("MODEL"), 2)


if __name__ == "__main__":
    unittest.main()
