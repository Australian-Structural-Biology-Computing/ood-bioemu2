import importlib.util
import json
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
        self.assertIn("unpairedMsa", protein)
        self.assertEqual(payload["sequences"][-1]["ligand"]["smiles"], "[Na+]")

    def test_fasta_records_become_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "input.fa"
            fasta.write_text(">one\nACDE\n>two\nFGHI\n", encoding="utf-8")
            payload, _ = MODULE.build_fold_input(arguments(protein=str(fasta), ligand_ccd="", ligand_smiles=""))
        proteins = [entity["protein"] for entity in payload["sequences"]]
        self.assertEqual([item["sequence"] for item in proteins], ["ACDE", "FGHI"])

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
