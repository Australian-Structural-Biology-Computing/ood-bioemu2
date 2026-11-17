#!/usr/bin/env python3
"""Build a small static result index for a ColabFold2-preview BioEmu2 run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

SAMPLE_RE = re.compile(r"seed-(\d+)_sample-(\d+)")
METRIC_KEYS = ("mean_plddt", "ptm", "iptm", "ranking_score")
# The bundle is a convenience, not the archive: bounding it keeps report
# generation cheap at the form's maximum (32 seeds x 100 samples) and keeps
# the PDB's 5-digit atom serial from overflowing.
MAX_BUNDLE_MODELS = 200


def load_json(path: Path):
    """Any JSON value -- lists included. The job-level physicality file is a
    LIST of per-frame entries, and the dict-only loader used to silently turn
    it into {}, losing every frame's verdict."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def load_json_dict(path: Path) -> dict:
    value = load_json(path)
    return value if isinstance(value, dict) else {}


def display(value: object) -> str:
    if value is None:
        return "not reported"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def relative_link(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return "/".join(quote(part) for part in relative.split("/"))


def sample_order(path: Path) -> tuple[int, int, str]:
    """Sort key that orders frames numerically: sample-2 before sample-10.

    Lexicographic sorting puts sample-10 before sample-2, which for a
    propagated trajectory presents time backwards.
    """
    match = SAMPLE_RE.search(str(path))
    if match:
        return (int(match.group(1)), int(match.group(2)), str(path))
    return (1 << 30, 0, str(path))  # job-level files last


def aggregate_metric(summaries: list[dict], key: str) -> float | None:
    """Mean of the finite numeric values one metric takes across samples.

    Averaging matters once a run produces many frames; absent values stay
    absent. NaN is absent too: json round-trips it happily, and a single NaN
    sample would otherwise poison the mean and flip the confidence logic.
    """
    numbers = [value for value in (summary.get(key) for summary in summaries)
               if isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value)]
    return sum(numbers) / len(numbers) if numbers else None


def write_ensemble_bundle(root: Path, job_name: str, frames: list[Path]) -> tuple[Path, int, bool] | None:
    """One multi-model PDB holding the frames, for viewers that load a whole
    ensemble or trajectory at once. Uses the container's biopython; best
    effort by contract -- returns None on any problem and never lets an
    exception reach the report. Returns the path, the number of models
    actually bundled (capped, see MAX_BUNDLE_MODELS), and whether the cap cut
    content short. Truncation is about the CAP, not about unreadable frames:
    a skipped frame is not "truncated", and multi-model CIFs can fill the cap
    with fewer frames than models.
    """
    if not frames:
        return None
    try:
        import copy

        from Bio.PDB import MMCIFParser, PDBIO
        from Bio.PDB.Structure import Structure

        parser = MMCIFParser(QUIET=True)
        combined = Structure(id="ensemble")
        model_id = 0
        truncated = False
        for frame in frames:
            if model_id >= MAX_BUNDLE_MODELS:
                truncated = True  # frames remain unattempted
                break
            try:
                structure = parser.get_structure("frame", str(frame))
                for model in structure:
                    if model_id >= MAX_BUNDLE_MODELS:
                        truncated = True  # models of this frame dropped
                        break
                    clone = copy.deepcopy(model)
                    clone.id = model_id
                    clone.parent = combined
                    combined.add(clone)
                    model_id += 1
            except Exception:
                continue  # one unreadable frame must not sink the bundle
        if model_id == 0:
            return None
        out = root / f"{job_name}_ensemble.pdb"
        io = PDBIO()
        io.set_structure(combined)
        io.save(str(out))
        return out, model_id, truncated
    except Exception:
        # Memory pressure, a biopython defect, a full disk: the report is the
        # deliverable, the bundle is a convenience.
        return None


def frame_physicality(entries: list[tuple[Path, object]]) -> dict[tuple[int, int], dict]:
    """Per-frame physicality verdicts, keyed by (seed, sample).

    Per-sample files hold one dict whose FILENAME carries the frame ids (the
    dict itself has no seed/sample keys); the job-level file holds a list of
    entries that do. The same verdict arriving twice is fine -- the map
    deduplicates -- and entries with no derivable frame are skipped.
    """
    keyed: dict[tuple[int, int], dict] = {}
    for path, data in entries:
        items = data if isinstance(data, list) else [data]
        for entry in items:
            if not isinstance(entry, dict):
                continue
            seed, sample = entry.get("seed"), entry.get("sample")
            if not (isinstance(seed, int) and isinstance(sample, int)) and path is not None:
                match = SAMPLE_RE.search(str(path))
                if match:
                    seed, sample = int(match.group(1)), int(match.group(2))
            if isinstance(seed, int) and isinstance(sample, int):
                keyed[(seed, sample)] = entry
    return keyed


def build(root: Path) -> tuple[str, dict]:
    metadata = load_json_dict(root / "run_metadata.json")
    job_name = metadata.get("job_name", "")
    job_dir = root / str(job_name)
    mode = metadata.get("mode", "structure")
    propagating = bool(metadata.get("propagation_steps"))
    kind = "trajectory" if propagating else ("ensemble" if mode == "equilibrium" else "structure")

    summaries = sorted(job_dir.rglob("*_summary_confidences.json")) if job_dir.is_dir() else []
    summaries_data = [load_json_dict(path) for path in summaries]
    summary = summaries_data[0] if summaries_data else {}
    # Aggregate the per-sample summaries; the job-level file covers the same
    # frames again and would double-count them in the mean. It stays the
    # raw summary in report.json for compatibility.
    sample_summaries = [data for path, data in zip(summaries, summaries_data)
                        if SAMPLE_RE.search(str(path))] or summaries_data
    metrics = {key: aggregate_metric(sample_summaries, key) for key in METRIC_KEYS}

    cifs = sorted(job_dir.rglob("*.cif"), key=sample_order) if job_dir.is_dir() else []
    # The bundle holds the per-sample frames; the job-level CIF duplicates one
    # of them and would repeat a model.
    frames = [path for path in cifs if SAMPLE_RE.search(str(path))] or cifs
    bundle_result = write_ensemble_bundle(root, str(job_name), frames)
    bundle, bundle_models, bundle_truncated = bundle_result if bundle_result else (None, 0, False)

    physicality_files = sorted(job_dir.rglob("*physicality*.json")) if job_dir.is_dir() else []
    physicality = [(path, load_json(path)) for path in physicality_files]
    physicality_by_frame = frame_physicality(physicality)
    physicality_passed = sum(1 for entry in physicality_by_frame.values() if entry.get("status") == "passed")
    physicality_failed = len(physicality_by_frame) - physicality_passed
    ranking_files = sorted(job_dir.rglob("*_ranking_scores.csv")) if job_dir.is_dir() else []
    ranking_rows: list[dict] = []
    if ranking_files:
        try:
            with ranking_files[0].open(newline="", encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))[:100]
        except OSError:
            ranking_rows = []

    # NaN must count as absent: json round-trips it, and a single NaN sample
    # would otherwise suppress the "no confidence head" warning.
    def meaningful(value: object) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value != 0)

    no_confidence = not any(meaningful(value) for value in metrics.values())

    lag_ns = metadata.get("lag_ns") or 0
    frame_rows = []
    for path in cifs:
        label = path.relative_to(root).as_posix()
        verdict = ""
        match = SAMPLE_RE.search(label)
        if match:
            if propagating:
                step = int(match.group(2))
                label += f" — step {step}, t = {step * float(lag_ns):g} ns"
            entry = physicality_by_frame.get((int(match.group(1)), int(match.group(2))))
            if entry:
                status = str(entry.get("status", "unknown"))
                verdict = f" · physicality: {status}"
                if status != "passed" and entry.get("reason"):
                    verdict += f" ({entry.get('reason')})"
        frame_rows.append(
            f'<li><a href="{html.escape(relative_link(path, root), quote=True)}">{html.escape(label)}</a>'
            f'{html.escape(verdict)}</li>'
        )
    file_rows = "".join(frame_rows) or "<li>No CIF files found.</li>"

    heading = {
        "trajectory": "Trajectory frames",
        "ensemble": "Ensemble samples",
    }.get(kind, "Structures")
    framing_note = {
        "trajectory": "Frames are ordered in time, one per propagation step; this is a sampled trajectory, not a physically integrated dynamics run.",
        "ensemble": "Independent samples of the equilibrium ensemble. They are equiprobable draws, not a time series, and are not ranked.",
    }.get(kind)

    propagation = metadata.get("propagation_steps")
    propagation_row = (
        f"<tr><th>Trajectory</th><td>{html.escape(str(metadata.get('propagation_steps')))} steps "
        f"x {html.escape(str(metadata.get('lag_ns')))} ns per step, propagated from "
        f"{html.escape(str(metadata.get('propagate_from')))}</td></tr>"
        if propagation else ""
    )
    summary_heading = "Model summary"
    if len(sample_summaries) > 1:
        summary_heading += f" (mean over {len(sample_summaries)} samples)"
    metric_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(display(metrics[key]))}</td></tr>"
        for label, key in (
            ("Mean pLDDT", "mean_plddt"),
            ("pTM", "ptm"),
            ("ipTM", "iptm"),
            ("Ranking score", "ranking_score"),
        )
    )
    confidence_note = (
        '<div class="note"><b>Confidence head unavailable:</b> this BioEmu2 port does not predict pLDDT or PAE. '
        "Zero or absent confidence values are not evidence of low confidence. Use the physicality report and inspect the ensemble.</div>"
        if no_confidence else ""
    )
    physicality_note = (
        (f"{physicality_passed} of {len(physicality_by_frame)} frames passed the "
         f"{html.escape(str(metadata.get('physicality') or 'high_quality'))} physicality screen"
         + (f"; {physicality_failed} failed but remain in the output." if physicality_failed else "."))
        if physicality_by_frame else
        "No separate physicality JSON was found; inspect the job log before using these structures."
    )

    # Ranking is a structure-prediction concept. Ensemble draws are
    # equiprobable and trajectory frames are time-ordered. csv.DictReader
    # yields strings, so a numeric score has to be parsed before it counts.
    def finite_score(row: dict) -> bool:
        try:
            return math.isfinite(float(row.get("ranking_score", "")))
        except (TypeError, ValueError):
            return False

    any_score = any(finite_score(row) for row in ranking_rows)
    ranking_table = ""
    if ranking_rows and (kind == "structure" or any_score):
        columns = list(ranking_rows[0])
        ranking_table = "<h2>Ranking</h2><div class='scroll'><table><thead><tr>" + "".join(
            f"<th>{html.escape(column)}</th>" for column in columns
        ) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
            for row in ranking_rows
        ) + "</tbody></table></div>"

    bundle_row = ""
    if bundle:
        truncated = f" (first {bundle_models} of {len(frames)} frames)" if bundle_truncated else ""
        bundle_row = (
            f'<li><a href="{html.escape(relative_link(bundle, root), quote=True)}">{html.escape(bundle.name)}</a>'
            f" — all frames as one multi-model file{html.escape(truncated)}</li>"
        )
    report = {
        "metadata": metadata,
        "summary": summary,
        "summary_metrics": metrics,
        "summary_samples": len(sample_summaries),
        "kind": kind,
        "confidence_available": not no_confidence,
        "structures": [path.relative_to(root).as_posix() for path in cifs],
        "ensemble_bundle": bundle.relative_to(root).as_posix() if bundle else None,
        "ensemble_bundle_frames": bundle_models if bundle else 0,
        "ensemble_bundle_truncated": bundle_truncated,
        "physicality_reports": [path.relative_to(root).as_posix() for path in physicality_files],
        "physicality_summary": {
            "passed": physicality_passed,
            "failed": physicality_failed,
            "frames": len(physicality_by_frame),
        },
        "ranking": ranking_rows,
    }
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>BioEmu2 results</title>
<style>body{{margin:0;background:#f4f7f9;color:#172033;font:16px/1.5 system-ui}}main{{max-width:960px;margin:3rem auto;padding:0 1rem}}.card{{background:white;border:1px solid #dbe4ee;border-radius:16px;padding:1.5rem;box-shadow:0 8px 30px #0f172a12;margin:1rem 0}}h1,h2{{line-height:1.15}}.muted{{color:#64748b}}.note{{padding:1rem;border-left:4px solid #d97706;background:#fffbeb}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #e2e8f0}}.scroll{{overflow:auto}}code{{background:#eef2f7;padding:.15rem .35rem;border-radius:4px}}a{{color:#0369a1}}</style>
</head><body><main><p class="muted">ColabFold2 preview · BioEmu2</p><h1>{html.escape(str(job_name))}</h1>
<div class="card"><h2>Run</h2><table>
<tr><th>Task</th><td>{html.escape(display(metadata.get('mode')))}</td></tr>{propagation_row}
<tr><th>Sampler</th><td>{html.escape(display(metadata.get('sampler')))}</td></tr>
<tr><th>MSA</th><td>{html.escape(display(metadata.get('msa_mode')))}</td></tr>
<tr><th>Stored weights</th><td>{html.escape(display(metadata.get('weights_precision')))}</td></tr>
<tr><th>Structures</th><td>{len(cifs)}</td></tr></table></div>
<div class="card"><h2>{html.escape(summary_heading)}</h2>{confidence_note}<table>{metric_rows}</table><p>{html.escape(physicality_note)}</p></div>
<div class="card"><h2>{html.escape(heading)}</h2>{f'<p>{html.escape(framing_note)}</p>' if framing_note else ''}<ul>{file_rows}{bundle_row}</ul></div>
<div class="card">{ranking_table}<p><a href="report.json">Machine-readable report</a> · <a href="fold_input.json">Resolved fold input</a> · <a href="run_metadata.json">Run metadata</a></p></div>
</main></body></html>"""
    return document, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    document, report = build(args.output_dir)
    (args.output_dir / "report.html").write_text(document, encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
