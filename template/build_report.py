#!/usr/bin/env python3
"""Build a small static result index for a ColabFold2-preview BioEmu2 run."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from urllib.parse import quote


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def display(value: object) -> str:
    if value is None:
        return "not reported"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def relative_link(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return "/".join(quote(part) for part in relative.split("/"))


def build(root: Path) -> tuple[str, dict]:
    metadata = load_json(root / "run_metadata.json")
    job_name = metadata.get("job_name", "")
    job_dir = root / str(job_name)
    summaries = sorted(job_dir.rglob("*_summary_confidences.json")) if job_dir.is_dir() else []
    summary = load_json(summaries[0]) if summaries else {}
    cifs = sorted(job_dir.rglob("*.cif")) if job_dir.is_dir() else []
    physicality_files = sorted(job_dir.rglob("*physicality*.json")) if job_dir.is_dir() else []
    physicality = [load_json(path) for path in physicality_files]
    ranking_files = sorted(job_dir.rglob("*_ranking_scores.csv")) if job_dir.is_dir() else []
    ranking_rows: list[dict] = []
    if ranking_files:
        try:
            with ranking_files[0].open(newline="", encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))[:100]
        except OSError:
            ranking_rows = []

    metrics = {
        "mean_plddt": summary.get("mean_plddt"),
        "ptm": summary.get("ptm"),
        "iptm": summary.get("iptm"),
        "ranking_score": summary.get("ranking_score"),
    }
    no_confidence = not any(value not in (None, 0, 0.0) for value in metrics.values())
    file_rows = "".join(
        f'<li><a href="{html.escape(relative_link(path, root), quote=True)}">{html.escape(path.relative_to(root).as_posix())}</a></li>'
        for path in cifs
    ) or "<li>No CIF files found.</li>"
    propagation = metadata.get("propagation_steps")
    propagation_row = (
        f"<tr><th>Trajectory</th><td>{html.escape(str(metadata.get('propagation_steps')))} steps "
        f"x {html.escape(str(metadata.get('lag_ns')))} ns per step, propagated from "
        f"{html.escape(str(metadata.get('propagate_from')))}</td></tr>"
        if propagation else ""
    )
    metric_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(display(value))}</td></tr>"
        for label, value in (
            ("Mean pLDDT", metrics["mean_plddt"]),
            ("pTM", metrics["ptm"]),
            ("ipTM", metrics["iptm"]),
            ("Ranking score", metrics["ranking_score"]),
        )
    )
    confidence_note = (
        '<div class="note"><b>Confidence head unavailable:</b> this BioEmu2 port does not predict pLDDT or PAE. '
        "Zero or absent confidence values are not evidence of low confidence. Use the physicality report and inspect the ensemble.</div>"
        if no_confidence else ""
    )
    physicality_note = (
        f"{len(physicality_files)} physicality report file(s) were produced. Failed checks do not remove structures."
        if physicality_files else
        "No separate physicality JSON was found; inspect the job log before using these structures."
    )
    ranking_table = ""
    if ranking_rows:
        columns = list(ranking_rows[0])
        ranking_table = "<h2>Ranking</h2><div class='scroll'><table><thead><tr>" + "".join(
            f"<th>{html.escape(column)}</th>" for column in columns
        ) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
            for row in ranking_rows
        ) + "</tbody></table></div>"

    report = {
        "metadata": metadata,
        "summary": summary,
        "confidence_available": not no_confidence,
        "structures": [path.relative_to(root).as_posix() for path in cifs],
        "physicality_reports": [path.relative_to(root).as_posix() for path in physicality_files],
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
<div class="card"><h2>Model summary</h2>{confidence_note}<table>{metric_rows}</table><p>{html.escape(physicality_note)}</p></div>
<div class="card"><h2>Structures</h2><ul>{file_rows}</ul></div>
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
