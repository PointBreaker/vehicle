"""Self-contained HTML report for one or more bench runs.

    uv run python -m blinddrive.report runs/bench-*.json -o runs/compare.html

Runs on the same suite are compared seed by seed. The page has no external
dependencies and can be opened offline or attached to an issue.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CSS = """
:root { --bg:#fafafa; --fg:#1d1f23; --dim:#6b7078; --line:#e3e5e8; --good:#1f8a4c; --bad:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#17191d; --fg:#e6e8eb; --dim:#9aa0a8; --line:#2c3036; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0; padding:24px 16px; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
p.meta { color:var(--dim); margin:0 0 16px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--dim); font-weight:600; }
td.bad { color:var(--bad); } td.good { color:var(--good); }
code { font-size:12px; }
ul { color:var(--dim); padding-left:18px; }
"""


def _fmt(x: Any, digits: int = 1, suffix: str = "") -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{0.0 if abs(x) < 0.5 * 10 ** -digits else x:.{digits}f}{suffix}"
    return f"{x}{suffix}"


def _label(run: dict[str, Any]) -> str:
    parts = [run["controller"]]
    if run.get("model"):
        parts.append(run["model"])
    if run.get("delay_ms"):
        parts.append(f"+{run['delay_ms']:g} ms")
    return " · ".join(parts)


def render(runs: Sequence[dict[str, Any]]) -> str:
    e = html.escape
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
           f"<meta name=viewport content='width=device-width, initial-scale=1'>"
           f"<title>BlindDrive report</title><style>{CSS}</style></head><body><main>"]
    suites = sorted({r.get("suite") or "ad-hoc" for r in runs})
    out.append("<h1>BlindDrive report</h1>")
    out.append(f"<p class=meta>{len(runs)} run(s) · suite: {e(', '.join(suites))} · "
               f"scoring {e(runs[0].get('scoring', '?'))} · score: crude rule = 0, full-vision oracle = 100</p>")

    out.append("<h2>Scorecard</h2><div class=scroll><table><tr><th>run</th><th>score</th><th>95% CI</th>"
               "<th>reference</th><th>finished</th><th>errors</th><th>bumps/ep</th><th>wall s/ep</th>"
               "<th>finish time</th><th>latency p50</th><th>p95</th><th>decisions</th><th>tokens in/out</th></tr>")
    for r in sorted(runs, key=lambda r: -(r["scorecard"]["score"]["mean"] or -1e9)):
        c = r["scorecard"]
        s = c["score"]
        ci = f"{s['ci95'][0]:.1f} – {s['ci95'][1]:.1f}" if s.get("ci95") else "–"
        out.append(
            f"<tr><td>{e(_label(r))}</td><td><b>{_fmt(s['mean'])}</b></td><td>{ci}</td>"
            f"<td>{_fmt(c['reference_score']['mean'])}</td><td>{_fmt(c['finish_rate'] * 100, 0, '%')}</td>"
            f"<td class={'bad' if c['errors'] else ''}>{c['errors']}</td>"
            f"<td>{_fmt(c['edge_hits_per_episode'], 2)}</td><td>{_fmt(c['edge_contact_s_per_episode'], 2)}</td>"
            f"<td>{_fmt(c['mean_finish_time_s'], 2, ' s')}</td>"
            f"<td>{_fmt(c['latency_ms']['p50'], 0, ' ms')}</td><td>{_fmt(c['latency_ms']['p95'], 0, ' ms')}</td>"
            f"<td>{c['decisions']}</td><td>{c['tokens']['input']} / {c['tokens']['output']}</td></tr>")
    out.append("</table></div>")

    # Seed-by-seed scores (same suite runs line up).
    seeds = sorted({row["seed"] for r in runs for row in r["episodes"]})
    out.append("<h2>Per seed</h2><div class=scroll><table><tr><th>seed</th>"
               + "".join(f"<th>{e(_label(r))}</th>" for r in runs) + "</tr>")
    for seed in seeds:
        cells = []
        for r in runs:
            row = next((x for x in r["episodes"] if x["seed"] == seed), None)
            if row is None:
                cells.append("<td>–</td>")
                continue
            cls = "bad" if row["status"] != "finished" else ""
            note = "" if row["status"] == "finished" else f" ({row['status']})"
            bumps = f" · {row['edge_hits']} bump{'s' if row['edge_hits'] != 1 else ''}" if row["edge_hits"] else ""
            cells.append(f"<td class={cls}>{_fmt(row['score'])}{note}{bumps}</td>")
        out.append(f"<tr><td>{seed}</td>{''.join(cells)}</tr>")
    out.append("</table></div>")

    out.append("<h2>Runs</h2><ul>")
    for r in runs:
        errs = [x for x in r["episodes"] if x["status"] == "error"]
        out.append(f"<li><b>{e(_label(r))}</b>: preset {e(r['preset'])}, edge {e(r['edge'])}, "
                   f"harness {e(str(r.get('harness') or '–'))}, fingerprint <code>{e(r.get('fingerprint', ''))}</code>, "
                   f"created {e(r['created'])}, wall {r['wall_s']} s"
                   + (f"; first error: {e(errs[0]['error'])}" if errs else "") + "</li>")
    out.append("</ul><ul><li>Effective time = elapsed time, plus distance not driven at 2 m/s for runs that did "
               "not finish. Score = 100 × (T<sub>crude</sub> − T) / (T<sub>crude</sub> − T<sub>oracle</sub>) per "
               "seed, floored at −100, then averaged. A run with less than 5 m of progress in 20 s ends as "
               "stalled.</li><li>Simulated time pauses while a remote controller thinks: latency is "
               "reported, not scored. Runs with a delay use a fixed reaction delay against zero-delay anchors."
               "</li></ul></main></body></html>")
    return "".join(out)


def write_report(runs: Sequence[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render(runs), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="blinddrive.report", description="Compare bench runs in one HTML page.")
    p.add_argument("runs", nargs="+", help="bench-*.json files")
    p.add_argument("-o", "--out", default="runs/report.html")
    args = p.parse_args(argv)
    runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in args.runs]
    fingerprints = {r.get("fingerprint") for r in runs}
    if len(fingerprints) > 1:
        print("warning: runs use different game configs; per-seed comparison is not like for like", file=sys.stderr)
    print(f"wrote {write_report(runs, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
