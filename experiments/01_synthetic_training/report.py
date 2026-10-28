"""HTML report of one batch: hand-drawn target, proposal before and after retraining.

Usage: uv run python experiments/01_synthetic_training/report.py <batch> "<title>" "<conclusion html>"
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(batch: str, title: str, conclusion: str) -> Path:
    folder = ROOT / "reports" / "synthetic_training" / batch
    results = json.loads((folder / "results.json").read_text())
    rows = []
    for name, r in results.items():
        def metric(key: str) -> str:
            m = r.get(key)
            return f"{m['exact']:.0%} exakt · {m['near']:.0%} ±1 Zelle" if m else "–"

        cells = []
        for kind, label in (("target", "Ziel (handgezeichnet)"), ("before", f"Vorher: {metric('before')}"), ("after", f"Nachher: {metric('after')}")):
            img = folder / f"{name}_{kind}.png"
            body = f'<img src="{img.name}" alt="{html.escape(name)} {kind}">' if img.exists() else "<p class=missing>kein Bild</p>"
            cells.append(f"<figure>{body}<figcaption>{html.escape(label)}</figcaption></figure>")
        ex = f" · Trainingsbeispiel #{r['example_id']}" if "example_id" in r else ""
        rows.append(f"<section><h2>{html.escape(name)}<small>{ex}</small></h2><div class=row>{''.join(cells)}</div></section>")

    def mean(key: str, field: str) -> str:
        values = [r[key][field] for r in results.values() if key in r]
        return f"{sum(values) / len(values):.0%}" if values else "–"

    page = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f8fafc; --fg:#0f172a; --muted:#64748b; --card:#ffffff; --line:#e2e8f0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0b1120; --fg:#e2e8f0; --muted:#94a3b8; --card:#111827; --line:#1f2937; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1400px; margin:auto; }}
h1 {{ margin:0 0 4px; }} h2 {{ font-size:18px; margin:0 0 8px; }} h2 small {{ color:var(--muted); font-weight:400; font-size:13px; }}
.lead {{ color:var(--muted); margin:0 0 20px; }}
.summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:160px; }}
.tile b {{ font-size:24px; display:block; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px; }}
.row {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:12px; }}
figure {{ margin:0; }} img {{ width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:6px; }}
figcaption {{ color:var(--muted); font-size:13px; margin-top:4px; }}
.conclusion {{ background:var(--card); border:1px solid var(--line); border-left:4px solid #2563eb; border-radius:10px; padding:12px 16px; margin-bottom:20px; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="lead">Metrik: Anteil verbundener Bauteilpaare, deren relative Lage und Orientierungen der Handzeichnung entsprechen
(exakt bzw. höchstens eine Rasterzelle daneben). Unabhängig davon, wo die Zeichnung auf dem Blatt liegt.</p>
<div class="summary">
<div class="tile"><b>{len(results)}</b>Schaltungen</div>
<div class="tile"><b>{mean('before', 'exact')}</b>vorher exakt (Mittel)</div>
<div class="tile"><b>{mean('after', 'exact')}</b>nachher exakt (Mittel)</div>
<div class="tile"><b>{mean('after', 'near')}</b>nachher ±1 Zelle (Mittel)</div>
</div>
<div class="conclusion">{conclusion}</div>
{''.join(rows)}
</main></body></html>
"""
    out = folder / "index.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
