import html
import json
import math
import shutil
import time
from collections import defaultdict

from mini_eval.registry import REPORTS_DIR, RESULTS_DIR, SUITES_DIR, load_registry, load_suite


def latest_runs(results_dir=RESULTS_DIR):
    runs = {}
    for meta_path in sorted(results_dir.glob("*/*/*/meta.json")):
        run = meta_path.parent
        runs[(run.parents[1].name, run.parents[0].name)] = run
    return runs


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def metric(results, task, name):
    entry = results["results"].get(task, {})
    if "," in name:
        base, specific = name.split(",", 1)
        return number(entry.get(name)), number(entry.get(f"{base}_stderr,{specific}"))
    return number(entry.get(f"{name},none")), number(entry.get(f"{name}_stderr,none"))


def collect(suite, registry, results_dir=RESULTS_DIR):
    rows = []
    for (model_name, suite_name), run in latest_runs(results_dir).items():
        if suite_name != suite.name or model_name not in registry:
            continue
        results = json.loads((run / "results.json").read_text())
        meta = json.loads((run / "meta.json").read_text())
        model = registry[model_name]
        rows.append(
            {
                "model": model,
                "meta": meta,
                "run": run.name,
                "date": (meta.get("date") or "").replace("T", " "),
                "scores": {t.name: metric(results, t.name, t.primary) for t in suite.tasks},
            }
        )
    order = list(registry)
    rows.sort(key=lambda r: order.index(r["model"].name))
    return rows


def consistency_warnings(rows):
    warnings = []
    for label, getter in [
        ("lm-eval version", lambda m: m["lm_eval"]),
        ("protocol version", lambda m: m["protocol"]),
        ("few-shot count", lambda m: m["suite"]["num_fewshot"]),
        ("example limit", lambda m: m["suite"]["limit"]),
        ("device", lambda m: m["device"]),
    ]:
        values = sorted({str(getter(r["meta"])) for r in rows})
        if len(values) > 1:
            warnings.append(f"Runs use different values of **{label}**: {', '.join(values)}.")
    if any(r["meta"].get("mini_eval", r["meta"].get("mini_llm_eval", {})).get("dirty") is not False for r in rows):
        warnings.append("Some runs come from a modified or untracked working tree.")
    if any(r["meta"]["suite"]["limit"] for r in rows):
        warnings.append("Some runs use only the first examples of each benchmark (limit set).")
    return warnings


def format_cell(value, stderr):
    if value is None:
        return ""
    return f"{value:.3f} ± {stderr:.3f}" if stderr is not None else f"{value:.3f}"


def markdown_table(suite, rows):
    header = ["Model", "Run date", "Non-embedding params", "Tokens seen"] + [f"{t.name} ({t.primary})" for t in suite.tasks]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        m = r["model"]
        cells = [
            m.name,
            r["date"],
            f"{m.non_embedding_params / 1e6:.1f}M",
            f"{m.tokens_seen / 1e9:.2f}B",
        ]
        cells += [format_cell(*r["scores"][t.name]) for t in suite.tasks]
        lines.append("| " + " | ".join(cells) + " |")
    chance = ["Random guessing", "", "", ""] + ["" if t.chance is None else f"{t.chance:.3f}" for t in suite.tasks]
    lines.append("| " + " | ".join(chance) + " |")
    return "\n".join(lines)


def plot_scores(suite, rows, x_attr, xlabel, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in suite.tasks if any(r["scores"][t.name][0] is not None for r in rows)]
    if not tasks:
        return None
    plt.rcParams.update(
        {
            "figure.facecolor": "#060709",
            "axes.facecolor": "#101218",
            "axes.edgecolor": "#1f242e",
            "axes.labelcolor": "#f5f3ec",
            "text.color": "#f5f3ec",
            "xtick.color": "#9a968c",
            "ytick.color": "#9a968c",
            "grid.color": "#1c1f26",
            "legend.facecolor": "#101218",
            "legend.edgecolor": "#1f242e",
            "legend.labelcolor": "#f5f3ec",
        }
    )
    colors = family_colors(rows)
    cols = min(3, len(tasks))
    nrows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 3.6 * nrows), squeeze=False)
    families = defaultdict(list)
    for r in rows:
        families[r["model"].family].append(r)
    for ax, task in zip(axes.flat, tasks):
        for family, members in families.items():
            points = sorted(
                (getattr(r["model"], x_attr), r["scores"][task.name]) for r in members if r["scores"][task.name][0] is not None
            )
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1][0] for p in points]
            errs = [p[1][1] or 0 for p in points]
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=family, color=colors[family])
        if task.chance is not None:
            ax.axhline(task.chance, linestyle="--", color="#d19c00", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(f"{task.name} ({task.primary})", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.5)
    for ax in list(axes.flat)[len(tasks) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="#060709")
    plt.close(fig)
    return out_path


def human_number(n):
    n = float(n)
    for suffix, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div * 0.9:
            return f"{n / div:.4g}{suffix}"
    return f"{n:.4g}"


REFERENCE_RAMP = ["#c7ccd6", "#898f9b", "#5f646d", "#3f444d"]
MINI_COLORS = ["#ffd21d", "#ffe066", "#d19c00", "#cbb418"]


def family_colors(rows):
    """Benchmark-style two-tone DA: mini families in Chloris yellow, references in gray ramp."""
    colors = {}
    reference = 0
    mini = 0
    for r in rows:
        family = r["model"].family
        if family in colors:
            continue
        if r["model"].group == "mini":
            colors[family] = MINI_COLORS[mini % len(MINI_COLORS)]
            mini += 1
        else:
            colors[family] = REFERENCE_RAMP[reference % len(REFERENCE_RAMP)]
            reference += 1
    return colors


def svg_chart(task, rows, x_attr):
    """One SVG scatter (log x) with error bars, a chance line and native tooltips."""
    points = []
    for r in rows:
        value, stderr = r["scores"][task.name]
        if value is None:
            continue
        points.append(
            {
                "family": r["model"].family,
                "name": r["model"].name,
                "x": float(getattr(r["model"], x_attr)),
                "y": value,
                "e": float(stderr or 0.0),
                "date": r["date"],
            }
        )
    if not points:
        return None

    colors = family_colors(rows)
    xmin = min(p["x"] for p in points) * 0.7
    xmax = max(p["x"] for p in points) * 1.4
    yhi = max(p["y"] + p["e"] for p in points) * 1.15

    W, H = 640, 340
    ML, MR, MT, MB = 52, 16, 14, 42
    PW, PH = W - ML - MR, H - MT - MB
    lo, hi = math.log10(xmin), math.log10(xmax)

    def X(x):
        return ML + (math.log10(x) - lo) / (hi - lo) * PW

    def Y(y):
        return MT + (1 - y / yhi) * PH

    els = []

    t = 10 ** math.floor(math.log10(xmin))
    while t <= xmax * 1.05:
        if t >= xmin * 0.95:
            xx = X(t)
            els.append(f'<line x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{H - MB}" stroke="#1f242e" stroke-width="1"/>')
            els.append(
                f'<text x="{xx:.1f}" y="{H - MB + 20}" fill="#9a968c" font-size="11" text-anchor="middle">{human_number(t)}</text>'
            )
        t *= 10
    for i in range(6):
        yv = yhi * i / 5
        yy = Y(yv)
        els.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W - MR}" y2="{yy:.1f}" stroke="#15181d" stroke-width="1"/>')
        els.append(f'<text x="{ML - 8}" y="{yy + 4:.1f}" fill="#9a968c" font-size="11" text-anchor="end">{yv:.2g}</text>')
    if task.chance is not None:
        yy = Y(task.chance)
        els.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W - MR}" y2="{yy:.1f}" stroke="#d19c00" stroke-width="1" stroke-dasharray="5 4"/>')
        els.append(
            f'<text x="{W - MR - 6}" y="{yy - 5:.1f}" fill="#d19c00" font-size="10" text-anchor="end">chance {task.chance:g}</text>'
        )
    families = []
    for p in points:
        if p["family"] not in families:
            families.append(p["family"])
    for family in families:
        color = colors[family]
        members = sorted((p for p in points if p["family"] == family), key=lambda p: p["x"])
        if len(members) > 1:
            path = " ".join(f"{X(p['x']):.1f},{Y(p['y']):.1f}" for p in members)
            els.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-opacity="0.4" stroke-width="1.5"/>')
        for p in members:
            xx, yy = X(p["x"]), Y(p["y"])
            ye = Y(p["y"] + p["e"])
            els.append(f'<line x1="{xx:.1f}" y1="{ye:.1f}" x2="{xx:.1f}" y2="{yy + (yy - ye):.1f}" stroke="{color}" stroke-width="1.5"/>')
            tooltip = html.escape(
                f"{p['name']} — {task.name}: {p['y']:.3f} ± {p['e']:.3f} ({p['date'] or 'date unknown'})"
            )
            els.append(
                f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{color}" stroke="#060709" stroke-width="1.5"><title>{tooltip}</title></circle>'
            )
    title = html.escape(f"{task.name} — {task.primary}")
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">'
        + "".join(els)
        + "</svg>"
    )


def html_report(suite, rows, out_path, stem=None):
    """Self-contained HTML report: Chloris DA (deep black + yellow), one interactive
    chart per task, on two axes (tokens seen and non-embedding parameters). Responsive."""
    legend_chips = []
    colors = family_colors(rows)
    families = []
    for r in rows:
        if r["model"].family not in families:
            families.append(r["model"].family)
    for family in families:
        color = colors[family]
        legend_chips.append(
            f'<span class="chip"><span class="dot" style="background:{color}"></span>{html.escape(family)}</span>'
        )
    legend = "".join(legend_chips)

    def section(x_attr, xlabel):
        cards = []
        for task in suite.tasks:
            chart = svg_chart(task, rows, x_attr)
            if chart is None:
                continue
            note = "lower is better" if task.primary.startswith("bits_per_byte") else ""
            cards.append(
                "<section class='card'>"
                f"<h3>{html.escape(task.name)} <span class='metric'>{html.escape(task.primary)}</span></h3>"
                f"{chart}"
                f"<p class='axis'>{html.escape(xlabel)} (log) {note}</p>"
                "</section>"
            )
        return f"<div class='legend'>{legend}</div><div class='grid'>" + "\n".join(cards) + "</div>"

    md_table = markdown_table(suite, rows)
    md_rows = [line for line in md_table.splitlines() if line.startswith("|")]
    head = [cell.strip() for cell in md_rows[0].strip("|").split("|")]
    body = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in md_rows[2:]
    ]
    table = (
        "<div class='table-scroll'><table><thead><tr>"
        + "".join(f"<th>{html.escape(c)}</th>" for c in head)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>" for cells in body)
        + "</tbody></table></div>"
    )
    meta = rows[0]["meta"]
    generated = time.strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results: {html.escape(suite.name)}</title>
<style>
:root {{ color-scheme: dark; --bg:#060709; --panel:#101218; --panel-2:#131720; --border:#1f242e; --grid:#15181d; --text:#f5f3ec; --muted:#9a968c; --accent:#ffd21d; --accent-soft:#ffe066; --amber:#d19c00; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; -webkit-font-smoothing:antialiased; }}
::selection {{ background:var(--accent); color:var(--bg); }}
a {{ color:var(--accent); }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 20px 64px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; display:inline-block; padding-bottom:10px; border-bottom:3px solid var(--accent); }}
h2 {{ font-size:19px; margin:36px 0 14px; color:var(--text); padding-left:12px; border-left:3px solid var(--accent); }}
h3 {{ font-size:15px; margin:0 0 10px; color:var(--text); }}
code {{ background:#181b21; border:1px solid var(--border); padding:1px 6px; border-radius:5px; color:var(--accent-soft); font-size:12px; }}
.metric {{ color:var(--muted); font-weight:400; font-size:12px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:18px 18px 12px; box-shadow:0 12px 32px rgba(0,0,0,.35); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(min(100%,480px),1fr)); gap:16px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px; margin:0 0 14px; }}
.chip {{ display:inline-flex; align-items:center; gap:7px; background:var(--panel); border:1px solid var(--border); border-radius:999px; padding:4px 12px; font-size:12px; color:#d8d5cc; }}
.dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.axis {{ color:var(--muted); font-size:12px; margin:6px 0 0; }}
.table-scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; background:var(--panel); border:1px solid var(--border); border-radius:14px; box-shadow:0 12px 32px rgba(0,0,0,.35); }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th,td {{ border-bottom:1px solid var(--border); padding:9px 12px; text-align:left; white-space:nowrap; }}
tr:last-child td {{ border-bottom:none; }}
th {{ color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; font-size:10.5px; }}
tr:last-child td {{ color:var(--muted); }}
tbody tr:hover {{ background:var(--panel-2); }}
.meta {{ color:var(--muted); font-size:12.5px; }}
.card p {{ margin:8px 0 0; }}
@media (max-width:640px) {{
  main {{ padding:20px 12px 48px; }}
  h1 {{ font-size:21px; padding-bottom:8px; }}
  h2 {{ font-size:16px; margin:28px 0 12px; padding-left:10px; }}
  h3 {{ font-size:14px; }}
  .card {{ padding:14px 14px 10px; border-radius:12px; }}
  .chip {{ font-size:11px; padding:3px 10px; }}
  table {{ font-size:11.5px; }}
  th,td {{ padding:7px 9px; }}
}}
</style></head><body><main>
<h1>Results: {html.escape(suite.name)}</h1>
<p class="meta">{html.escape(suite.description)}</p>
<p class="meta">Protocol v{html.escape(str(meta.get('protocol', '?')))} · lm-eval {html.escape(str(meta.get('lm_eval', '?')))} · {suite.num_fewshot}-shot · hover a point for model, score and run date · generated {generated}</p>
{table}
<h2>Score against tokens seen</h2>
{section('tokens_seen', 'tokens seen')}
<h2>Score against non-embedding parameters</h2>
{section('non_embedding_params', 'non-embedding parameters')}
<p class="meta">Static PNG versions live next to this file (<code>{html.escape(stem)}_vs_tokens.png</code>, <code>{html.escape(stem)}_vs_params.png</code>); older reports are archived under <code>history/</code>.</p>
</main></body></html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def archive_report(suite_name, reports_dir=REPORTS_DIR, suffixes=("", "_private", "_public")):
    """Move the current report + plots (all views) to reports/history/<suite>/<stamp>/."""
    existing = []
    for suffix in suffixes:
        md = reports_dir / f"{suite_name}{suffix}.md"
        pngs = [reports_dir / f"{suite_name}{suffix}_{kind}.png" for kind in ("vs_tokens", "vs_params")]
        html = reports_dir / f"{suite_name}{suffix}.html"
        existing += [p for p in (md, *pngs, html) if p.exists()]
    if not existing:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(existing[0].stat().st_mtime))
    dest = reports_dir / "history" / suite_name / stamp
    n = 1
    while dest.exists():
        n += 1
        dest = reports_dir / "history" / suite_name / f"{stamp}-{n}"
    dest.mkdir(parents=True)
    for p in existing:
        shutil.move(str(p), dest / p.name)
    return dest


def write_report(suite, rows, reports_dir, stem):
    tokens_png = plot_scores(suite, rows, "tokens_seen", "tokens seen", reports_dir / f"{stem}_vs_tokens.png")
    params_png = plot_scores(
        suite, rows, "non_embedding_params", "non-embedding parameters", reports_dir / f"{stem}_vs_params.png"
    )
    meta = rows[0]["meta"]
    parts = [
        f"# Results: {suite.name}",
        "",
        suite.description,
        "",
        f"Protocol v{meta.get('protocol', '?')}, lm-eval {meta.get('lm_eval', '?')}, {suite.num_fewshot}-shot. "
        "Values are the primary metric of each task with their standard error, from the "
        "latest run of each model.",
        "",
    ]
    warnings = consistency_warnings(rows)
    if warnings:
        parts += ["> **Warnings**", ">"] + [f"> - {w}" for w in warnings] + [""]
    parts += [markdown_table(suite, rows), ""]
    if tokens_png:
        parts += ["## Score against tokens seen", "", f"![]({tokens_png.name})", ""]
    if params_png:
        parts += ["## Score against non-embedding parameters", "", f"![]({params_png.name})", ""]
    parts += [
        f"Previous versions of this report (with their plots) are archived under "
        f"`history/{suite.name}/`, sorted by generation date. An interactive version "
        f"with tooltips lives in [{stem}.html]({stem}.html).",
        "",
    ]
    out = reports_dir / f"{stem}.md"
    out.write_text("\n".join(parts))
    html_report(suite, rows, reports_dir / f"{stem}.html", stem)
    return out


def build_report(suite_name, results_dir=RESULTS_DIR, reports_dir=REPORTS_DIR, registry=None, suites_dir=SUITES_DIR):
    registry = registry or load_registry()
    suite = load_suite(suite_name, suites_dir)
    rows_all = collect(suite, registry, results_dir)
    if not rows_all:
        raise SystemExit(f"No results for suite {suite.name!r}. Run `mini-eval run {suite.name} MODEL` first.")
    reports_dir.mkdir(parents=True, exist_ok=True)
    archive_report(suite.name, reports_dir)
    views = [
        ("private", rows_all),
        ("public", [r for r in rows_all if r["model"].group != "mini"]),
    ]
    written = []
    for suffix, rows in views:
        if rows:
            written.append(write_report(suite, rows, reports_dir, f"{suite.name}_{suffix}"))
    return written[0] if written else None