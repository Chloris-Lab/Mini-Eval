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
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=family)
        if task.chance is not None:
            ax.axhline(task.chance, linestyle="--", color="gray", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(f"{task.name} ({task.primary})", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(tasks) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def human_number(n):
    n = float(n)
    for suffix, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div * 0.9:
            return f"{n / div:.4g}{suffix}"
    return f"{n:.4g}"


FAMILY_COLORS = ["#4ea1ff", "#f39c6b", "#7bd88f", "#c07bf5", "#f5c15c", "#ff7b7b", "#6fe0c8"]


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
            els.append(f'<line x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{H - MB}" stroke="#232c38" stroke-width="1"/>')
            els.append(
                f'<text x="{xx:.1f}" y="{H - MB + 20}" fill="#8aa0b4" font-size="11" text-anchor="middle">{human_number(t)}</text>'
            )
        t *= 10
    for i in range(6):
        yv = yhi * i / 5
        yy = Y(yv)
        els.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W - MR}" y2="{yy:.1f}" stroke="#1a222c" stroke-width="1"/>')
        els.append(f'<text x="{ML - 8}" y="{yy + 4:.1f}" fill="#8aa0b4" font-size="11" text-anchor="end">{yv:.2g}</text>')
    if task.chance is not None:
        yy = Y(task.chance)
        els.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W - MR}" y2="{yy:.1f}" stroke="#5a6b7c" stroke-width="1" stroke-dasharray="5 4"/>')
        els.append(
            f'<text x="{W - MR - 6}" y="{yy - 5:.1f}" fill="#71828f" font-size="10" text-anchor="end">chance {task.chance:g}</text>'
        )
    families = []
    for p in points:
        if p["family"] not in families:
            families.append(p["family"])
    for index, family in enumerate(families):
        color = FAMILY_COLORS[index % len(FAMILY_COLORS)]
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
                f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{color}" stroke="#0d1117" stroke-width="1.5"><title>{tooltip}</title></circle>'
            )
    title = html.escape(f"{task.name} — {task.primary}")
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">'
        + "".join(els)
        + "</svg>"
    )


def html_report(suite, rows, out_path):
    """Self-contained HTML report: dark theme, one interactive chart per task,
    on two axes (tokens seen and non-embedding parameters)."""
    legend_chips = []
    families = []
    for r in rows:
        if r["model"].family not in families:
            families.append(r["model"].family)
    for index, family in enumerate(families):
        color = FAMILY_COLORS[index % len(FAMILY_COLORS)]
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
        return f"<div class='legend'>{legend}</div>" + "\n".join(cards)

    md_table = markdown_table(suite, rows)
    md_rows = [line for line in md_table.splitlines() if line.startswith("|")]
    head = [cell.strip() for cell in md_rows[0].strip("|").split("|")]
    body = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in md_rows[2:]
    ]
    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(c)}</th>" for c in head)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>" for cells in body)
        + "</tbody></table>"
    )
    meta = rows[0]["meta"]
    generated = time.strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results: {html.escape(suite.name)}</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin:0; background:#0d1117; color:#dbe6f1; font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 20px 64px; }}
h1 {{ font-size:26px; margin:0 0 6px; }}
h2 {{ font-size:19px; margin:34px 0 12px; color:#e6eef6; }}
h3 {{ font-size:15px; margin:0 0 10px; color:#e6eef6; }}
.metric {{ color:#8aa0b4; font-weight:400; font-size:12px; }}
.card {{ background:#131a23; border:1px solid #232c38; border-radius:12px; padding:18px 18px 12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(560px,1fr)); gap:16px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px; margin:0 0 14px; }}
.chip {{ display:inline-flex; align-items:center; gap:7px; background:#131a23; border:1px solid #232c38; border-radius:999px; padding:4px 12px; font-size:12px; color:#c4d2df; }}
.dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.axis {{ color:#8aa0b4; font-size:12px; margin:6px 0 0; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th,td {{ border-bottom:1px solid #232c38; padding:7px 10px; text-align:left; white-space:nowrap; }}
th {{ color:#8aa0b4; font-weight:600; }}
tr:last-child td {{ color:#8aa0b4; }}
.meta {{ color:#8aa0b4; font-size:12.5px; }}
.card p {{ margin:8px 0 0; }}
</style></head><body><main>
<h1>Results: {html.escape(suite.name)}</h1>
<p class="meta">{html.escape(suite.description)}</p>
<p class="meta">Protocol v{html.escape(str(meta.get('protocol', '?')))} · lm-eval {html.escape(str(meta.get('lm_eval', '?')))} · {suite.num_fewshot}-shot · hover a point for model, score and run date · generated {generated}</p>
{table}
<h2>Score against tokens seen</h2>
{section('tokens_seen', 'tokens seen')}
<h2>Score against non-embedding parameters</h2>
{section('non_embedding_params', 'non-embedding parameters')}
<p class="meta">Static PNG versions live next to this file (<code>{html.escape(suite.name)}_vs_tokens.png</code>, <code>{html.escape(suite.name)}_vs_params.png</code>); older reports are archived under <code>history/</code>.</p>
</main></body></html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def archive_report(suite_name, reports_dir=REPORTS_DIR):
    """Move the current report + plots to reports/history/<suite>/<stamp>/."""
    md = reports_dir / f"{suite_name}.md"
    pngs = [reports_dir / f"{suite_name}_{kind}.png" for kind in ("vs_tokens", "vs_params")]
    existing = [p for p in (md, *pngs) if p.exists()]
    if not existing:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(md.stat().st_mtime))
    dest = reports_dir / "history" / suite_name / stamp
    n = 1
    while dest.exists():
        n += 1
        dest = reports_dir / "history" / suite_name / f"{stamp}-{n}"
    dest.mkdir(parents=True)
    for p in existing:
        shutil.move(str(p), dest / p.name)
    return dest


def build_report(suite_name, results_dir=RESULTS_DIR, reports_dir=REPORTS_DIR, registry=None, suites_dir=SUITES_DIR):
    registry = registry or load_registry()
    suite = load_suite(suite_name, suites_dir)
    rows = collect(suite, registry, results_dir)
    if not rows:
        raise SystemExit(f"No results for suite {suite_name!r}. Run `mini-eval run {suite_name} MODEL` first.")
    reports_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_report(suite.name, reports_dir)
    tokens_png = plot_scores(suite, rows, "tokens_seen", "tokens seen", reports_dir / f"{suite.name}_vs_tokens.png")
    params_png = plot_scores(
        suite, rows, "non_embedding_params", "non-embedding parameters", reports_dir / f"{suite.name}_vs_params.png"
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
        f"with tooltips lives in [{suite.name}.html]({suite.name}.html).",
        "",
    ]
    out = reports_dir / f"{suite.name}.md"
    out.write_text("\n".join(parts))
    html_report(suite, rows, reports_dir / f"{suite.name}.html")
    return out