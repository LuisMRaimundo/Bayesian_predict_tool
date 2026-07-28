"""Unified illustrated HTML run report (complements RUN_REPORT.md / JSON)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_list(xs) -> list:
    out = []
    for x in xs:
        if pd.isna(x):
            continue
        if isinstance(x, (np.floating, float)):
            out.append(float(x))
        elif isinstance(x, (np.integer, int)):
            out.append(int(x))
        else:
            out.append(x)
    return out


def build_chart_payload(
    *,
    bridge_ratios: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compact JSON-serializable series for Chart.js."""
    payload: dict[str, Any] = {}

    if bridge_ratios is not None and len(bridge_ratios):
        br = bridge_ratios.dropna(subset=["midi", "factor"]).copy()
        if len(br):
            br = br.sort_values("midi")
            by_tech = {}
            tech_key = br["technique"] if "technique" in br.columns else pd.Series(["all"] * len(br), index=br.index)
            for tech, g in br.groupby(tech_key):
                g = g.sort_values("midi")
                by_tech[str(tech)] = {
                    "midi": _safe_list(g["midi"].tolist()),
                    "factor": _safe_list(g["factor"].tolist()),
                    "log_ratio": _safe_list(g["log_ratio"].tolist()) if "log_ratio" in g else [],
                }
            payload["bridge_by_technique"] = by_tech
            payload["bridge_factor_values"] = _safe_list(br["factor"].tolist())

    if predictions is not None and len(predictions):
        pr = predictions.copy()
        if "support_level" in pr.columns:
            counts = pr["support_level"].astype(str).value_counts()
            payload["support_labels"] = counts.index.tolist()
            payload["support_counts"] = [int(v) for v in counts.values]
            sup = pr[pr["support_level"].isin(["supported", "supported_outlier_target"])]
        else:
            sup = pr
            payload["support_labels"] = ["all"]
            payload["support_counts"] = [int(len(pr))]

        if len(sup) and {"midi", "y_ordinario", "y_pred"}.issubset(sup.columns):
            s = sup.dropna(subset=["midi", "y_ordinario", "y_pred"]).sort_values("midi")
            payload["supported"] = {
                "midi": _safe_list(s["midi"].tolist()),
                "y_ordinario": _safe_list(s["y_ordinario"].tolist()),
                "y_pred": _safe_list(s["y_pred"].tolist()),
                "factor": _safe_list(s["factor"].tolist()) if "factor" in s else [],
                "note": [str(x) for x in s["note"].tolist()] if "note" in s else [],
            }
        if "dynamic" in pr.columns:
            dc = pr["dynamic"].astype(str).value_counts()
            payload["pred_dynamic_labels"] = dc.index.tolist()
            payload["pred_dynamic_counts"] = [int(v) for v in dc.values]
            if len(sup) and "dynamic" in sup.columns:
                med = (
                    sup.groupby("dynamic")[["y_ordinario", "y_pred", "factor"]]
                    .median(numeric_only=True)
                    .reset_index()
                )
                payload["supported_medians_by_dynamic"] = {
                    "dynamic": med["dynamic"].astype(str).tolist(),
                    "y_ordinario": _safe_list(med["y_ordinario"].tolist())
                    if "y_ordinario" in med
                    else [],
                    "y_pred": _safe_list(med["y_pred"].tolist()) if "y_pred" in med else [],
                    "factor": _safe_list(med["factor"].tolist()) if "factor" in med else [],
                }
    return payload


def write_html_report(
    record: dict[str, Any],
    *,
    bridge_ratios: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
) -> Path:
    """Write RUN_REPORT.html next to the Markdown report."""
    run_dir = Path(record["paths"]["run_dir"])
    out = run_dir / "RUN_REPORT.html"
    charts = build_chart_payload(bridge_ratios=bridge_ratios, predictions=predictions)
    record.setdefault("paths", {})["report_html"] = str(out)

    def esc(x) -> str:
        return html.escape("" if x is None else str(x))

    bridge_files = record.get("inputs", {}).get("bridge_files") or []
    tgt = record.get("inputs", {}).get("target_file") or {}
    uploaded_names = []
    for f in bridge_files:
        name = f.get("name") or Path(str(f.get("path") or "")).name
        if name:
            uploaded_names.append(("bridge", name, f))
    target_name = tgt.get("name") or Path(str(tgt.get("path") or "")).name
    if target_name:
        uploaded_names.append(("target ordinario", target_name, tgt))

    uploaded_list_html = "".join(
        f"<li><span class='tag'>{esc(role)}</span> <b>{esc(name)}</b></li>"
        for role, name, _ in uploaded_names
    ) or "<li><i>No Excel files recorded for this run</i></li>"

    bridge_rows = []
    for i, f in enumerate(bridge_files, start=1):
        bridge_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><b>{esc(f.get('name'))}</b></td>"
            f"<td>{esc(f.get('path'))}</td>"
            f"<td>{esc(f.get('size_bytes'))}</td>"
            f"<td><code>{esc((f.get('sha256') or '')[:16])}…</code></td>"
            "</tr>"
        )
    cfg_rows = "".join(
        f"<tr><td>{esc(k)}</td><td><code>{esc(v)}</code></td></tr>"
        for k, v in (record.get("config") or {}).items()
    )
    op_rows = "".join(
        "<tr>"
        f"<td>{esc(op.get('time_local'))}</td>"
        f"<td>{esc(op.get('operation'))}</td>"
        f"<td><code>{esc(op.get('detail'))}</code></td>"
        "</tr>"
        for op in (record.get("operations") or [])
    )
    summ_blocks = []
    for name, summ in (record.get("data_summaries") or {}).items():
        items = "".join(f"<li><b>{esc(k)}:</b> <code>{esc(v)}</code></li>" for k, v in (summ or {}).items())
        summ_blocks.append(f"<div class='card'><h3>{esc(name)}</h3><ul>{items}</ul></div>")

    fit = dict(record.get("fit") or {})
    cfg = record.get("config") or {}
    # Preflight / failed runs have no fit payload — fall back to selected config.
    if not fit.get("model_id"):
        fit.setdefault("model_id", cfg.get("model_id"))
    if not fit.get("metric"):
        fit.setdefault("metric", cfg.get("metric"))
    if not fit.get("backend"):
        fit.setdefault("backend", "(not fitted — Preflight or failed run)")
    if fit.get("bridge_n") in (None, ""):
        fit.setdefault("bridge_n", "—")

    cv = record.get("blocked_cv") or {}
    excel = (record.get("outputs") or {}).get("excel_audit") or {}
    n_sup = 0
    n_all = 0
    pred_sum = (record.get("data_summaries") or {}).get("predictions") or {}
    if "support_level_counts" in pred_sum:
        counts = pred_sum["support_level_counts"] or {}
        n_all = int(sum(int(v) for v in counts.values()))
        n_sup = int(counts.get("supported", 0)) + int(counts.get("supported_outlier_target", 0))
    elif "n_rows" in pred_sum:
        n_all = int(pred_sum["n_rows"])

    status = str(record.get("status") or "")
    kind = str(record.get("kind") or "")
    status_class = "ok" if status == "ok" else ("warn" if "preflight" in status or kind == "preflight" else "bad")
    has_charts = bool(
        charts.get("support_labels")
        or charts.get("bridge_by_technique")
        or charts.get("supported")
    )
    is_preflight = kind == "preflight" or status in {"preflight_fail", "preflight"}
    if is_preflight and not has_charts:
        results_note = (
            "This is a <b>Preflight</b> report. Charts, fitted-model fields, and predictions "
            "appear only after <b>Fit &amp; predict</b>."
        )
    elif not has_charts:
        results_note = (
            "No prediction/bridge chart data was saved for this run "
            "(failed early, or transfer did not produce pairs)."
        )
    else:
        results_note = (
            "Charts need an internet connection the first time (Chart.js CDN). "
            "If boxes stay blank offline, reopen with network or check the CSV twins in this folder."
        )

    def _chart_box(canvas_id: str, *, tall: bool = False, empty_msg: str) -> str:
        cls = "chart-box tall" if tall else "chart-box"
        return (
            f"<div class='{cls}' data-canvas='{canvas_id}'>"
            f"<canvas id='{canvas_id}'></canvas>"
            f"<div class='empty-chart' id='{canvas_id}_empty'>{esc(empty_msg)}</div>"
            f"</div>"
        )

    empty_msg = (
        "No chart data for this run (Preflight does not fit or predict)."
        if is_preflight
        else "No chart data available for this run."
    )
    cv_list = "".join(f"<li><b>{esc(k)}:</b> <code>{esc(v)}</code></li>" for k, v in cv.items())
    if not cv_list:
        if cfg.get("run_blocked_cv") is False:
            cv_list = "<li>Blocked CV was <b>disabled</b> in config for this run.</li>"
        elif is_preflight:
            cv_list = "<li>n/a — Preflight does not run blocked CV.</li>"
        else:
            cv_list = "<li>n/a — no blocked-CV table was written.</li>"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Run report — {esc(record.get('run_id'))}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:#f6f4ef; --card:#fff; --text:#1c1c1c; --muted:#5f5a52; --line:#e4dfd4;
  --accent:#2f5d50; --ok:#2f5d50; --warn:#8a6d2f; --bad:#8b3a3a;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.45 Georgia, 'Times New Roman', serif; color:var(--text); background:var(--bg); }}
.wrap {{ max-width:1120px; margin:0 auto; padding:28px 20px 72px; }}
h1 {{ font-size:1.8rem; margin:0 0 6px; letter-spacing:-0.02em; }}
h2 {{ font-size:1.2rem; margin:28px 0 10px; padding-top:18px; border-top:1px solid var(--line); }}
h3 {{ font-size:1rem; margin:0 0 8px; }}
.sub {{ color:var(--muted); margin:0 0 14px; }}
.pills {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 18px; }}
.pill {{ font:12px/1.2 system-ui,sans-serif; padding:5px 10px; border-radius:999px; background:#e8efec; color:var(--accent); }}
.pill.ok {{ background:#e5f0ea; color:var(--ok); }}
.pill.warn {{ background:#f4eedc; color:var(--warn); }}
.pill.bad {{ background:#f3e4e4; color:var(--bad); }}
.files {{ background:var(--card); border:1px solid var(--line); padding:14px 16px; margin:0 0 16px; }}
.files h2 {{ margin:0 0 8px; border:0; padding:0; font-size:1.1rem; }}
.files ol, .files ul {{ margin:0; padding-left:20px; }}
.files li {{ margin:4px 0; }}
.tag {{ font:11px/1.2 system-ui,sans-serif; color:#fff; background:var(--accent); padding:2px 7px; border-radius:999px; margin-right:6px; }}
.stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0 18px; }}
.stat {{ background:var(--card); border:1px solid var(--line); padding:12px 14px; }}
.stat b {{ display:block; font:1.25rem/1.2 system-ui,sans-serif; }}
.stat span {{ color:var(--muted); font:12px system-ui,sans-serif; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.card {{ background:var(--card); border:1px solid var(--line); padding:12px 14px; margin:0 0 12px; }}
.chart-box {{ position:relative; background:var(--card); border:1px solid var(--line); padding:10px; height:320px; margin:0 0 14px; }}
.chart-box.tall {{ height:420px; }}
.chart-box.has-data .empty-chart {{ display:none; }}
.empty-chart {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; padding:24px; text-align:center; color:var(--muted); font:13px/1.4 system-ui,sans-serif; background:var(--card); }}
table {{ width:100%; border-collapse:collapse; font:12.5px/1.35 system-ui,sans-serif; background:var(--card); margin:8px 0 14px; }}
th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; word-break:break-word; }}
th {{ background:#f0ebe2; position:sticky; top:0; }}
ul {{ margin:0; padding-left:18px; }}
code {{ font-family:Consolas, monospace; font-size:12px; }}
a {{ color:var(--accent); }}
.callout {{ background:#eef4f1; border-left:3px solid var(--accent); padding:10px 12px; margin:12px 0; }}
.callout.warn {{ background:#f4eedc; border-left-color:var(--warn); }}
@media (max-width:900px) {{ .stats,.grid2 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Run report — {esc(record.get('run_id'))}</h1>
  <p class="sub">Unified illustrated history for this Bayesian technique-transfer run. Markdown/JSON twins remain in the same folder.</p>
  <div class="pills">
    <span class="pill {status_class}">{esc(status)}</span>
    <span class="pill">{esc(record.get('kind'))}</span>
    <span class="pill">{esc((record.get('config') or {}).get('model_id'))}</span>
    <span class="pill">{esc(record.get('started_local'))} → {esc(record.get('finished_local'))}</span>
    <span class="pill">{esc(record.get('duration_seconds'))}s</span>
  </div>

  <div class="files">
    <h2>Uploaded Excel files ({len(uploaded_names)})</h2>
    <p class="sub" style="margin:0 0 8px">Every workbook selected for this run (bridges + target).</p>
    <ol>{uploaded_list_html}</ol>
  </div>

  <div class="stats">
    <div class="stat"><b>{esc(n_sup)}/{esc(n_all)}</b><span>Supported / all predictions</span></div>
    <div class="stat"><b>{esc((record.get('data_summaries') or {}).get('bridge_log_ratios', {}).get('n_rows', 0))}</b><span>Bridge log-ratio pairs</span></div>
    <div class="stat"><b>{esc(len(bridge_files))}</b><span>Bridge Excel files</span></div>
    <div class="stat"><b>{esc(cv.get('mae_log', '—'))}</b><span>Blocked CV MAE (log)</span></div>
  </div>

  <div class="callout{' warn' if not has_charts else ''}">
    {results_note}
  </div>
  <div class="callout">
    <b>Primary result (after Fit &amp; predict):</b> Excel sheet <code>Predictions_supported</code>, column <code>y_pred</code>
    {(' — file <code>' + esc(excel.get('name')) + '</code>') if excel.get('name') else ''}.
    Also see <a href="RUN_REPORT.md">RUN_REPORT.md</a> and <a href="run_manifest.json">run_manifest.json</a>.
  </div>

  <h2>Inputs (paths &amp; hashes)</h2>
  <table>
    <thead><tr><th>#</th><th>Bridge Excel name</th><th>Path</th><th>Bytes</th><th>SHA256</th></tr></thead>
    <tbody>{''.join(bridge_rows) if bridge_rows else '<tr><td colspan="5">No bridge files recorded</td></tr>'}</tbody>
  </table>
  <div class="card">
    <h3>Target ordinario — <b>{esc(target_name or '(none)')}</b></h3>
    <ul>
      <li><b>name:</b> <code>{esc(tgt.get('name'))}</code></li>
      <li><b>path:</b> <code>{esc(tgt.get('path'))}</code></li>
      <li><b>sha256:</b> <code>{esc(tgt.get('sha256'))}</code></li>
      <li><b>instrument:</b> <code>{esc(record.get('inputs', {}).get('instrument'))}</code></li>
      <li><b>zenodo_collection:</b> <code>{esc(record.get('inputs', {}).get('zenodo_collection'))}</code></li>
    </ul>
  </div>

  <h2>Config</h2>
  <table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>{cfg_rows}</tbody></table>

  <h2>Illustrated results</h2>
  <div class="grid2">
    <div>
      <h3>Support levels</h3>
      {_chart_box("supportChart", empty_msg=empty_msg)}
    </div>
    <div>
      <h3>Prediction rows by dynamic</h3>
      {_chart_box("dynChart", empty_msg=empty_msg)}
    </div>
  </div>
  <h3>Bridge transfer factors vs MIDI</h3>
  {_chart_box("bridgeChart", tall=True, empty_msg=empty_msg)}
  <h3>Supported: ordinario vs predicted (by MIDI)</h3>
  {_chart_box("predChart", tall=True, empty_msg=empty_msg)}
  <h3>Supported medians by dynamic</h3>
  {_chart_box("medChart", empty_msg=empty_msg)}

  <h2>Model / CV</h2>
  <div class="grid2">
    <div class="card">
      <h3>Fit</h3>
      <ul>
        <li><b>model_id:</b> <code>{esc(fit.get('model_id') or '—')}</code></li>
        <li><b>backend:</b> <code>{esc(fit.get('backend') or '—')}</code></li>
        <li><b>metric:</b> <code>{esc(fit.get('metric') or '—')}</code></li>
        <li><b>bridge_n:</b> <code>{esc(fit.get('bridge_n') if fit.get('bridge_n') is not None else '—')}</code></li>
      </ul>
    </div>
    <div class="card">
      <h3>Blocked CV</h3>
      <ul>
        {cv_list}
      </ul>
    </div>
  </div>

  <h2>Data summaries</h2>
  <div class="grid2">{''.join(summ_blocks) or '<div class="card">No summaries</div>'}</div>

  <h2>Operations</h2>
  <table>
    <thead><tr><th>Time (local)</th><th>Operation</th><th>Detail</th></tr></thead>
    <tbody>{op_rows or '<tr><td colspan="3">None</td></tr>'}</tbody>
  </table>

  <h2>Outputs</h2>
  <ul>
    <li><b>Excel audit:</b> <code>{esc(excel.get('path'))}</code></li>
    <li><b>This HTML:</b> <code>{esc(out.name)}</code></li>
    <li><b>Markdown:</b> <a href="RUN_REPORT.md">RUN_REPORT.md</a></li>
    <li><b>Manifest:</b> <a href="run_manifest.json">run_manifest.json</a></li>
    <li><b>CSVs:</b> predictions_supported.csv, bridge_log_ratios.csv, …</li>
  </ul>
</div>
<script>
const CHARTS = {json.dumps(charts, ensure_ascii=False)};
const colors = ['#2f5d50','#3d6e8c','#8a6d2f','#8b3a3a','#5c7a6e','#6b5b95','#c06c3f'];
function markDrawn(id) {{
  const box = document.querySelector('[data-canvas=\"' + id + '\"]');
  if (box) box.classList.add('has-data');
}}
function chartReady() {{ return typeof Chart !== 'undefined'; }}

if (chartReady() && CHARTS.support_labels && CHARTS.support_labels.length) {{
  new Chart(document.getElementById('supportChart'), {{
    type: 'doughnut',
    data: {{
      labels: CHARTS.support_labels,
      datasets: [{{ data: CHARTS.support_counts, backgroundColor: colors }}]
    }},
    options: {{ plugins: {{ legend: {{ position: 'bottom' }} }}, maintainAspectRatio: false }}
  }});
  markDrawn('supportChart');
}}
if (chartReady() && CHARTS.pred_dynamic_labels && CHARTS.pred_dynamic_labels.length) {{
  new Chart(document.getElementById('dynChart'), {{
    type: 'bar',
    data: {{
      labels: CHARTS.pred_dynamic_labels,
      datasets: [{{ label: 'n predictions', data: CHARTS.pred_dynamic_counts, backgroundColor: '#3d6e8c' }}]
    }},
    options: {{ plugins: {{ legend: {{ display:false }} }}, scales: {{ y: {{ beginAtZero:true, title: {{ display:true, text:'count' }} }} }}, maintainAspectRatio:false }}
  }});
  markDrawn('dynChart');
}}
if (chartReady() && CHARTS.bridge_by_technique) {{
  const ds = Object.entries(CHARTS.bridge_by_technique).map(([tech, o], i) => ({{
    label: tech + ' factor',
    data: o.midi.map((m, j) => ({{ x: m, y: o.factor[j] }})),
    showLine: true,
    borderColor: colors[i % colors.length],
    backgroundColor: colors[i % colors.length],
    pointRadius: 3
  }}));
  new Chart(document.getElementById('bridgeChart'), {{
    type: 'scatter',
    data: {{ datasets: ds }},
    options: {{
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom' }} }},
      scales: {{
        x: {{ title: {{ display:true, text:'MIDI' }} }},
        y: {{ title: {{ display:true, text:'transfer factor' }}, beginAtZero:false }}
      }}
    }}
  }});
  markDrawn('bridgeChart');
}}
if (chartReady() && CHARTS.supported) {{
  const s = CHARTS.supported;
  new Chart(document.getElementById('predChart'), {{
    type: 'line',
    data: {{
      labels: s.midi,
      datasets: [
        {{ label: 'y_ordinario', data: s.y_ordinario, borderColor: '#8a6d2f', backgroundColor: '#8a6d2f', pointRadius: 2, tension: 0.15 }},
        {{ label: 'y_pred', data: s.y_pred, borderColor: '#2f5d50', backgroundColor: '#2f5d50', pointRadius: 2, tension: 0.15 }}
      ]
    }},
    options: {{
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom' }}, tooltip: {{ callbacks: {{
        afterBody: (items) => {{
          const i = items[0].dataIndex;
          const note = (s.note && s.note[i]) ? s.note[i] : '';
          const fac = (s.factor && s.factor[i] != null) ? Number(s.factor[i]).toFixed(3) : '';
          return [note ? ('note: ' + note) : '', fac ? ('factor: ' + fac) : ''].filter(Boolean);
        }}
      }} }} }},
      scales: {{
        x: {{ title: {{ display:true, text:'MIDI (supported rows)' }} }},
        y: {{ title: {{ display:true, text:'EWSD / density metric' }} }}
      }}
    }}
  }});
  markDrawn('predChart');
}}
if (chartReady() && CHARTS.supported_medians_by_dynamic) {{
  const m = CHARTS.supported_medians_by_dynamic;
  new Chart(document.getElementById('medChart'), {{
    type: 'bar',
    data: {{
      labels: m.dynamic,
      datasets: [
        {{ label: 'median y_ordinario', data: m.y_ordinario, backgroundColor: '#8a6d2f' }},
        {{ label: 'median y_pred', data: m.y_pred, backgroundColor: '#2f5d50' }}
      ]
    }},
    options: {{
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom' }} }},
      scales: {{ y: {{ beginAtZero:true, title: {{ display:true, text:'median metric' }} }} }}
    }}
  }});
  markDrawn('medChart');
}}
if (!chartReady()) {{
  document.querySelectorAll('.empty-chart').forEach((el) => {{
    el.textContent = 'Chart library failed to load (need network for Chart.js CDN).';
  }});
}}
</script>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out
