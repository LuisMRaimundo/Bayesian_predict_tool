"""Persistent run history: timestamped manifests + human-readable reports."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_ROOT = ROOT / "outputs" / "run_history"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _file_info(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path).strip() == "":
        return {"path": None, "exists": False}
    p = Path(path).expanduser()
    info: dict[str, Any] = {
        "path": str(p.resolve()) if p.exists() else str(p),
        "name": p.name,
        "exists": p.exists(),
    }
    if p.exists() and p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        st = p.stat()
        info.update(
            {
                "size_bytes": int(st.st_size),
                "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "sha256": h.hexdigest(),
            }
        )
    return info


def _panel_summary(df: pd.DataFrame | None, label: str) -> dict[str, Any]:
    if df is None or len(df) == 0:
        return {"label": label, "n_rows": 0}
    out: dict[str, Any] = {"label": label, "n_rows": int(len(df))}
    for col in ("technique", "dynamic", "collection", "instrument"):
        if col in df.columns:
            out[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    if "midi" in df.columns and df["midi"].notna().any():
        out["midi_min"] = float(df["midi"].min())
        out["midi_max"] = float(df["midi"].max())
    if "support_level" in df.columns:
        out["support_level_counts"] = df["support_level"].astype(str).value_counts().to_dict()
    if "is_ordinario" in df.columns:
        out["n_ordinario"] = int(df["is_ordinario"].sum())
        out["n_special"] = int((~df["is_ordinario"]).sum())
    return out


def start_run(
    *,
    kind: str,
    bridge_paths: list[str | Path] | None = None,
    target_path: str | Path | None = None,
    output_xlsx: str | Path | None = None,
    config: dict | None = None,
    instrument: str | None = None,
    zenodo_collection: str | None = None,
    history_root: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Create a new run folder and return a mutable run record."""
    root = Path(history_root) if history_root else DEFAULT_HISTORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    stamp = _local_stamp()
    run_id = f"{stamp}_{kind}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = _utc_now()
    record: dict[str, Any] = {
        "run_id": run_id,
        "kind": kind,
        "status": "started",
        "started_local": datetime.now().isoformat(timespec="seconds"),
        "started_utc": started.isoformat(),
        "finished_local": None,
        "finished_utc": None,
        "duration_seconds": None,
        "tool": {
            "package": "string_technique_transfer",
            "repo": "https://github.com/LuisMRaimundo/Bayesian_predict_tool",
            "local_root": str(ROOT),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "user": os.environ.get("USERNAME") or os.environ.get("USER"),
        },
        "inputs": {
            "bridge_files": [_file_info(p) for p in (bridge_paths or [])],
            "target_file": _file_info(target_path),
            "instrument": instrument,
            "zenodo_collection": zenodo_collection,
        },
        "config": config or {},
        "operations": [],
        "data_summaries": {},
        "outputs": {"requested_xlsx": str(output_xlsx) if output_xlsx else None},
        "errors": [],
        "warnings": [],
        "notes": notes,
        "paths": {
            "run_dir": str(run_dir),
            "manifest_json": str(run_dir / "run_manifest.json"),
            "report_md": str(run_dir / "RUN_REPORT.md"),
            "report_html": str(run_dir / "RUN_REPORT.html"),
        },
    }
    _write_manifest(record)
    return record


def log_operation(record: dict[str, Any], name: str, detail: dict | str | None = None) -> None:
    entry: dict[str, Any] = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "time_utc": _utc_now().isoformat(),
        "operation": name,
    }
    if detail is not None:
        entry["detail"] = detail
    record.setdefault("operations", []).append(entry)
    _write_manifest(record)


def finalize_run(
    record: dict[str, Any],
    *,
    status: str = "ok",
    bridge_panel: pd.DataFrame | None = None,
    target: pd.DataFrame | None = None,
    bridge_ratios: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    preflight_df: pd.DataFrame | None = None,
    cv_table: pd.DataFrame | None = None,
    fit_summary: dict | None = None,
    output_xlsx: str | Path | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """Write final JSON + Markdown report and append to INDEX."""
    finished = _utc_now()
    started = datetime.fromisoformat(record["started_utc"])
    record["status"] = status
    record["finished_local"] = datetime.now().isoformat(timespec="seconds")
    record["finished_utc"] = finished.isoformat()
    record["duration_seconds"] = round((finished - started).total_seconds(), 3)
    if errors:
        record["errors"] = list(errors)
    if warnings:
        record["warnings"] = list(warnings)
    record["data_summaries"] = {
        "bridge_panel": _panel_summary(bridge_panel, "bridge_panel"),
        "target_ordinario": _panel_summary(target, "target_ordinario"),
        "bridge_log_ratios": _panel_summary(bridge_ratios, "bridge_log_ratios"),
        "predictions": _panel_summary(predictions, "predictions"),
    }
    if fit_summary:
        record["fit"] = fit_summary
    if cv_table is not None and len(cv_table):
        record["blocked_cv"] = cv_table.iloc[0].astype(object).where(pd.notna(cv_table.iloc[0]), None).to_dict()
    if preflight_df is not None and len(preflight_df):
        record["preflight_rows"] = preflight_df.to_dict(orient="records")
    if output_xlsx is not None:
        record["outputs"]["excel_audit"] = _file_info(output_xlsx)
    if extra:
        record["extra"] = extra

    run_dir = Path(record["paths"]["run_dir"])
    from .run_report_html import write_html_report

    html_path = write_html_report(
        record,
        bridge_ratios=bridge_ratios,
        predictions=predictions,
    )
    record["paths"]["report_html"] = str(html_path)
    _write_manifest(record)
    report_path = _write_markdown(record)
    if preflight_df is not None and len(preflight_df):
        preflight_df.to_csv(run_dir / "preflight.csv", index=False)
    if cv_table is not None and len(cv_table):
        cv_table.to_csv(run_dir / "blocked_cv.csv", index=False)
    if bridge_ratios is not None and len(bridge_ratios):
        bridge_ratios.to_csv(run_dir / "bridge_log_ratios.csv", index=False)
    if predictions is not None and len(predictions):
        # keep history lean: supported-only if available
        if "support_level" in predictions.columns:
            sup = predictions[
                predictions["support_level"].isin(["supported", "supported_outlier_target"])
            ]
            sup.to_csv(run_dir / "predictions_supported.csv", index=False)
        predictions.head(500).to_csv(run_dir / "predictions_all_head.csv", index=False)

    _append_index(record)
    # Prefer HTML as the primary human-facing history link; MD remains as twin.
    return html_path if html_path.exists() else report_path


def _write_manifest(record: dict[str, Any]) -> None:
    path = Path(record["paths"]["manifest_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_markdown(record: dict[str, Any]) -> Path:
    path = Path(record["paths"]["report_md"])
    bridge_files = record.get("inputs", {}).get("bridge_files") or []
    tgt = record.get("inputs", {}).get("target_file") or {}
    uploaded_names = [f.get("name") or Path(str(f.get("path") or "")).name for f in bridge_files]
    target_name = tgt.get("name") or Path(str(tgt.get("path") or "")).name
    html_name = Path(record.get("paths", {}).get("report_html") or "RUN_REPORT.html").name

    lines = [
        f"# Run report — `{record['run_id']}`",
        "",
        f"- **Kind:** {record.get('kind')}",
        f"- **Status:** {record.get('status')}",
        f"- **Started (local):** {record.get('started_local')}",
        f"- **Finished (local):** {record.get('finished_local')}",
        f"- **Started (UTC):** {record.get('started_utc')}",
        f"- **Finished (UTC):** {record.get('finished_utc')}",
        f"- **Duration (s):** {record.get('duration_seconds')}",
        f"- **Illustrated HTML:** [`{html_name}`]({html_name})",
        "",
        "## Uploaded Excel files",
        "",
    ]
    if uploaded_names:
        for i, name in enumerate(uploaded_names, start=1):
            lines.append(f"{i}. `{name}` *(bridge)*")
    else:
        lines.append("- *(no bridge Excel files recorded)*")
    if target_name:
        lines.append(f"- `{target_name}` *(target ordinario)*")
    else:
        lines.append("- *(no target Excel recorded)*")

    lines += ["", "## Tool", ""]
    for k, v in (record.get("tool") or {}).items():
        lines.append(f"- **{k}:** `{v}`")

    lines += ["", "## Inputs (paths & hashes)", ""]
    for i, f in enumerate(bridge_files, start=1):
        lines.append(f"### Bridge file {i}: `{f.get('name')}`")
        for k, v in f.items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")
    lines.append(f"### Target ordinario: `{target_name}`")
    for k, v in tgt.items():
        lines.append(f"- **{k}:** `{v}`")
    lines.append(f"- **instrument:** `{record.get('inputs', {}).get('instrument')}`")
    lines.append(f"- **zenodo_collection:** `{record.get('inputs', {}).get('zenodo_collection')}`")

    lines += ["", "## Config", ""]
    for k, v in (record.get("config") or {}).items():
        lines.append(f"- **{k}:** `{v}`")

    lines += ["", "## Operations (chronological)", ""]
    for op in record.get("operations") or []:
        lines.append(f"- `{op.get('time_local')}` — **{op.get('operation')}**")
        if op.get("detail"):
            lines.append(f"  - detail: `{op['detail']}`")

    lines += ["", "## Data summaries", ""]
    for name, summ in (record.get("data_summaries") or {}).items():
        lines.append(f"### {name}")
        for k, v in (summ or {}).items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    if record.get("fit"):
        lines += ["## Model fit", ""]
        for k, v in record["fit"].items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    if record.get("blocked_cv"):
        lines += ["## Blocked CV", ""]
        for k, v in record["blocked_cv"].items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    lines += ["## Outputs", ""]
    for k, v in (record.get("outputs") or {}).items():
        lines.append(f"- **{k}:** `{v}`")

    if record.get("errors"):
        lines += ["", "## Errors", ""]
        for e in record["errors"]:
            lines.append(f"- {e}")
    if record.get("warnings"):
        lines += ["", "## Warnings", ""]
        for w in record["warnings"]:
            lines.append(f"- {w}")
    if record.get("notes"):
        lines += ["", "## Notes", "", str(record["notes"]), ""]

    lines += [
        "",
        "## How to reuse the prediction",
        "",
        "1. Excel audit file (if present) → sheet **Predictions_supported** → column **y_pred**",
        "2. Or CSV in this folder: `predictions_supported.csv`",
        "",
        f"Illustrated HTML twin: `{html_name}`",
        f"Machine-readable twin: `{Path(record['paths']['manifest_json']).name}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _append_index(record: dict[str, Any]) -> None:
    root = Path(record["paths"]["run_dir"]).parent
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": record.get("run_id"),
        "kind": record.get("kind"),
        "status": record.get("status"),
        "started_local": record.get("started_local"),
        "finished_local": record.get("finished_local"),
        "duration_seconds": record.get("duration_seconds"),
        "model_id": (record.get("config") or {}).get("model_id"),
        "n_bridge_files": len((record.get("inputs") or {}).get("bridge_files") or []),
        "target": ((record.get("inputs") or {}).get("target_file") or {}).get("name"),
        "output_xlsx": ((record.get("outputs") or {}).get("excel_audit") or {}).get("name"),
        "report_md": record.get("paths", {}).get("report_md"),
        "report_html": record.get("paths", {}).get("report_html"),
        "uploaded_excels": "; ".join(
            [
                *(
                    f.get("name") or Path(str(f.get("path") or "")).name
                    for f in ((record.get("inputs") or {}).get("bridge_files") or [])
                ),
                *(
                    [((record.get("inputs") or {}).get("target_file") or {}).get("name")]
                    if ((record.get("inputs") or {}).get("target_file") or {}).get("name")
                    else []
                ),
            ]
        ),
    }
    idx_csv = root / "index.csv"
    df = pd.DataFrame([row])
    if idx_csv.exists():
        old = pd.read_csv(idx_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(idx_csv, index=False)

    idx_md = root / "INDEX.md"
    line = (
        f"| `{row['run_id']}` | {row['kind']} | {row['status']} | {row['started_local']} | "
        f"{row['model_id']} | {row['n_bridge_files']} | {row['target']} | "
        f"{row['uploaded_excels']} | "
        f"[HTML]({row['run_id']}/RUN_REPORT.html) / [MD]({row['run_id']}/RUN_REPORT.md) |"
    )
    header = (
        "# Run history index\n\n"
        "Each Fit & predict / Preflight writes a timestamped folder under this directory.\n"
        "Open **RUN_REPORT.html** for the illustrated compilation (charts + all uploaded Excel names).\n\n"
        "| run_id | kind | status | started_local | model | n_bridge_files | target | uploaded_excels | report |\n"
        "|---|---|---|---|---|---:|---|---|---|\n"
    )
    if idx_md.exists():
        text = idx_md.read_text(encoding="utf-8")
        if line not in text:
            if not text.endswith("\n"):
                text += "\n"
            idx_md.write_text(text + line + "\n", encoding="utf-8")
    else:
        idx_md.write_text(header + line + "\n", encoding="utf-8")
