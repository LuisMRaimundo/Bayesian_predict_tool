"""End-to-end local pipeline with preflight, robust transfer, and validation pack."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .bridge import build_log_ratios, summarize_factors
from .clean import audit_summary, dedupe_panel
from .config import TransferConfig
from .export.excel_audit import export_audit_workbook
from .io.loaders import load_panel
from .models.base import FitResult
from .models.fit import fit_model, predict_transfer
from .preflight import PreflightResult, preflight_transfer
from .quality import build_quality_report
from .validation.blocked_cv import blocked_pitch_cv
from .validation.calibration import calibrate_from_bridge
from .validation.compare import compare_models, recommended_model_id
from .validation.holdout import holdout_bridge_validation
from .validation.sensitivity import sensitivity_grid


def load_and_clean(path: str | Path, metric: str = "EWSD_score_acoustic_balanced") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = load_panel(path, default_metric=metric)
    if "metric" in panel.columns:
        panel.loc[panel["metric"].isna() | (panel["metric"] == "unknown"), "metric"] = metric
        panel["metric"] = panel["metric"].fillna(metric)
    panel["metric"] = metric
    clean, dups = dedupe_panel(panel)
    return clean, dups, audit_summary(panel)


def run_transfer(
    bridge_panel: pd.DataFrame,
    target_ordinario: pd.DataFrame,
    *,
    techniques: Sequence[str] | None = None,
    model_id: str | None = None,
    metric: str | None = None,
    require_same_collection: bool | None = None,
    strict_dynamics: bool | None = None,
    max_dynamic_distance: int | None = None,
    run_blocked_cv: bool | None = None,
    config: TransferConfig | None = None,
    output_xlsx: str | Path | None = None,
    skip_preflight: bool = False,
    run_meta: dict | None = None,
) -> tuple[FitResult, pd.DataFrame, pd.DataFrame, Path | None, PreflightResult, pd.DataFrame]:
    """Run full robust transfer.

    Returns
    -------
    fit, bridge, predictions_all, out_path, preflight, cv_table

    If ``run_meta`` is provided (or omitted — auto-created), a comprehensive
    timestamped report is written under ``outputs/run_history/``.
    """
    from .run_history import finalize_run, log_operation, start_run

    cfg = config or TransferConfig()
    if model_id is not None:
        cfg.model_id = model_id
    if metric is not None:
        cfg.metric = metric
    if require_same_collection is not None:
        cfg.require_same_collection = require_same_collection
    if strict_dynamics is not None:
        cfg.strict_dynamics = strict_dynamics
    if max_dynamic_distance is not None:
        cfg.max_dynamic_distance = max_dynamic_distance
    if run_blocked_cv is not None:
        cfg.run_blocked_cv = run_blocked_cv
    cfg.validate()

    meta = run_meta or {}
    record = start_run(
        kind=str(meta.get("kind") or "transfer"),
        bridge_paths=meta.get("bridge_paths") or [],
        target_path=meta.get("target_path"),
        output_xlsx=output_xlsx,
        config=cfg.to_dict(),
        instrument=meta.get("instrument"),
        zenodo_collection=meta.get("zenodo_collection"),
        notes=meta.get("notes"),
        history_root=meta.get("history_root"),
    )
    log_operation(record, "run_transfer_started")

    try:
        pf = (
            PreflightResult(True, [], [], {"skipped": True})
            if skip_preflight
            else preflight_transfer(bridge_panel, target_ordinario, cfg)
        )
        log_operation(
            record,
            "preflight",
            {"ok": pf.ok, "errors": pf.errors, "warnings": pf.warnings, "skipped": skip_preflight},
        )
        if not pf.ok:
            raise ValueError("Preflight failed:\n- " + "\n- ".join(pf.errors))

        bridge = build_log_ratios(
            bridge_panel,
            metric=cfg.metric,
            require_same_collection=cfg.require_same_collection,
            max_dynamic_distance=cfg.max_dynamic_distance,
        )
        log_operation(record, "build_log_ratios", {"n_pairs": int(len(bridge))})
        if techniques is None:
            techniques = sorted(bridge["technique"].unique().tolist())

        comparison = (
            compare_models(bridge, metric=cfg.metric, block_semitones=cfg.cv_block_semitones)
            if cfg.run_model_comparison
            else pd.DataFrame()
        )
        if len(comparison):
            log_operation(
                record,
                "model_comparison",
                {"recommended": recommended_model_id(comparison, default=cfg.model_id)},
            )
        if cfg.auto_select_model and len(comparison):
            cfg.model_id = recommended_model_id(comparison, default=cfg.model_id)
            pf.summary["auto_selected_model"] = cfg.model_id
            log_operation(record, "auto_select_model", {"model_id": cfg.model_id})

        calib = (
            calibrate_from_bridge(
                bridge,
                model_id=cfg.model_id,
                metric=cfg.metric,
                block_semitones=cfg.cv_block_semitones,
            )
            if cfg.run_calibration
            else None
        )
        if calib is not None:
            log_operation(record, "calibration", calib.to_dict())

        fit = fit_model(bridge, model_id=cfg.model_id, metric=cfg.metric)
        log_operation(
            record,
            "fit_model",
            {"model_id": fit.model_id, "backend": fit.backend, "bridge_n": fit.bridge_n},
        )
        if calib is not None:
            fit.params["calibration"] = calib.to_dict()
            fit.diagnostics["calibration_status"] = calib.status
        if len(comparison):
            fit.params["model_comparison"] = comparison.to_dict(orient="list")
            rec = recommended_model_id(comparison, default=cfg.model_id)
            fit.diagnostics["recommended_model"] = rec
            pf.summary["recommended_model"] = rec

        bridge_dyns = (
            bridge.groupby("technique")["dynamic"]
            .apply(lambda s: sorted(s.dropna().astype(str).unique()))
            .to_dict()
        )
        preds = predict_transfer(
            fit,
            target_ordinario,
            list(techniques),
            bridge_dynamics_by_technique=bridge_dyns,
            max_dynamic_distance=cfg.max_dynamic_distance,
            strict_dynamics=cfg.strict_dynamics,
            calibration=calib,
        )
        log_operation(record, "predict_transfer", {"n_predictions": int(len(preds))})
        if len(preds) and "support_level" in preds.columns:
            supported = preds[preds["support_level"].isin(["supported", "supported_outlier_target"])].copy()
        else:
            supported = preds.copy()

        cv_table = (
            blocked_pitch_cv(
                bridge,
                model_id=cfg.model_id,
                metric=cfg.metric,
                block_semitones=cfg.cv_block_semitones,
            )
            if cfg.run_blocked_cv
            else pd.DataFrame([{"status": "skipped", "reason": "disabled_by_config"}])
        )
        log_operation(record, "blocked_cv", cv_table.iloc[0].to_dict() if len(cv_table) else {})

        holdout_detail, holdout_summary = (
            holdout_bridge_validation(
                bridge,
                model_id=cfg.model_id,
                metric=cfg.metric,
                holdout_frac=cfg.holdout_frac,
            )
            if cfg.run_holdout
            else (pd.DataFrame(), pd.DataFrame([{"status": "skipped"}]))
        )
        if len(holdout_summary):
            log_operation(record, "holdout", holdout_summary.iloc[0].to_dict())

        sens = (
            sensitivity_grid(
                bridge_panel,
                metric=cfg.metric,
                model_id=cfg.model_id,
                require_same_collection=cfg.require_same_collection,
                max_dynamic_distance=cfg.max_dynamic_distance,
            )
            if cfg.run_sensitivity
            else pd.DataFrame()
        )
        if len(sens):
            log_operation(record, "sensitivity", {"n_rows": int(len(sens))})

        factors = summarize_factors(bridge)
        quality = build_quality_report(
            bridge,
            preds,
            max_dynamic_distance=cfg.max_dynamic_distance,
            output_xlsx=output_xlsx,
        )
        extras = []
        if len(cv_table):
            for k, v in cv_table.iloc[0].to_dict().items():
                extras.append({"item": f"cv_{k}", "value": v})
        if calib is not None:
            for k, v in calib.to_dict().items():
                extras.append({"item": f"calib_{k}", "value": v})
        if len(holdout_summary):
            for k, v in holdout_summary.iloc[0].to_dict().items():
                extras.append({"item": f"holdout_{k}", "value": v})
        if len(comparison) and "recommended" in comparison.columns:
            extras.append(
                {
                    "item": "recommended_model",
                    "value": recommended_model_id(comparison, default=cfg.model_id),
                }
            )
        extras.append({"item": "run_history_id", "value": record["run_id"]})
        extras.append(
            {
                "item": "run_history_report",
                "value": record["paths"].get("report_html") or record["paths"]["report_md"],
            }
        )
        extras.append({"item": "run_history_report_md", "value": record["paths"]["report_md"]})
        if extras:
            quality = pd.concat([quality, pd.DataFrame(extras)], ignore_index=True)

        out_path = None
        if output_xlsx is not None:
            out_path = export_audit_workbook(
                output_xlsx,
                fit=fit,
                bridge=bridge,
                target=target_ordinario,
                predictions=preds,
                predictions_supported=supported,
                factor_summary=factors,
                quality_report=quality,
                preflight=pf.as_dataframe(),
                cv_table=cv_table,
                config=cfg.to_dict(),
                audit=audit_summary(bridge_panel),
                model_comparison=comparison if len(comparison) else None,
                calibration=calib.as_dataframe() if calib is not None else None,
                holdout_summary=holdout_summary if len(holdout_summary) else None,
                holdout_detail=holdout_detail if len(holdout_detail) else None,
                sensitivity=sens if len(sens) else None,
            )
            log_operation(record, "export_excel", {"path": str(out_path)})

        report = finalize_run(
            record,
            status="ok",
            bridge_panel=bridge_panel,
            target=target_ordinario,
            bridge_ratios=bridge,
            predictions=preds,
            preflight_df=pf.as_dataframe(),
            cv_table=cv_table,
            fit_summary={
                "model_id": fit.model_id,
                "backend": fit.backend,
                "metric": fit.metric,
                "bridge_n": fit.bridge_n,
                "diagnostics": fit.diagnostics,
            },
            output_xlsx=out_path,
            warnings=pf.warnings,
        )
        fit.diagnostics["run_history_id"] = record["run_id"]
        fit.diagnostics["run_history_report"] = str(report)
        fit.diagnostics["run_history_report_md"] = record["paths"].get("report_md")
        fit.diagnostics["run_history_report_html"] = record["paths"].get("report_html")
        return fit, bridge, preds, out_path, pf, cv_table
    except Exception as exc:  # noqa: BLE001
        log_operation(record, "failed", {"error": str(exc)})
        finalize_run(
            record,
            status="failed",
            bridge_panel=bridge_panel,
            target=target_ordinario,
            errors=[str(exc)],
            output_xlsx=output_xlsx,
        )
        raise


def concat_panels(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No panels to concatenate")
    return pd.concat(frames, ignore_index=True)
