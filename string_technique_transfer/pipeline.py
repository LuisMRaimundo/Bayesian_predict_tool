"""End-to-end local pipeline with preflight, robust transfer, and blocked CV."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .bridge import build_log_ratios, summarize_factors
from .clean import audit_summary, dedupe_panel
from .config import DEFAULT_CONFIG, TransferConfig
from .export.excel_audit import export_audit_workbook
from .io.loaders import load_panel
from .models.base import FitResult
from .models.fit import fit_model, predict_transfer
from .preflight import PreflightResult, preflight_transfer
from .quality import build_quality_report
from .validation.blocked_cv import blocked_pitch_cv


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
) -> tuple[FitResult, pd.DataFrame, pd.DataFrame, Path | None, PreflightResult, pd.DataFrame]:
    """Run full robust transfer.

    Returns
    -------
    fit, bridge, predictions_all, out_path, preflight, cv_table
    """
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

    pf = (
        PreflightResult(True, [], [], {"skipped": True})
        if skip_preflight
        else preflight_transfer(bridge_panel, target_ordinario, cfg)
    )
    if not pf.ok:
        raise ValueError(
            "Preflight failed:\n- " + "\n- ".join(pf.errors)
        )

    bridge = build_log_ratios(
        bridge_panel,
        metric=cfg.metric,
        require_same_collection=cfg.require_same_collection,
        max_dynamic_distance=cfg.max_dynamic_distance,
    )
    if techniques is None:
        techniques = sorted(bridge["technique"].unique().tolist())
    fit = fit_model(bridge, model_id=cfg.model_id, metric=cfg.metric)
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
    )
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

    factors = summarize_factors(bridge)
    quality = build_quality_report(
        bridge,
        preds,
        max_dynamic_distance=cfg.max_dynamic_distance,
        output_xlsx=output_xlsx,
    )
    # Append CV metrics into quality report
    if len(cv_table):
        extra = []
        row = cv_table.iloc[0].to_dict()
        for k, v in row.items():
            extra.append({"item": f"cv_{k}", "value": v})
        quality = pd.concat([quality, pd.DataFrame(extra)], ignore_index=True)

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
        )
    return fit, bridge, preds, out_path, pf, cv_table


def concat_panels(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No panels to concatenate")
    return pd.concat(frames, ignore_index=True)
