"""Excel audit workbook for predictions and provenance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models.base import FitResult


def export_audit_workbook(
    path: str | Path,
    *,
    fit: FitResult,
    bridge: pd.DataFrame,
    target: pd.DataFrame,
    predictions: pd.DataFrame,
    predictions_supported: pd.DataFrame | None = None,
    factor_summary: pd.DataFrame | None = None,
    quality_report: pd.DataFrame | None = None,
    preflight: pd.DataFrame | None = None,
    cv_table: pd.DataFrame | None = None,
    config: dict | None = None,
    audit: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_sup = len(predictions_supported) if predictions_supported is not None else 0
    readme = pd.DataFrame(
        {
            "Field": [
                "estimate_class",
                "scientific_label",
                "model_id",
                "backend",
                "metric",
                "bridge_n",
                "n_predictions_total",
                "n_predictions_supported",
                "primary_sheet",
                "warning",
            ],
            "Value": [
                "model_derived_synthetic",
                fit.label,
                fit.model_id,
                fit.backend,
                fit.metric,
                fit.bridge_n,
                len(predictions),
                n_sup,
                "Predictions_supported",
                "Use Predictions_supported for robust work. "
                "Predictions_all includes extrapolated dynamics/registers for diagnostics only. "
                "Never label rows as measured IOWA/ORCHIDEA observations. "
                "See METHODOLOGY.md in the tool folder.",
            ],
        }
    )

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        readme.to_excel(xl, sheet_name="README", index=False)
        if config is not None:
            pd.DataFrame([{"key": k, "value": str(v)} for k, v in config.items()]).to_excel(
                xl, sheet_name="Config", index=False
            )
        if preflight is not None:
            preflight.to_excel(xl, sheet_name="Preflight", index=False)
        if quality_report is not None:
            quality_report.to_excel(xl, sheet_name="Quality_report", index=False)
        if cv_table is not None:
            cv_table.to_excel(xl, sheet_name="Blocked_CV", index=False)
        if audit:
            pd.DataFrame([audit]).to_excel(xl, sheet_name="Audit", index=False)
        bridge.to_excel(xl, sheet_name="Bridge_log_ratios", index=False)
        if factor_summary is not None:
            factor_summary.to_excel(xl, sheet_name="Factor_summary", index=False)
        if fit.effects is not None:
            fit.effects.to_excel(xl, sheet_name="Model_effects", index=False)
        target.to_excel(xl, sheet_name="Target_ordinario", index=False)
        if predictions_supported is not None:
            predictions_supported.to_excel(xl, sheet_name="Predictions_supported", index=False)
        predictions.to_excel(xl, sheet_name="Predictions_all", index=False)
        needed = {"dynamic", "bridge_dynamic_used", "dynamic_match", "support_level"}
        if needed.issubset(predictions.columns):
            cols = ["dynamic", "bridge_dynamic_used", "dynamic_match", "support_level"]
            if "dynamic_adequate" in predictions.columns:
                cols.append("dynamic_adequate")
            (
                predictions.groupby(cols, dropna=False)
                .size()
                .reset_index(name="n")
                .to_excel(xl, sheet_name="Dynamic_matching", index=False)
            )
        pd.DataFrame(
            [
                {
                    "formula_prediction": "y_pred = y_ordinario * exp(log_effect)",
                    "formula_interval": "y_pred * exp(+/-1.96 * combined_se_log)",
                    "dynamic_policy": "Supported only for adequate pairs: pp<->{pp,p}, mf<->{mf,mp}, ff<->{ff,f}.",
                    "acoustic_policy": "Winsorize + shrink toward technique priors + clip plausible factor bounds.",
                    "register_policy": "Outside bridge MIDI: shrunk global effect, not spline extrapolation.",
                    "validation": "Blocked pitch-region CV metrics in Blocked_CV sheet.",
                }
            ]
        ).to_excel(xl, sheet_name="Formulas", index=False)

    return path
