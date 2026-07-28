"""Quality assessment for transfer predictions."""

from __future__ import annotations

import pandas as pd

from .dynamics import MAX_ADEQUATE_DISTANCE, supported_zenodo_dynamics_for_bridge


def build_quality_report(
    bridge: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    max_dynamic_distance: int = MAX_ADEQUATE_DISTANCE,
) -> pd.DataFrame:
    rows = []
    n_pred = len(predictions)
    if n_pred and "support_level" in predictions.columns:
        n_sup = int(
            predictions["support_level"].isin(["supported", "supported_outlier_target"]).sum()
        )
    else:
        n_sup = 0
    n_ext = n_pred - n_sup
    rows.append({"item": "n_bridge_pairs", "value": len(bridge)})
    rows.append({"item": "n_predictions_total", "value": n_pred})
    rows.append({"item": "n_predictions_supported", "value": n_sup})
    rows.append({"item": "n_predictions_extrapolated", "value": n_ext})
    rows.append(
        {
            "item": "supported_fraction",
            "value": round(n_sup / n_pred, 3) if n_pred else 0.0,
        }
    )
    if len(bridge):
        rows.append(
            {
                "item": "bridge_techniques",
                "value": ", ".join(sorted(bridge["technique"].astype(str).unique())),
            }
        )
        rows.append(
            {
                "item": "bridge_dynamics",
                "value": ", ".join(sorted(bridge["dynamic"].astype(str).unique())),
            }
        )
        rows.append(
            {
                "item": "bridge_midi_min",
                "value": float(bridge["midi"].min()),
            }
        )
        rows.append(
            {
                "item": "bridge_midi_max",
                "value": float(bridge["midi"].max()),
            }
        )
        rows.append(
            {
                "item": "transport_prior_fraction",
                "value": round(float(bridge["is_transport_prior"].mean()), 3)
                if "is_transport_prior" in bridge
                else None,
            }
        )
        for tech, g in bridge.groupby("technique"):
            zsup = supported_zenodo_dynamics_for_bridge(
                g["dynamic"].unique(), max_distance=max_dynamic_distance
            )
            rows.append(
                {
                    "item": f"zenodo_dynamics_supported[{tech}]",
                    "value": ", ".join(zsup) if zsup else "(none)",
                }
            )
    if n_pred and "target_outlier" in predictions.columns:
        rows.append(
            {
                "item": "target_outlier_flags",
                "value": int(predictions["target_outlier"].sum()),
            }
        )
    if n_pred and "factor_clipped" in predictions.columns:
        rows.append(
            {
                "item": "factor_clipped_count",
                "value": int(predictions["factor_clipped"].sum()),
            }
        )
    rows.append(
        {
            "item": "recommendation",
            "value": (
                "Use Predictions_supported only. Treat extrapolated rows as exploratory. "
                "Do not label any row as measured collection data."
                if n_sup
                else "No supported predictions under the adequate-dynamic policy; enrich the bridge."
            ),
        }
    )
    return pd.DataFrame(rows)
