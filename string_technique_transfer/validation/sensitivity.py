"""Sensitivity analyses: acoustic prior / winsor on/off."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..acoustics import TECHNIQUE_PRIOR, shrink_log_ratio
from ..bridge import build_log_ratios
from ..models.fit import fit_model
from .blocked_cv import blocked_pitch_cv


def _bridge_variant(
    panel: pd.DataFrame,
    *,
    metric: str,
    require_same_collection: bool,
    max_dynamic_distance: int,
    use_winsor: bool,
    use_prior: bool,
) -> pd.DataFrame:
    br = build_log_ratios(
        panel,
        metric=metric,
        require_same_collection=require_same_collection,
        max_dynamic_distance=max_dynamic_distance,
        winsor_q=0.05 if use_winsor else 0.0,
    )
    if not use_prior:
        # undo shrink: restore from raw then re-winsor only
        if "log_ratio_raw" in br.columns:
            if use_winsor:
                cleaned = []
                for tech, g in br.groupby("technique"):
                    g = g.copy()
                    lo = g["log_ratio_raw"].quantile(0.05)
                    hi = g["log_ratio_raw"].quantile(0.95)
                    g["log_ratio"] = g["log_ratio_raw"].clip(lo, hi)
                    g["factor"] = np.exp(g["log_ratio"])
                    cleaned.append(g)
                br = pd.concat(cleaned, ignore_index=True)
            else:
                br = br.copy()
                br["log_ratio"] = br["log_ratio_raw"]
                br["factor"] = np.exp(br["log_ratio"])
    return br


def sensitivity_grid(
    bridge_panel: pd.DataFrame,
    *,
    metric: str = "EWSD_score_acoustic_balanced",
    model_id: str = "M2_midi_gam",
    require_same_collection: bool = False,
    max_dynamic_distance: int = 1,
) -> pd.DataFrame:
    """Compare CV MAE under winsor/prior on/off combinations."""
    rows = []
    for use_winsor in (True, False):
        for use_prior in (True, False):
            try:
                # winsor_q=0 disables winsor in build; prior always applied inside build —
                # so for use_prior False we rebuild from raw after.
                br = build_log_ratios(
                    bridge_panel,
                    metric=metric,
                    require_same_collection=require_same_collection,
                    max_dynamic_distance=max_dynamic_distance,
                    winsor_q=0.05 if use_winsor else 0.0,
                )
                if not use_prior and "log_ratio_raw" in br.columns:
                    br = br.copy()
                    if use_winsor:
                        parts = []
                        for _, g in br.groupby("technique"):
                            g = g.copy()
                            lo, hi = g["log_ratio_raw"].quantile(0.05), g["log_ratio_raw"].quantile(0.95)
                            g["log_ratio"] = g["log_ratio_raw"].clip(lo, hi)
                            g["factor"] = np.exp(g["log_ratio"])
                            parts.append(g)
                        br = pd.concat(parts, ignore_index=True)
                    else:
                        br["log_ratio"] = br["log_ratio_raw"]
                        br["factor"] = np.exp(br["log_ratio"])
                cv = blocked_pitch_cv(br, model_id=model_id, metric=metric)
                row = {
                    "winsor": use_winsor,
                    "acoustic_prior": use_prior,
                    "n_bridge": len(br),
                    "status": cv.iloc[0]["status"] if len(cv) else "empty",
                }
                if len(cv) and cv.iloc[0].get("status") == "ok":
                    row.update(
                        {
                            "mae_log": cv.iloc[0]["mae_log"],
                            "rmse_log": cv.iloc[0]["rmse_log"],
                            "mape_factor": cv.iloc[0]["mape_factor"],
                        }
                    )
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "winsor": use_winsor,
                        "acoustic_prior": use_prior,
                        "status": "error",
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def prior_table() -> pd.DataFrame:
    rows = []
    for tech, conf in TECHNIQUE_PRIOR.items():
        rows.append(
            {
                "technique": tech,
                "prior_factor": float(np.exp(conf["prior_log"])),
                "factor_lo": conf["factor_lo"],
                "factor_hi": conf["factor_hi"],
                "prior_strength": conf["prior_strength"],
                "direction": conf["direction"],
            }
        )
    return pd.DataFrame(rows)
