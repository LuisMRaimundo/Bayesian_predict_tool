"""Sensitivity analyses: acoustic prior / winsor on/off."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..acoustics import TECHNIQUE_PRIOR
from ..bridge import build_log_ratios
from .blocked_cv import blocked_pitch_cv


def sensitivity_grid(
    bridge_panel: pd.DataFrame,
    *,
    metric: str = "EWSD_score_acoustic_balanced",
    model_id: str = "M2_midi_gam",
    require_same_collection: bool = True,
    max_dynamic_distance: int = 1,
) -> pd.DataFrame:
    """Compare CV MAE under winsor / coefficient-level prior on/off.

    Acoustic prior is applied once inside ``fit_model`` (not on each response).
    """
    rows = []
    for use_winsor in (True, False):
        for use_prior in (True, False):
            try:
                br = build_log_ratios(
                    bridge_panel,
                    metric=metric,
                    require_same_collection=require_same_collection,
                    max_dynamic_distance=max_dynamic_distance,
                    winsor_q=0.05 if use_winsor else 0.0,
                )
                cv = blocked_pitch_cv(
                    br,
                    model_id=model_id,
                    metric=metric,
                    winsor_q=0.0,  # already applied (or not) above; avoid double
                    apply_acoustic_prior=use_prior,
                    allow_m3_approx_fallback=True,
                )
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
                "application": "model_coefficient_once",
            }
        )
    return pd.DataFrame(rows)
