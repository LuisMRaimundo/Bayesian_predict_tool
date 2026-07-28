"""Formal blocked-CV comparison across M0–M3 with auto-recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.base import MODEL_CHOICES
from .blocked_cv import blocked_pitch_cv


def compare_models(
    bridge: pd.DataFrame,
    *,
    metric: str = "EWSD_score_acoustic_balanced",
    model_ids: tuple[str, ...] | None = None,
    block_semitones: int = 12,
) -> pd.DataFrame:
    """Return one row per model with CV metrics + rank/recommendation flags."""
    ids = list(model_ids or MODEL_CHOICES)
    rows = []
    for mid in ids:
        tab = blocked_pitch_cv(
            bridge,
            model_id=mid,
            metric=metric,
            block_semitones=block_semitones,
            allow_m3_approx_fallback=True,
        )
        if len(tab) == 0:
            rows.append({"model_id": mid, "status": "empty", "mae_log": np.nan, "rmse_log": np.nan})
            continue
        r = tab.iloc[0].to_dict()
        r["model_id"] = mid
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    ok = out["status"].astype(str).eq("ok") if "status" in out.columns else pd.Series([False] * len(out))
    out["rankable"] = ok & out["mae_log"].notna()
    out["rank"] = pd.NA
    out["recommended"] = False
    if out["rankable"].any():
        ranked = out.loc[out["rankable"]].sort_values(["mae_log", "rmse_log"], ascending=True)
        for i, idx in enumerate(ranked.index, start=1):
            out.loc[idx, "rank"] = i
        best = str(ranked.iloc[0]["model_id"])
        # Prefer M2 if M3 has fewer successful folds
        if best == "M3_hierarchical_bayes" and (ranked["model_id"] == "M2_midi_gam").any():
            m2 = ranked.loc[ranked["model_id"] == "M2_midi_gam"].iloc[0]
            m3 = ranked.loc[ranked["model_id"] == "M3_hierarchical_bayes"].iloc[0]
            if int(m3.get("n_folds", 0) or 0) < int(m2.get("n_folds", 0) or 0):
                best = "M2_midi_gam"
        out["recommended"] = out["model_id"].eq(best)
    else:
        out["recommended"] = out["model_id"].eq("M2_midi_gam")
    return out


def recommended_model_id(comparison: pd.DataFrame, default: str = "M2_midi_gam") -> str:
    if comparison is None or len(comparison) == 0:
        return default
    if "recommended" in comparison.columns and comparison["recommended"].any():
        return str(comparison.loc[comparison["recommended"]].iloc[0]["model_id"])
    return default
