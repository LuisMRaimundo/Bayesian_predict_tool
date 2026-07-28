"""Held-out note validation for technique transfer (bridge self-check)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.fit import _effect_from_fit, fit_model, predict_transfer


def holdout_bridge_validation(
    bridge: pd.DataFrame,
    *,
    model_id: str = "M2_midi_gam",
    metric: str = "EWSD_score_acoustic_balanced",
    holdout_frac: float = 0.25,
    seed: int = 0,
    min_train: int = 6,
) -> pd.DataFrame:
    """Hold out a random fraction of bridge MIDI notes; score δ and factor errors.

    This is the in-repo validation pack when external measured technique curves
    on the target corpus are unavailable.
    """
    df = bridge.dropna(subset=["log_ratio", "midi", "technique", "dynamic"]).copy()
    if len(df) < min_train + 3:
        summary = pd.DataFrame(
            [{"status": "skipped", "reason": f"n={len(df)} too small", "n": len(df)}]
        )
        return pd.DataFrame(), summary

    rng = np.random.default_rng(seed)
    midis = sorted(df["midi"].astype(float).unique())
    n_hold = max(1, int(round(len(midis) * holdout_frac)))
    hold_midis = set(rng.choice(midis, size=min(n_hold, len(midis)), replace=False).tolist())
    train = df[~df["midi"].isin(hold_midis)]
    test = df[df["midi"].isin(hold_midis)]
    if len(train) < min_train or len(test) == 0:
        summary = pd.DataFrame(
            [{"status": "skipped", "reason": "holdout split empty", "n": len(df)}]
        )
        return pd.DataFrame(), summary

    fit = fit_model(train, model_id=model_id, metric=metric)
    rows = []
    for _, r in test.iterrows():
        delta, se, flag = _effect_from_fit(
            fit, str(r["technique"]), str(r["dynamic"]), float(r["midi"])
        )
        yt = float(r["log_ratio"])
        rows.append(
            {
                "status": "ok",
                "technique": r["technique"],
                "dynamic": r["dynamic"],
                "midi": float(r["midi"]),
                "note": r.get("note"),
                "log_true": yt,
                "log_pred": float(delta),
                "abs_log_err": abs(yt - float(delta)),
                "factor_true": float(np.exp(yt)),
                "factor_pred": float(np.exp(delta)),
                "se": float(se),
                "covered_95": abs(yt - float(delta)) <= 1.96 * max(float(se), 1e-6),
                "model_flag": flag,
                "model_id": model_id,
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "status": "ok",
        "model_id": model_id,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_hold_midis": int(len(hold_midis)),
        "mae_log": float(detail["abs_log_err"].mean()),
        "rmse_log": float(np.sqrt(np.mean((detail["log_true"] - detail["log_pred"]) ** 2))),
        "mape_factor": float(
            np.mean(
                np.abs(detail["factor_true"] - detail["factor_pred"])
                / np.clip(detail["factor_true"], 1e-6, None)
            )
        ),
        "coverage_95": float(detail["covered_95"].mean()),
        "holdout_frac": holdout_frac,
        "seed": seed,
    }
    # Attach summary as attrs via a one-row frame + detail return pattern:
    # callers use holdout_bridge_validation(...)[0] summary if we return tuple.
    return detail, pd.DataFrame([summary])


def holdout_against_measured_technique(
    fit,
    target_ordinario: pd.DataFrame,
    measured_technique: pd.DataFrame,
    *,
    technique: str,
    bridge_dynamics_by_technique: dict | None = None,
    max_dynamic_distance: int = 1,
    strict_dynamics: bool = True,
) -> pd.DataFrame:
    """Score predictions against an external measured special-technique panel.

    measured_technique must have columns: midi (or note), dynamic, value, technique.
    """
    preds = predict_transfer(
        fit,
        target_ordinario,
        [technique],
        bridge_dynamics_by_technique=bridge_dynamics_by_technique,
        max_dynamic_distance=max_dynamic_distance,
        strict_dynamics=strict_dynamics,
    )
    meas = measured_technique.copy()
    if "technique" in meas.columns:
        meas = meas.loc[meas["technique"].astype(str) == technique]
    key = ["midi", "dynamic"]
    if not set(key).issubset(meas.columns) or not set(key).issubset(preds.columns):
        raise ValueError("measured and predictions need midi + dynamic columns")
    m = meas.rename(columns={"value": "y_measured"})[key + ["y_measured"]]
    p = preds.merge(m, on=key, how="inner")
    if p.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no overlapping midi/dynamic keys"}])
    p["abs_err"] = (p["y_pred"] - p["y_measured"]).abs()
    p["abs_log_err"] = (np.log(np.clip(p["y_pred"], 1e-9, None)) - np.log(np.clip(p["y_measured"], 1e-9, None))).abs()
    p["status"] = "ok"
    return p
