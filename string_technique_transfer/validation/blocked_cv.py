"""Blocked pitch-region cross-validation for transfer models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.fit import _effect_from_fit, fit_model


def _make_blocks(midis: np.ndarray, block_semitones: int = 12) -> list[np.ndarray]:
    midis = np.array(sorted(set(float(m) for m in midis if np.isfinite(m))))
    if len(midis) == 0:
        return []
    blocks = []
    start = midis[0]
    cur = [midis[0]]
    for m in midis[1:]:
        if m - start >= block_semitones:
            blocks.append(np.array(cur))
            start = m
            cur = [m]
        else:
            cur.append(m)
    if cur:
        blocks.append(np.array(cur))
    return blocks


def blocked_pitch_cv(
    bridge: pd.DataFrame,
    *,
    model_id: str = "M2_midi_gam",
    metric: str = "EWSD_score_acoustic_balanced",
    block_semitones: int = 12,
    min_train: int = 6,
) -> pd.DataFrame:
    """Leave-one-pitch-block-out CV on bridge log-ratios.

    Returns a metrics table (one row if successful, empty if CV impossible).
    """
    df = bridge.dropna(subset=["log_ratio", "midi", "technique", "dynamic"]).copy()
    if len(df) < min_train + 3:
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": f"n={len(df)} too small for blocked CV",
                    "n": len(df),
                }
            ]
        )

    blocks = _make_blocks(df["midi"].to_numpy(), block_semitones=block_semitones)
    if len(blocks) < 2:
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "need >=2 pitch blocks",
                    "n": len(df),
                    "n_blocks": len(blocks),
                }
            ]
        )

    y_true, y_pred, abs_err = [], [], []
    n_folds = 0
    for block in blocks:
        test = df[df["midi"].isin(block)]
        train = df[~df["midi"].isin(block)]
        if len(train) < min_train or len(test) == 0:
            continue
        try:
            fit = fit_model(train, model_id=model_id, metric=metric)
        except Exception:
            continue
        n_folds += 1
        for _, r in test.iterrows():
            delta, _se, _flag = _effect_from_fit(
                fit, str(r["technique"]), str(r["dynamic"]), float(r["midi"])
            )
            yt = float(r["log_ratio"])
            yp = float(delta)
            y_true.append(yt)
            y_pred.append(yp)
            abs_err.append(abs(yt - yp))

    if not y_true:
        return pd.DataFrame(
            [{"status": "skipped", "reason": "no successful folds", "n": len(df), "n_blocks": len(blocks)}]
        )

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    resid = yt - yp
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    # proportional error on factor scale
    factor_true = np.exp(yt)
    factor_pred = np.exp(yp)
    mape = float(np.mean(np.abs(factor_true - factor_pred) / np.clip(factor_true, 1e-6, None)))

    return pd.DataFrame(
        [
            {
                "status": "ok",
                "model_id": model_id,
                "n": len(df),
                "n_folds": n_folds,
                "n_test_points": len(yt),
                "block_semitones": block_semitones,
                "mae_log": mae,
                "rmse_log": rmse,
                "mape_factor": mape,
                "median_abs_log_err": float(np.median(np.abs(resid))),
                "bias_log": float(np.mean(resid)),
            }
        ]
    )
