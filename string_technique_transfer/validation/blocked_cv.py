"""Blocked pitch-region cross-validation for transfer models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..bridge import winsorize_log_ratios
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


def _prepare_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    winsor_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-winsorize from raw inside the training fold; apply train bounds to test y_true."""
    if "log_ratio_raw" not in train.columns:
        return train, test
    train_w = winsorize_log_ratios(train, winsor_q=winsor_q)
    test_w = test.copy()
    # Evaluate against train-fold winsor thresholds (no test leakage of quantiles)
    bounds = (
        train_w.groupby("technique")[["winsor_lo", "winsor_hi"]].first()
        if winsor_q and winsor_q > 0
        else None
    )
    if bounds is not None and len(bounds):
        ys = []
        for _, r in test_w.iterrows():
            raw = float(r["log_ratio_raw"])
            tech = r["technique"]
            if tech in bounds.index:
                lo, hi = float(bounds.loc[tech, "winsor_lo"]), float(bounds.loc[tech, "winsor_hi"])
                ys.append(float(np.clip(raw, lo, hi)))
            else:
                ys.append(raw)
        test_w["log_ratio"] = ys
    else:
        test_w["log_ratio"] = test_w["log_ratio_raw"]
    return train_w, test_w


def blocked_pitch_cv(
    bridge: pd.DataFrame,
    *,
    model_id: str = "M2_midi_gam",
    metric: str = "EWSD_score_acoustic_balanced",
    block_semitones: int = 12,
    min_train: int = 6,
    winsor_q: float = 0.05,
    apply_acoustic_prior: bool = True,
    allow_m3_approx_fallback: bool = True,
) -> pd.DataFrame:
    """Leave-one-pitch-block-out CV on bridge log-ratios.

    Winsorization (and model prior application) are estimated inside each training fold.
    ``allow_m3_approx_fallback`` defaults True for CV so thin folds do not abort the pack.
    """
    df = bridge.dropna(subset=["log_ratio", "midi", "technique", "dynamic"]).copy()
    if "log_ratio_raw" not in df.columns:
        df["log_ratio_raw"] = df["log_ratio"]
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
        train_w, test_w = _prepare_fold(train, test, winsor_q=winsor_q)
        try:
            fit = fit_model(
                train_w,
                model_id=model_id,
                metric=metric,
                apply_acoustic_prior=apply_acoustic_prior,
                allow_m3_approx_fallback=allow_m3_approx_fallback,
            )
        except Exception:
            continue
        n_folds += 1
        for _, r in test_w.iterrows():
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
                "winsor_inside_fold": True,
            }
        ]
    )
