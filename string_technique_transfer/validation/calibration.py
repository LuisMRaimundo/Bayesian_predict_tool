"""Calibrate predictive uncertainty from blocked-CV residuals / conformal scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..models.fit import _effect_from_fit, fit_model
from .blocked_cv import _make_blocks


@dataclass
class CalibrationResult:
    status: str
    n_test: int = 0
    n_folds: int = 0
    resid_sd: float = 0.18
    mae_log: float = np.nan
    conformal_q90: float = 0.25
    conformal_q95: float = 0.35
    coverage_raw_95: float = np.nan
    scale_factor: float = 1.0
    inflate_dynamic: float = 0.25
    inflate_register: float = 0.12
    inflate_outlier: float = 0.20
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{"item": k, "value": v} for k, v in self.to_dict().items()])


def calibrate_from_bridge(
    bridge: pd.DataFrame,
    *,
    model_id: str = "M2_midi_gam",
    metric: str = "EWSD_score_acoustic_balanced",
    block_semitones: int = 12,
    min_train: int = 6,
) -> CalibrationResult:
    """Estimate residual scale and conformal quantiles on pitch-blocked folds."""
    df = bridge.dropna(subset=["log_ratio", "midi", "technique", "dynamic"]).copy()
    if len(df) < min_train + 3:
        return CalibrationResult(
            status="default_priors",
            note=f"n={len(df)} too small; using default inflate constants",
        )

    blocks = _make_blocks(df["midi"].to_numpy(), block_semitones=block_semitones)
    if len(blocks) < 2:
        return CalibrationResult(status="default_priors", note="need >=2 pitch blocks")

    resid, abs_resid, inside95 = [], [], []
    n_folds = 0
    for block in blocks:
        test = df[df["midi"].isin(block)]
        train = df[~df["midi"].isin(block)]
        if len(train) < min_train or len(test) == 0:
            continue
        try:
            fit = fit_model(
                train, model_id=model_id, metric=metric, allow_m3_approx_fallback=True
            )
        except Exception:
            continue
        n_folds += 1
        for _, r in test.iterrows():
            delta, se, _flag = _effect_from_fit(
                fit, str(r["technique"]), str(r["dynamic"]), float(r["midi"])
            )
            e = float(r["log_ratio"]) - float(delta)
            resid.append(e)
            abs_resid.append(abs(e))
            inside95.append(abs(e) <= 1.96 * max(float(se), 1e-6))

    if len(resid) < 4:
        return CalibrationResult(
            status="default_priors",
            n_folds=n_folds,
            note="too few test residuals; using defaults",
        )

    r = np.asarray(resid, dtype=float)
    ar = np.asarray(abs_resid, dtype=float)
    resid_sd = float(max(np.std(r, ddof=1), 0.05))
    mae = float(np.mean(ar))
    q90 = float(np.quantile(ar, 0.90))
    q95 = float(np.quantile(ar, 0.95))
    cov = float(np.mean(inside95))

    # Scale nominal SE so empirical coverage moves toward 95%
    # If cov << 0.95, scale_factor > 1
    if cov < 0.5:
        scale = 1.6
    elif cov < 0.8:
        scale = 1.35
    elif cov < 0.92:
        scale = 1.15
    elif cov > 0.98:
        scale = 0.90
    else:
        scale = 1.0

    # Recalibrate inflate constants from residual scale
    inflate_dyn = float(np.clip(0.15 + 0.5 * resid_sd, 0.12, 0.40))
    inflate_reg = float(np.clip(0.08 + 0.35 * resid_sd, 0.08, 0.30))
    inflate_out = float(np.clip(0.12 + 0.4 * resid_sd, 0.10, 0.35))

    return CalibrationResult(
        status="ok",
        n_test=len(resid),
        n_folds=n_folds,
        resid_sd=resid_sd,
        mae_log=mae,
        conformal_q90=q90,
        conformal_q95=max(q95, q90),
        coverage_raw_95=cov,
        scale_factor=scale,
        inflate_dynamic=inflate_dyn,
        inflate_register=inflate_reg,
        inflate_outlier=inflate_out,
        note="blocked-CV residual calibration",
    )


def apply_conformal_halfwidth(se: float, calib: CalibrationResult | None) -> float:
    """Return log-scale half-width for ~95% interval (max of scaled SE and conformal)."""
    if calib is None or calib.status != "ok":
        return float(1.96 * se)
    scaled = 1.96 * float(se) * float(calib.scale_factor)
    return float(max(scaled, calib.conformal_q95))
