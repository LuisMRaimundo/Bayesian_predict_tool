"""Acoustic plausibility priors and soft constraints for technique transfer."""

from __future__ import annotations

import numpy as np

# Soft expected direction / bounds on multiplicative EWSD factors.
# These are research priors for bowed-string special techniques, not hard laws.
TECHNIQUE_PRIOR = {
    "con_sordino": {
        "direction": "decrease",
        "factor_lo": 0.45,
        "factor_hi": 1.05,
        "prior_log": np.log(0.85),
        "prior_strength": 4.0,  # pseudo-observations toward prior
    },
    "sul_ponticello": {
        "direction": "increase",
        "factor_lo": 0.90,
        "factor_hi": 2.40,
        "prior_log": np.log(1.25),
        "prior_strength": 3.0,
    },
    "sul_tasto": {
        "direction": "decrease",
        "factor_lo": 0.50,
        "factor_hi": 1.10,
        "prior_log": np.log(0.80),
        "prior_strength": 3.0,
    },
    "natural_harmonics": {
        "direction": "decrease",
        "factor_lo": 0.25,
        "factor_hi": 1.05,
        "prior_log": np.log(0.65),
        "prior_strength": 4.0,
    },
    "artificial_harmonics": {
        "direction": "decrease",
        "factor_lo": 0.25,
        "factor_hi": 1.05,
        "prior_log": np.log(0.60),
        "prior_strength": 4.0,
    },
}


def shrink_log_ratio(log_ratio: float, technique: str, n_eff: float = 1.0) -> float:
    """Partial-pool a log-ratio toward the acoustic prior."""
    conf = TECHNIQUE_PRIOR.get(technique)
    if conf is None:
        return float(log_ratio)
    k = float(conf["prior_strength"])
    return float((n_eff * log_ratio + k * conf["prior_log"]) / (n_eff + k))


def clip_log_effect(log_effect: float, technique: str) -> tuple[float, bool]:
    """Clip log-effect to technique-specific plausible factor bounds."""
    conf = TECHNIQUE_PRIOR.get(technique)
    if conf is None:
        # generic mild clip
        lo, hi = np.log(0.25), np.log(2.5)
    else:
        lo, hi = np.log(conf["factor_lo"]), np.log(conf["factor_hi"])
    clipped = float(np.clip(log_effect, lo, hi))
    return clipped, clipped != float(log_effect)


def robust_log_weights(log_ratios: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Huber-like weights downweighting extreme bridge ratios."""
    x = np.asarray(log_ratios, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    z = np.abs(x - med) / (1.4826 * mad)
    w = np.where(z <= 2.5, 1.0, 2.5 / np.maximum(z, 1e-6))
    if se is not None:
        se = np.asarray(se, dtype=float)
        inv = 1.0 / np.clip(np.nan_to_num(se, nan=0.25) ** 2, 1e-3, None)
        inv = inv / np.nanmean(inv)
        w = w * inv
    return w
