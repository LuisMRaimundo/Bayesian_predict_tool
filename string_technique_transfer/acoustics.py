r"""Acoustic plausibility priors and soft constraints for technique transfer.

Aligned with the STE literature layer (Desktop Extrapolação repo):
- Peer-reviewed sources do **not** activate universal EWSD technique coefficients.
- Mute physics is frequency-dependent, S_muted(f)=S_ord(f)*A_{m,i}(f);
  this tool uses a **scalar log-ratio** approximation on a density metric.
- dB / loudness / bridge-mobility figures are **not** density multipliers.
- Heavy practice mutes are out of scope (performance con sordino only).
- Priors below are wide soft constraints + direction hints, overwritten by bridge data.
See LITERATURE_ALIGNMENT.md in this package root.
"""

from __future__ import annotations

import numpy as np

# Soft expected direction / bounds on multiplicative EWSD factors.
# Research priors for bowed-string special techniques — NOT response-level laws.
# Apply once at the **model-coefficient** level with n_eff = n_obs for that technique.
# Do not shrink every bridge row before fitting (that double-counts the prior).
TECHNIQUE_PRIOR = {
    "con_sordino": {
        "direction": "decrease",
        "factor_lo": 0.45,
        "factor_hi": 1.05,
        "prior_log": np.log(0.85),
        # Weak coefficient-level pseudo-count (relative to n_obs of the technique)
        "prior_strength": 1.0,
    },
    "sul_ponticello": {
        "direction": "increase",
        "factor_lo": 0.90,
        "factor_hi": 2.40,
        "prior_log": np.log(1.25),
        "prior_strength": 1.0,
    },
    "sul_tasto": {
        "direction": "decrease",
        "factor_lo": 0.50,
        "factor_hi": 1.10,
        "prior_log": np.log(0.80),
        "prior_strength": 1.0,
    },
    "natural_harmonics": {
        "direction": "decrease",
        "factor_lo": 0.25,
        "factor_hi": 1.05,
        "prior_log": np.log(0.65),
        "prior_strength": 1.0,
    },
    "artificial_harmonics": {
        "direction": "decrease",
        "factor_lo": 0.25,
        "factor_hi": 1.05,
        "prior_log": np.log(0.60),
        "prior_strength": 1.0,
    },
}


def shrink_log_ratio(log_ratio: float, technique: str, n_eff: float = 1.0) -> float:
    """Partial-pool a **technique coefficient** toward the acoustic prior.

    Intended for model-level means with ``n_eff`` ≈ number of bridge pairs for
    that technique — not for overwriting each observation (n_eff=1).
    """
    conf = TECHNIQUE_PRIOR.get(technique)
    if conf is None:
        return float(log_ratio)
    k = float(conf["prior_strength"])
    n = max(float(n_eff), 1e-6)
    return float((n * log_ratio + k * conf["prior_log"]) / (n + k))


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
