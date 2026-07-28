"""Runtime configuration for robust transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .dynamics import MAX_ADEQUATE_DISTANCE, ZENODO_DYNAMICS
from .models.base import MODEL_CHOICES


@dataclass
class TransferConfig:
    metric: str = "EWSD_score_acoustic_balanced"
    # Re-audit on cross-corpus violin bridges: M1 currently best exploratory CV
    model_id: str = "M1_register_dynamic"
    # Audit default: do not silently invent same-collection pairs
    require_same_collection: bool = True
    strict_dynamics: bool = True
    max_dynamic_distance: int = MAX_ADEQUATE_DISTANCE
    run_blocked_cv: bool = True
    run_model_comparison: bool = True
    run_calibration: bool = True
    run_holdout: bool = True
    run_sensitivity: bool = True
    auto_select_model: bool = False
    # Acoustic prior applied once at model coefficients (never overwrite each response)
    apply_acoustic_prior: bool = True
    # When False, explicit M3 requests fail instead of silent hierarchical_approx_*
    allow_m3_approx_fallback: bool = False
    cv_block_semitones: int = 12
    min_bridge_pairs: int = 8
    min_supported_predictions: int = 1
    holdout_frac: float = 0.25
    zenodo_dynamics: tuple[str, ...] = ZENODO_DYNAMICS

    def validate(self) -> None:
        if self.model_id not in MODEL_CHOICES:
            raise ValueError(f"model_id must be one of {MODEL_CHOICES}")
        if self.max_dynamic_distance < 0:
            raise ValueError("max_dynamic_distance must be >= 0")
        if self.cv_block_semitones < 1:
            raise ValueError("cv_block_semitones must be >= 1")
        if not (0.05 <= self.holdout_frac <= 0.5):
            raise ValueError("holdout_frac must be in [0.05, 0.5]")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = TransferConfig()
