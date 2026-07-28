"""Runtime configuration for robust transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .dynamics import MAX_ADEQUATE_DISTANCE, ZENODO_DYNAMICS
from .models.base import MODEL_CHOICES


@dataclass
class TransferConfig:
    metric: str = "EWSD_score_acoustic_balanced"
    model_id: str = "M2_midi_gam"
    require_same_collection: bool = False
    strict_dynamics: bool = True
    max_dynamic_distance: int = MAX_ADEQUATE_DISTANCE
    run_blocked_cv: bool = True
    cv_block_semitones: int = 12
    min_bridge_pairs: int = 8
    min_supported_predictions: int = 1
    zenodo_dynamics: tuple[str, ...] = ZENODO_DYNAMICS

    def validate(self) -> None:
        if self.model_id not in MODEL_CHOICES:
            raise ValueError(f"model_id must be one of {MODEL_CHOICES}")
        if self.max_dynamic_distance < 0:
            raise ValueError("max_dynamic_distance must be >= 0")
        if self.cv_block_semitones < 1:
            raise ValueError("cv_block_semitones must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = TransferConfig()
