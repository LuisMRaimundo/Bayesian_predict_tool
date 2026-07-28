from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

MODEL_CHOICES = (
    "M0_global_factor",
    "M1_register_dynamic",
    "M2_midi_gam",
    "M3_hierarchical_bayes",
)


@dataclass
class FitResult:
    model_id: str
    backend: str
    metric: str
    bridge_n: int
    params: dict[str, Any] = field(default_factory=dict)
    effects: pd.DataFrame | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    label: str = (
        "Model-derived synthetic estimates of missing technique conditions, "
        "transported from external corpora and conditioned on observed ordinario profiles."
    )

    def summary_text(self) -> str:
        lines = [
            f"Model: {self.model_id}",
            f"Backend: {self.backend}",
            f"Metric: {self.metric}",
            f"Bridge n: {self.bridge_n}",
        ]
        for k, v in self.diagnostics.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
