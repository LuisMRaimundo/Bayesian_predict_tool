"""Helpers to assemble multi-file / multi-dynamic bridge panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import concat_panels, load_and_clean
from .schema import is_ordinario


def build_bridge_from_paths(
    paths: list[str | Path],
    *,
    metric: str = "EWSD_score_acoustic_balanced",
    instrument: str | None = None,
) -> pd.DataFrame:
    """Load and concatenate research workbooks / CSVs into one bridge panel."""
    frames = []
    for p in paths:
        clean, _, _ = load_and_clean(p, metric=metric)
        if instrument and (clean["instrument"] == "unknown").all():
            clean = clean.copy()
            clean["instrument"] = instrument
            clean["corpus_id"] = clean["instrument"].astype(str) + "|" + clean["collection"].astype(str)
        frames.append(clean)
    return concat_panels(frames)


def bridge_coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Technique × dynamic × MIDI coverage summary for planning richer bridges."""
    df = panel.copy()
    if "is_ordinario" in df.columns:
        special = df.loc[~df["is_ordinario"]].copy()
        ord_df = df.loc[df["is_ordinario"]].copy()
    else:
        special = df.loc[~df["technique"].map(is_ordinario)].copy()
        ord_df = df.loc[df["technique"].map(is_ordinario)].copy()
    rows = []
    for (tech, dyn), g in special.groupby(["technique", "dynamic"], dropna=False):
        midis = set(g["midi"].dropna().astype(float))
        ord_same = ord_df.loc[ord_df["dynamic"].astype(str) == str(dyn), "midi"].dropna().astype(float)
        overlap = midis & set(ord_same)
        rows.append(
            {
                "technique": tech,
                "dynamic": dyn,
                "n_special": len(g),
                "n_midi_special": len(midis),
                "n_midi_overlap_ordinario": len(overlap),
                "midi_min": float(min(midis)) if midis else np.nan,
                "midi_max": float(max(midis)) if midis else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["technique", "dynamic"])
