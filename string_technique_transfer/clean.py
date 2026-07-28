"""Deduplication and audit helpers."""

from __future__ import annotations

import pandas as pd


def dedupe_panel(df: pd.DataFrame, prefer_ci: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact duplicates on instrument/collection/technique/dynamic/midi/metric.

    Returns (clean_df, duplicate_audit_df).
    """
    keys = ["instrument", "collection", "technique", "dynamic", "midi", "metric"]
    work = df.copy()
    work = work.dropna(subset=["value", "midi"])
    dup_mask = work.duplicated(subset=keys, keep=False)
    audit = work.loc[dup_mask].sort_values(keys)

    if prefer_ci and {"ci_low", "ci_high"}.issubset(work.columns):
        width = (work["ci_high"] - work["ci_low"]).astype(float)
        work = work.assign(_ci_width=width.fillna(float("inf")))
        work = work.sort_values(["_ci_width"], ascending=True)
        clean = work.drop_duplicates(subset=keys, keep="first").drop(columns=["_ci_width"])
    else:
        clean = work.drop_duplicates(subset=keys, keep="first")

    clean = clean.reset_index(drop=True)
    return clean, audit.reset_index(drop=True)


def audit_summary(df: pd.DataFrame) -> dict:
    keys = ["instrument", "collection", "technique", "dynamic", "midi", "metric"]
    n = len(df)
    n_unique = df.drop_duplicates(subset=keys).shape[0] if n else 0
    return {
        "n_rows": int(n),
        "n_unique_keys": int(n_unique),
        "n_duplicate_rows": int(n - n_unique),
        "instruments": sorted(df["instrument"].dropna().unique().tolist()) if n else [],
        "techniques": sorted(df["technique"].dropna().unique().tolist()) if n else [],
        "collections": sorted(df["collection"].dropna().unique().tolist()) if n else [],
        "metrics": sorted(df["metric"].dropna().unique().tolist()) if n else [],
    }
