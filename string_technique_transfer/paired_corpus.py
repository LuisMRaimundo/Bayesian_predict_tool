"""Paired-corpus scientific readiness helpers.

A *genuine paired corpus* has ordinario and special-technique observations that
share performer, instrument, microphone, room, and processing chain — i.e.
``special_corpus_id == ordinario_corpus_id`` (same-collection pairs).

Cross-collection transport can still be modelled, but cannot identify a pure
technique effect when technique is confounded with corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PairedCorpusReport:
    n_pairs: int
    n_same_collection: int
    n_transport_prior: int
    paired_fraction: float
    techniques_same_collection: list[str]
    techniques_transport_only: list[str]
    scientific_tier: str  # "paired" | "mixed" | "transport_only" | "empty"
    message: str

    @property
    def is_paired_ready(self) -> bool:
        return self.scientific_tier in {"paired", "mixed"} and self.n_same_collection >= 8


def assess_paired_corpus(bridge: pd.DataFrame) -> PairedCorpusReport:
    if bridge is None or len(bridge) == 0:
        return PairedCorpusReport(
            0, 0, 0, 0.0, [], [], "empty", "No bridge pairs."
        )
    df = bridge.copy()
    if "is_transport_prior" in df.columns:
        transport = df["is_transport_prior"].astype(bool)
    elif {"special_corpus_id", "ordinario_corpus_id"}.issubset(df.columns):
        transport = df["special_corpus_id"].astype(str) != df["ordinario_corpus_id"].astype(str)
    else:
        transport = pd.Series([False] * len(df), index=df.index)

    n = len(df)
    n_tr = int(transport.sum())
    n_same = n - n_tr
    frac = float(n_same / n) if n else 0.0
    same_techs = sorted(df.loc[~transport, "technique"].astype(str).unique().tolist()) if n_same else []
    tr_techs = sorted(df.loc[transport, "technique"].astype(str).unique().tolist()) if n_tr else []
    only_tr = sorted(set(tr_techs) - set(same_techs))

    if n_same == 0:
        tier = "transport_only"
        msg = (
            "All pairs are cross-collection transport priors. "
            "Technique is confounded with corpus; results are exploratory transport estimates only. "
            "Add a genuine paired corpus (ordinario + technique, same chain) for scientific identification."
        )
    elif n_tr == 0:
        tier = "paired"
        msg = f"All {n_same} pairs are same-collection; paired-corpus tier."
    else:
        tier = "mixed"
        msg = (
            f"{n_same}/{n} pairs same-collection ({frac:.0%}); "
            f"{n_tr} transport priors remain exploratory."
        )

    return PairedCorpusReport(
        n_pairs=n,
        n_same_collection=n_same,
        n_transport_prior=n_tr,
        paired_fraction=round(frac, 3),
        techniques_same_collection=same_techs,
        techniques_transport_only=only_tr,
        scientific_tier=tier,
        message=msg,
    )
