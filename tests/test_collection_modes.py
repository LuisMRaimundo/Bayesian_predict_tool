"""Strict same-collection vs explicit transport_prior modes."""

import pandas as pd
import pytest

from string_technique_transfer.bridge import build_log_ratios
from string_technique_transfer.config import TransferConfig
from string_technique_transfer.preflight import preflight_transfer


def _cross_panel():
    rows = []
    for midi in (60.0, 62.0, 64.0):
        rows.append(
            dict(
                instrument="Violin",
                collection="ord",
                technique="ordinario",
                dynamic="f",
                midi=midi,
                metric="EWSD_score_acoustic_balanced",
                value=10.0,
                corpus_id="Violin|ord",
                is_ordinario=True,
            )
        )
        rows.append(
            dict(
                instrument="Violin",
                collection="sord",
                technique="con_sordino",
                dynamic="f",
                midi=midi,
                metric="EWSD_score_acoustic_balanced",
                value=8.2,
                corpus_id="Violin|sord",
                is_ordinario=False,
            )
        )
    return pd.DataFrame(rows)


def _same_panel():
    rows = []
    for midi in (60.0, 62.0, 64.0):
        for tech, y, ord_flag in (("ordinario", 10.0, True), ("con_sordino", 8.2, False)):
            rows.append(
                dict(
                    instrument="Violin",
                    collection="lab",
                    technique=tech,
                    dynamic="f",
                    midi=midi,
                    metric="EWSD_score_acoustic_balanced",
                    value=y,
                    corpus_id="Violin|lab",
                    is_ordinario=ord_flag,
                )
            )
    return pd.DataFrame(rows)


def _target():
    rows = []
    for dyn in ("pp", "mf", "ff"):
        rows.append(
            dict(
                instrument="Violin",
                collection="MEDIA",
                technique="ordinario",
                dynamic=dyn,
                midi=60.0,
                metric="EWSD_score_acoustic_balanced",
                value=15.0,
                corpus_id="Violin|MEDIA",
                is_ordinario=True,
            )
        )
    return pd.DataFrame(rows)


def test_strict_mode_rejects_cross_collection_preflight():
    cfg = TransferConfig(require_same_collection=True, run_blocked_cv=False)
    pf = preflight_transfer(_cross_panel(), _target(), cfg)
    assert not pf.ok
    assert any("bridge" in e.lower() or "log-ratio" in e.lower() or "collection" in e.lower() for e in pf.errors)


def test_transport_mode_accepts_and_labels_transport_prior():
    br = build_log_ratios(_cross_panel(), require_same_collection=False)
    assert len(br) == 3
    assert br["is_transport_prior"].all()
    assert set(br["support_flag"]) == {"transport_prior"}
    cfg = TransferConfig(require_same_collection=False, run_blocked_cv=False)
    pf = preflight_transfer(_cross_panel(), _target(), cfg)
    assert pf.ok
    assert float(pf.summary.get("transport_prior_fraction", 0)) == 1.0


def test_same_collection_passes_strict_mode():
    br = build_log_ratios(_same_panel(), require_same_collection=True)
    assert len(br) == 3
    assert not br["is_transport_prior"].any()
    assert set(br["support_flag"]) == {"paired_same_collection"}
    cfg = TransferConfig(require_same_collection=True, run_blocked_cv=False)
    pf = preflight_transfer(_same_panel(), _target(), cfg)
    assert pf.ok
