"""Regression tests for audit defects in bridge construction."""

import numpy as np
import pandas as pd
import pytest

from string_technique_transfer.bridge import build_log_ratios
from string_technique_transfer.models.fit import fit_model


def _row(instrument, collection, technique, dynamic, midi, value, ordinario):
    return dict(
        instrument=instrument,
        collection=collection,
        technique=technique,
        dynamic=dynamic,
        midi=float(midi),
        note=f"N{midi}",
        metric="EWSD_score_acoustic_balanced",
        value=float(value),
        corpus_id=f"{instrument}|{collection}",
        is_ordinario=ordinario,
    )


def test_cross_collection_not_labelled_same_collection():
    rows = [
        _row("Violin", "ord_lab", "ordinario", "f", 60, 10.0, True),
        _row("Violin", "sord_lab", "con_sordino", "f", 60, 8.0, False),
    ]
    panel = pd.DataFrame(rows)
    br = build_log_ratios(panel, require_same_collection=False)
    assert len(br) == 1
    assert br.iloc[0]["support_flag"] == "transport_prior"
    assert bool(br.iloc[0]["is_transport_prior"]) is True
    assert br.iloc[0]["special_corpus_id"] == "Violin|sord_lab"
    assert br.iloc[0]["ordinario_corpus_id"] == "Violin|ord_lab"


def test_require_same_collection_rejects_cross_collection():
    rows = [
        _row("Violin", "ord_lab", "ordinario", "f", 60, 10.0, True),
        _row("Violin", "sord_lab", "con_sordino", "f", 60, 8.0, False),
    ]
    panel = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="same-collection"):
        build_log_ratios(panel, require_same_collection=True)


def test_unspecified_dynamic_not_invented():
    rows = []
    for midi in (55.0, 56.0):
        rows.append(_row("Violin", "lab", "ordinario", "pp", midi, 20.0, True))
        rows.append(_row("Violin", "lab", "ordinario", "mf", midi, 22.0, True))
        rows.append(_row("Violin", "lab", "sul_tasto", "unspecified", midi, 16.0, False))
    panel = pd.DataFrame(rows)
    br = build_log_ratios(panel, require_same_collection=True)
    assert len(br) == 2
    assert set(br["dynamic"]) == {"unspecified"}
    assert set(br["dynamic_support"]) == {"unknown"}
    assert br["ordinario_dynamic_used"].eq("median_across_dynamics").all()


def test_response_level_prior_not_applied_in_bridge():
    rows = []
    for midi in (60, 62, 64, 66):
        rows.append(_row("Violin", "A", "ordinario", "f", midi, 10.0, True))
        rows.append(_row("Violin", "A", "artificial_harmonics", "f", midi, 8.75, False))
    panel = pd.DataFrame(rows)
    br = build_log_ratios(panel, require_same_collection=True, winsor_q=0.0)
    # Raw median factor 0.875 must survive bridge construction (no κ=4 shrink)
    assert abs(float(br["factor"].median()) - 0.875) < 1e-9
    assert np.allclose(br["log_ratio"], br["log_ratio_raw"])


def test_m3_hard_fails_without_bayes_unless_authorized():
    rows = []
    for tech, coll, ord_flag, scale in (
        ("ordinario", "A", True, 1.0),
        ("con_sordino", "A", False, 0.85),
        ("sul_ponticello", "A", False, 1.2),
    ):
        for i, midi in enumerate(np.linspace(55, 90, 15)):
            rows.append(
                _row("Violin", coll, tech, "f", midi, (12 + 0.05 * i) * scale, ord_flag)
            )
    panel = pd.DataFrame(rows)
    br = build_log_ratios(panel, require_same_collection=True)
    # Force approx path via thin? n is large enough — force no bayes by mocking
    from string_technique_transfer.models import fit as fit_mod

    orig = fit_mod._bayes_stack
    fit_mod._bayes_stack = lambda: (None, None)
    try:
        with pytest.raises(RuntimeError, match="M3 hierarchical Bayes"):
            fit_model(br, model_id="M3_hierarchical_bayes", allow_m3_approx_fallback=False)
        approx = fit_model(br, model_id="M3_hierarchical_bayes", allow_m3_approx_fallback=True)
        assert "approx" in approx.backend
        assert approx.diagnostics.get("is_bayesian") is False
    finally:
        fit_mod._bayes_stack = orig
