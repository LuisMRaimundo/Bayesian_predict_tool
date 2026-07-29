"""Paired-corpus assessment + optional PyMC heteroscedastic M3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from string_technique_transfer.bridge import build_log_ratios
from string_technique_transfer.models.fit import fit_model
from string_technique_transfer.paired_corpus import assess_paired_corpus


def _paired_panel(n_midi: int = 14):
    rows = []
    for i, midi in enumerate(np.linspace(55, 85, n_midi)):
        yo = 12 + 0.04 * (midi - 70)
        yt = yo * 0.82
        for tech, y, ord_flag in (("ordinario", yo, True), ("con_sordino", yt, False)):
            rows.append(
                dict(
                    instrument="Violin",
                    collection="paired_lab",
                    technique=tech,
                    dynamic="f",
                    midi=float(midi),
                    metric="EWSD_score_acoustic_balanced",
                    value=float(y),
                    ci_low=float(y * 0.9),
                    ci_high=float(y * 1.1),
                    corpus_id="Violin|paired_lab",
                    is_ordinario=ord_flag,
                )
            )
    return pd.DataFrame(rows)


def _transport_panel(n_midi: int = 10):
    rows = []
    for i, midi in enumerate(np.linspace(55, 85, n_midi)):
        yo = 12.0
        yt = 10.0
        rows.append(
            dict(
                instrument="Violin",
                collection="ord",
                technique="ordinario",
                dynamic="f",
                midi=float(midi),
                metric="EWSD_score_acoustic_balanced",
                value=yo,
                ci_low=yo * 0.9,
                ci_high=yo * 1.1,
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
                midi=float(midi),
                metric="EWSD_score_acoustic_balanced",
                value=yt,
                ci_low=yt * 0.9,
                ci_high=yt * 1.1,
                corpus_id="Violin|sord",
                is_ordinario=False,
            )
        )
    return pd.DataFrame(rows)


def test_paired_corpus_assessment_tiers():
    br = build_log_ratios(_paired_panel(), require_same_collection=True)
    rep = assess_paired_corpus(br)
    assert rep.scientific_tier == "paired"
    assert rep.paired_fraction == 1.0
    assert rep.is_paired_ready

    br_tr = build_log_ratios(_transport_panel(), require_same_collection=False)
    rep_tr = assess_paired_corpus(br_tr)
    assert rep_tr.scientific_tier == "transport_only"
    assert not rep_tr.is_paired_ready


def test_m3_refuses_transport_only_when_paired_required():
    br = build_log_ratios(_transport_panel(), require_same_collection=False)
    with pytest.raises(RuntimeError, match="paired"):
        fit_model(
            br,
            model_id="M3_hierarchical_bayes",
            allow_m3_approx_fallback=False,
            require_paired_corpus_for_m3=True,
        )


@pytest.mark.slow
def test_m3_pymc_heteroscedastic_on_paired_bridge():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    from string_technique_transfer.models.fit import _fit_m2
    from string_technique_transfer.models.m3_pymc import fit_m3_heteroscedastic

    br = build_log_ratios(_paired_panel(16), require_same_collection=True, winsor_q=0.05)
    extra = br.copy()
    extra["technique"] = "sul_ponticello"
    extra["log_ratio"] = np.log(1.25)
    extra["log_ratio_raw"] = extra["log_ratio"]
    extra["factor"] = 1.25
    extra["y_technique"] = extra["y_ordinario"] * 1.25
    br2 = pd.concat([br, extra], ignore_index=True)

    fit = fit_m3_heteroscedastic(
        br2,
        "EWSD_score_acoustic_balanced",
        apply_acoustic_prior=True,
        m2_fallback_fitter=_fit_m2,
        draws=40,
        tune=40,
        chains=1,
    )
    assert fit.backend == "pymc_heteroscedastic_student_t"
    assert fit.diagnostics.get("is_bayesian") is True
    assert fit.diagnostics.get("observation_se_in_likelihood") is True
    assert "SE" in fit.diagnostics.get("likelihood", "") or "se_log_obs" in fit.params.get(
        "likelihood", ""
    )
