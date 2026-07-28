import numpy as np
import pandas as pd

from string_technique_transfer.config import TransferConfig
from string_technique_transfer.pipeline import run_transfer
from string_technique_transfer.preflight import preflight_transfer


def _synth_bridge(n=16):
    rows = []
    midis = np.linspace(55, 85, n)
    for i, midi in enumerate(midis):
        yo = 12 + 0.05 * (midi - 70) + np.random.default_rng(i).normal(0, 0.3)
        yt = yo * 0.82
        rows.append(
            dict(
                instrument="Violin",
                collection="ord_corp",
                technique="ordinario",
                dynamic="f",
                midi=float(midi),
                note=f"N{i}",
                metric="EWSD_score_acoustic_balanced",
                value=float(yo),
                ci_low=float(yo * 0.9),
                ci_high=float(yo * 1.1),
                corpus_id="Violin|ord_corp",
                is_ordinario=True,
            )
        )
        rows.append(
            dict(
                instrument="Violin",
                collection="sord_corp",
                technique="con_sordino",
                dynamic="f",
                midi=float(midi),
                note=f"N{i}",
                metric="EWSD_score_acoustic_balanced",
                value=float(yt),
                ci_low=float(yt * 0.9),
                ci_high=float(yt * 1.1),
                corpus_id="Violin|sord_corp",
                is_ordinario=False,
            )
        )
    return pd.DataFrame(rows)


def _synth_target():
    rows = []
    for dyn in ("pp", "mf", "ff"):
        for midi in range(55, 101):
            y = 15 + 0.04 * (midi - 70)
            if midi == 55 and dyn == "ff":
                y = 90  # outlier
            rows.append(
                dict(
                    instrument="Violin",
                    collection="ORCH",
                    technique="ordinario",
                    dynamic=dyn,
                    midi=float(midi),
                    note=f"M{midi}",
                    metric="EWSD_score_acoustic_balanced",
                    value=float(y),
                    ci_low=np.nan,
                    ci_high=np.nan,
                    corpus_id="Violin|ORCH",
                    is_ordinario=True,
                )
            )
    return pd.DataFrame(rows)


def test_preflight_and_supported_only_ff():
    bridge = _synth_bridge()
    target = _synth_target()
    cfg = TransferConfig(strict_dynamics=True, model_id="M2_midi_gam", run_blocked_cv=True)
    pf = preflight_transfer(bridge, target, cfg)
    assert pf.ok
    assert "ff" in pf.summary.get("supported_zenodo[con_sordino]", "")

    fit, br, preds, out, pf2, cv = run_transfer(
        bridge, target, config=cfg, output_xlsx=None, skip_preflight=True
    )
    assert fit.model_id == "M2_midi_gam"
    supported = preds[preds["support_level"].isin(["supported", "supported_outlier_target"])]
    assert set(supported["dynamic"].unique()) == {"ff"}
    assert supported["factor"].median() < 1.0
    assert len(cv) >= 1
