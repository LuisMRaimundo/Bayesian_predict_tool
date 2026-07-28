import numpy as np
import pandas as pd

from string_technique_transfer.bridge import build_log_ratios
from string_technique_transfer.config import TransferConfig
from string_technique_transfer.pipeline import run_transfer
from string_technique_transfer.validation.calibration import calibrate_from_bridge
from string_technique_transfer.validation.compare import compare_models, recommended_model_id
from string_technique_transfer.validation.holdout import holdout_bridge_validation
from string_technique_transfer.validation.sensitivity import sensitivity_grid


def _synth_bridge(n=16):
    rows = []
    midis = np.linspace(55, 85, n)
    for i, midi in enumerate(midis):
        yo = 12 + 0.05 * (midi - 70) + np.random.default_rng(i).normal(0, 0.3)
        yt = yo * 0.82
        for tech, y, ord_flag in (
            ("ordinario", yo, True),
            ("con_sordino", yt, False),
        ):
            rows.append(
                dict(
                    instrument="Violin",
                    collection="lab",
                    technique=tech,
                    dynamic="f",
                    midi=float(midi),
                    note=f"N{i}",
                    metric="EWSD_score_acoustic_balanced",
                    value=float(y),
                    ci_low=float(y * 0.9),
                    ci_high=float(y * 1.1),
                    corpus_id="Violin|lab",
                    is_ordinario=ord_flag,
                )
            )
    return pd.DataFrame(rows)


def _synth_target():
    rows = []
    for dyn in ("pp", "mf", "ff"):
        for midi in range(55, 101):
            y = 15 + 0.04 * (midi - 70)
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
                    corpus_id="Violin|ORCH",
                    is_ordinario=True,
                )
            )
    return pd.DataFrame(rows)


def test_compare_and_recommend():
    bridge_panel = _synth_bridge(20)
    br = build_log_ratios(bridge_panel, require_same_collection=False)
    tab = compare_models(br, model_ids=("M0_global_factor", "M1_register_dynamic", "M2_midi_gam"))
    assert len(tab) == 3
    assert "recommended" in tab.columns
    mid = recommended_model_id(tab)
    assert mid in {"M0_global_factor", "M1_register_dynamic", "M2_midi_gam"}


def test_calibration_and_holdout():
    bridge_panel = _synth_bridge(20)
    br = build_log_ratios(bridge_panel, require_same_collection=False)
    calib = calibrate_from_bridge(br, model_id="M2_midi_gam")
    assert calib.status in {"ok", "default_priors"}
    detail, summary = holdout_bridge_validation(br, model_id="M2_midi_gam", holdout_frac=0.25)
    assert len(summary) == 1
    assert summary.iloc[0]["status"] in {"ok", "skipped"}


def test_sensitivity_grid_runs():
    panel = _synth_bridge(16)
    sens = sensitivity_grid(panel, model_id="M2_midi_gam", require_same_collection=False)
    assert len(sens) == 4
    assert set(sens["winsor"]) == {True, False}


def test_pipeline_with_validation_pack(tmp_path):
    cfg = TransferConfig(
        strict_dynamics=True,
        model_id="M2_midi_gam",
        run_blocked_cv=True,
        run_model_comparison=True,
        run_calibration=True,
        run_holdout=True,
        run_sensitivity=False,  # keep test light
        auto_select_model=False,
    )
    out = tmp_path / "val.xlsx"
    fit, br, preds, path, pf, cv = run_transfer(
        _synth_bridge(18),
        _synth_target(),
        config=cfg,
        output_xlsx=out,
        skip_preflight=True,
    )
    assert path is not None and path.exists()
    xl = pd.ExcelFile(path)
    assert "Calibration" in xl.sheet_names or "Blocked_CV" in xl.sheet_names
    assert "Model_comparison" in xl.sheet_names
    assert fit.diagnostics.get("recommended_model")
