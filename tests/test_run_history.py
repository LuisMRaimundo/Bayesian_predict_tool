from pathlib import Path

import numpy as np
import pandas as pd

from string_technique_transfer.config import TransferConfig
from string_technique_transfer.pipeline import run_transfer


def _bridge(n=12):
    rows = []
    for i, midi in enumerate(np.linspace(55, 85, n)):
        yo = 12 + 0.05 * (midi - 70)
        yt = yo * 0.82
        for tech, y, coll, ord_flag in (
            ("ordinario", yo, "ord", True),
            ("con_sordino", yt, "sord", False),
        ):
            rows.append(
                dict(
                    instrument="Violin",
                    collection=coll,
                    technique=tech,
                    dynamic="f",
                    midi=float(midi),
                    note=f"N{i}",
                    metric="EWSD_score_acoustic_balanced",
                    value=float(y),
                    corpus_id=f"Violin|{coll}",
                    is_ordinario=ord_flag,
                )
            )
    return pd.DataFrame(rows)


def _target():
    rows = []
    for dyn in ("pp", "mf", "ff"):
        for midi in range(55, 90):
            rows.append(
                dict(
                    instrument="Violin",
                    collection="MEDIA",
                    technique="ordinario",
                    dynamic=dyn,
                    midi=float(midi),
                    note=f"M{midi}",
                    metric="EWSD_score_acoustic_balanced",
                    value=float(15 + 0.04 * (midi - 70)),
                    corpus_id="Violin|MEDIA",
                    is_ordinario=True,
                )
            )
    return pd.DataFrame(rows)


def test_run_history_written(tmp_path):
    hist = tmp_path / "run_history"
    cfg = TransferConfig(
        model_id="M2_midi_gam",
        run_blocked_cv=False,
        run_model_comparison=False,
        run_calibration=False,
        run_holdout=False,
        run_sensitivity=False,
    )
    out = tmp_path / "out.xlsx"
    fit, br, preds, path, pf, cv = run_transfer(
        _bridge(),
        _target(),
        config=cfg,
        output_xlsx=out,
        skip_preflight=True,
        run_meta={
            "kind": "transfer",
            "bridge_paths": ["dummy_bridge.xlsx"],
            "target_path": "dummy_target.xlsx",
            "instrument": "Violin",
            "zenodo_collection": "MEDIA",
            "history_root": str(hist),
        },
    )
    assert "run_history_report" in fit.diagnostics
    report = Path(fit.diagnostics["run_history_report"])
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Run report" in text
    assert "dummy_bridge.xlsx" in text or "Bridge file" in text
    assert (hist / "INDEX.md").exists()
    assert (hist / "index.csv").exists()
