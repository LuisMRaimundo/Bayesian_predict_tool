"""End-to-end smoke: synthetic bridge → target → Excel audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from string_technique_transfer.config import TransferConfig
from string_technique_transfer.pipeline import run_transfer


def _bridge(n: int = 14) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for i, midi in enumerate(np.linspace(55, 85, n)):
        yo = 12 + 0.05 * (midi - 70) + rng.normal(0, 0.2)
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
                    ci_low=float(y * 0.9),
                    ci_high=float(y * 1.1),
                    corpus_id=f"Violin|{coll}",
                    is_ordinario=ord_flag,
                )
            )
    return pd.DataFrame(rows)


def _target() -> pd.DataFrame:
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


def main() -> int:
    out = Path("outputs") / "smoke_transfer.xlsx"
    cfg = TransferConfig(
        model_id="M2_midi_gam",
        run_model_comparison=True,
        run_calibration=True,
        run_holdout=True,
        run_sensitivity=False,
        run_blocked_cv=True,
    )
    fit, bridge, preds, path, pf, cv = run_transfer(
        _bridge(), _target(), config=cfg, output_xlsx=out, skip_preflight=True
    )
    n_sup = int(preds["support_level"].isin(["supported", "supported_outlier_target"]).sum())
    print(f"OK smoke: out={path} supported={n_sup}/{len(preds)} model={fit.model_id}")
    print(f"recommended={fit.diagnostics.get('recommended_model')} cv={cv.iloc[0].to_dict() if len(cv) else {}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
