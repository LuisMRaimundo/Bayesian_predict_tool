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


def _bridge_same_collection(n: int = 14) -> pd.DataFrame:
    """Strict-mode compatible: ordinario + technique share collection/corpus."""
    rows = []
    rng = np.random.default_rng(0)
    for i, midi in enumerate(np.linspace(55, 85, n)):
        yo = 12 + 0.05 * (midi - 70) + rng.normal(0, 0.2)
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


def _bridge_cross_collection(n: int = 14) -> pd.DataFrame:
    """Transport-mode only: different collections for ordinario vs technique."""
    rows = []
    rng = np.random.default_rng(1)
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
        model_id="M1_register_dynamic",
        require_same_collection=True,
        run_model_comparison=True,
        run_calibration=True,
        run_holdout=True,
        run_sensitivity=False,
        run_blocked_cv=True,
    )
    fit, bridge, preds, path, pf, cv = run_transfer(
        _bridge_same_collection(),
        _target(),
        config=cfg,
        output_xlsx=out,
        skip_preflight=True,
    )
    assert not bridge["is_transport_prior"].any()
    n_sup = int(preds["support_level"].isin(["supported", "supported_outlier_target"]).sum())
    print(f"OK smoke (same-collection): out={path} supported={n_sup}/{len(preds)} model={fit.model_id}")
    print(
        f"recommended={fit.diagnostics.get('recommended_model')} "
        f"cv={cv.iloc[0].to_dict() if len(cv) else {}}"
    )

    # Explicit transport path (must opt in)
    cfg_tr = TransferConfig(
        model_id="M1_register_dynamic",
        require_same_collection=False,
        run_model_comparison=False,
        run_calibration=False,
        run_holdout=False,
        run_sensitivity=False,
        run_blocked_cv=False,
    )
    fit2, br2, preds2, _, _, _ = run_transfer(
        _bridge_cross_collection(),
        _target(),
        config=cfg_tr,
        output_xlsx=None,
        skip_preflight=True,
    )
    assert br2["is_transport_prior"].all()
    print(
        f"OK smoke (transport_prior): n_bridge={len(br2)} "
        f"transport_frac={float(br2['is_transport_prior'].mean()):.2f} model={fit2.model_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
