import numpy as np
import pandas as pd

from string_technique_transfer.acoustics import clip_log_effect, shrink_log_ratio
from string_technique_transfer.bridge import build_log_ratios
from string_technique_transfer.schema import normalize_technique, parse_condition_label


def test_parse_condition_labels():
    assert parse_condition_label("Arco_Normal_forte") == ("ordinario", "f")
    assert parse_condition_label("sordina_forte")[0] == "con_sordino"
    assert normalize_technique("mezzo-forte_sul-ponticello") == "sul_ponticello"


def test_shrink_and_clip_con_sordino():
    shrunk = shrink_log_ratio(np.log(1.5), "con_sordino", n_eff=2)
    assert shrunk < np.log(1.5)
    clipped, was = clip_log_effect(np.log(3.0), "con_sordino")
    assert was
    assert np.exp(clipped) <= 1.05 + 1e-9


def test_build_log_ratios_basic():
    rows = []
    for midi, yo, yt in [(60, 10.0, 8.0), (62, 12.0, 9.0), (64, 11.0, 9.5), (66, 13.0, 10.0)]:
        rows.append(
            {
                "instrument": "Violin",
                "collection": "A",
                "technique": "ordinario",
                "dynamic": "f",
                "midi": midi,
                "metric": "EWSD_score_acoustic_balanced",
                "value": yo,
                "ci_low": yo * 0.9,
                "ci_high": yo * 1.1,
                "corpus_id": "Violin|A",
                "is_ordinario": True,
            }
        )
        rows.append(
            {
                "instrument": "Violin",
                "collection": "B",
                "technique": "con_sordino",
                "dynamic": "f",
                "midi": midi,
                "metric": "EWSD_score_acoustic_balanced",
                "value": yt,
                "ci_low": yt * 0.9,
                "ci_high": yt * 1.1,
                "corpus_id": "Violin|B",
                "is_ordinario": False,
            }
        )
    panel = pd.DataFrame(rows)
    bridge = build_log_ratios(panel, require_same_collection=False)
    assert len(bridge) == 4
    assert (bridge["factor"] < 1.0).all()
