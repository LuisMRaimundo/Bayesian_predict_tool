import pandas as pd

from string_technique_transfer.bridge import build_log_ratios


def test_unspecified_technique_dynamic_kept_unspecified():
    rows = []
    for midi in (55.0, 56.0, 58.0):
        rows.append(
            dict(
                instrument="Violin",
                collection="lab",
                technique="ordinario",
                dynamic="pp",
                midi=midi,
                note=f"N{midi}",
                metric="EWSD_score_acoustic_balanced",
                value=20.0,
                corpus_id="Violin|lab",
                is_ordinario=True,
            )
        )
        rows.append(
            dict(
                instrument="Violin",
                collection="lab",
                technique="sul_tasto",
                dynamic="unspecified",
                midi=midi,
                note=f"N{midi}",
                metric="EWSD_score_acoustic_balanced",
                value=16.0,
                corpus_id="Violin|lab",
                is_ordinario=False,
            )
        )
    panel = pd.DataFrame(rows)
    br = build_log_ratios(panel, require_same_collection=True)
    assert len(br) == 3
    assert set(br["dynamic"]) == {"unspecified"}
    assert br["dynamic_support"].eq("unknown").all()
    assert (br["factor"] < 1.0).all()
