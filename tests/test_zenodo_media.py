from pathlib import Path

import pandas as pd

from string_technique_transfer.io.loaders import (
    find_zenodo_media_sheet,
    load_zenodo_media_ordinario,
)


def test_load_violin_media_column_o(tmp_path: Path):
    path = tmp_path / "zenodo_fake.xlsx"
    # Mirror Violin_Media layout: A=Note … O(index 14)=Media ff
    cols = [f"c{i}" for i in range(15)]
    cols[0] = "Note"
    cols[12] = "Media pp"
    cols[13] = "Media mf"
    cols[14] = "Media ff"
    df = pd.DataFrame(
        [
            ["G3", *([0.0] * 11), 50.0, 60.0, 70.62352825418856],
            ["Ab3", *([0.0] * 11), 30.0, 35.0, 38.5543055427018],
        ],
        columns=cols,
    )
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Violin_Media", index=False)
        # decoy ORCH sheet (should be ignored for MEDIA loader)
        pd.DataFrame(
            {
                "Source note": ["G3"],
                "Combined density metric": [87.9],
                "Dynamic": ["ff"],
                "Collection": ["ORCH"],
            }
        ).to_excel(xl, sheet_name="Violin__ORCH_ff", index=False)

    assert find_zenodo_media_sheet(path, "Violin") == "Violin_Media"
    out = load_zenodo_media_ordinario(path, instrument="Violin")
    ff = out[out["dynamic"] == "ff"].reset_index(drop=True)
    assert len(ff) == 2
    assert ff.loc[0, "source_column"] == "Media ff"
    assert abs(float(ff.loc[0, "value"]) - 70.62352825418856) < 1e-9
    assert float(ff.loc[0, "value"]) != 87.9
