"""Convert the violin EWSD panel CSV into a bridge-ready table for the tool."""

from pathlib import Path
import pandas as pd

PANEL = Path(r"C:\Users\lmr20\Desktop\Strings\Strings\violin\_EWSD_comparison_analysis\ewsd_panel_all_conditions.csv")
OUT = Path(__file__).resolve().parents[1] / "data" / "violin_bridge_panel.csv"

TECH_MAP = {
    "arco_normal_15": "ordinario",
    "arco_normal_1": "ordinario",
    "con_sord": "con_sordino",
    "sul_ponticello": "sul_ponticello",
    "sul_tasto": "sul_tasto",
    "harmonics_natural": "natural_harmonics",
    "harmonics_artificial": "artificial_harmonics",
}


def main() -> None:
    df = pd.read_csv(PANEL)
    out = pd.DataFrame(
        {
            "instrument": "Violin",
            "collection": df["corpus"].astype(str),
            "technique": df["family"].map(TECH_MAP).fillna(df["family"]),
            "dynamic": df["dynamics"].fillna("unspecified"),
            "midi": df["MIDI"],
            "note": df["Note"],
            "register": df["Register"],
            "metric": "EWSD_score_acoustic_balanced",
            "value": df["EWSD_score_acoustic_balanced"],
            "ci_low": df.get("EWSD_score_acoustic_balanced_ci_low"),
            "ci_high": df.get("EWSD_score_acoustic_balanced_ci_high"),
            "rel_uncertainty": df.get("EWSD_score_acoustic_balanced_rel_uncertainty"),
            "source_file": df.get("source_file"),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} rows={len(out)}")


if __name__ == "__main__":
    main()
