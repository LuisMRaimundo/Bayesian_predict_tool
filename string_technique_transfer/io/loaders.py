"""Load bridge/target tables from CSV or Excel into the canonical schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..dynamics import ZENODO_DYNAMICS, detect_zenodo_style_sheets
from ..schema import normalize_dynamic, normalize_technique, standardize_frame
from .path_infer import infer_from_path


def load_panel(path: str | Path, default_metric: str = "EWSD_score_acoustic_balanced") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        raw = pd.read_csv(path, sep=sep)
        std = standardize_frame(raw, default_metric=default_metric)
        return _apply_path_inference(std, path)

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        # Prefer Spectral_Density_Metrics if present; else first sheet / long panel
        xl = pd.ExcelFile(path)
        if "Spectral_Density_Metrics" in xl.sheet_names:
            raw = pd.read_excel(path, sheet_name="Spectral_Density_Metrics")
            std = standardize_frame(raw, default_metric=default_metric)
            std = _apply_path_inference(std, path)
            if (std["instrument"] == "unknown").all() and "Instrument" in raw.columns:
                std["instrument"] = raw["Instrument"].astype(str)
            std["corpus_id"] = std["instrument"].astype(str) + "|" + std["collection"].astype(str)
            std["is_ordinario"] = std["technique"].map(lambda t: t == "ordinario")
            std["source_file"] = str(path)
            return std
        # already long / panel-like
        raw = pd.read_excel(path, sheet_name=0)
        std = standardize_frame(raw, default_metric=default_metric)
        return _apply_path_inference(std, path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def _apply_path_inference(std: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Fill/override technique+dynamic from folder/file names for research workbooks."""
    meta = infer_from_path(path)
    # Always prefer path inference for research xlsx stems (Arco_Normal_forte, sordina_forte…)
    # when technique is unknown OR still carries a raw stem-like label.
    tech_now = std["technique"].astype(str)
    needs_tech = tech_now.isin(["unknown", "nan"]) | tech_now.str.contains(
        r"arco_normal|sordina|ponticello|tasto|harmonic", case=False, na=True
    )
    if needs_tech.all() or (std["technique"] == "unknown").all():
        std["technique"] = meta["technique"]
    if (std["dynamic"].astype(str).isin(["unspecified", "unknown", "nan"])).all() and meta["dynamic"] != "unspecified":
        std["dynamic"] = meta["dynamic"]
    elif needs_tech.all() and meta["dynamic"] != "unspecified":
        # path stem often encodes the true dynamic for these exports
        std["dynamic"] = meta["dynamic"]
    if (std["collection"].astype(str).isin(["unknown", "nan"])).all():
        std["collection"] = meta["collection"]
    if (std["instrument"].astype(str).isin(["unknown", "nan"])).all() and meta["instrument"] != "unknown":
        std["instrument"] = meta["instrument"]
    std["technique"] = std["technique"].map(normalize_technique)
    std["dynamic"] = std["dynamic"].map(normalize_dynamic)
    std["is_ordinario"] = std["technique"].map(lambda t: t == "ordinario")
    std["corpus_id"] = std["instrument"].astype(str) + "|" + std["collection"].astype(str)
    return std


def load_research_workbook(
    path: str | Path,
    *,
    instrument: str,
    collection: str,
    technique: str,
    dynamic: str = "unspecified",
    metric: str = "EWSD_score_acoustic_balanced",
) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="Spectral_Density_Metrics")
    std = standardize_frame(raw, default_metric=metric)
    std["instrument"] = instrument
    std["collection"] = collection
    std["technique"] = normalize_technique(technique)
    std["dynamic"] = normalize_dynamic(dynamic)
    std["metric"] = metric
    std["is_ordinario"] = std["technique"] == "ordinario"
    std["corpus_id"] = f"{instrument}|{collection}"
    std["source_file"] = str(path)
    return std


def resolve_zenodo_sheet(path: str | Path, sheet_name: str) -> str:
    """Fuzzy-resolve sheet names (Violin_ORCH_ff → Violin__ORCH_ff)."""
    path = Path(path)
    names = pd.ExcelFile(path).sheet_names
    if sheet_name in names:
        return sheet_name
    want = sheet_name.strip().lower().replace(" ", "").replace("-", "_")
    for n in names:
        got = n.strip().lower().replace(" ", "").replace("-", "_")
        if got == want or got.replace("__", "_") == want.replace("__", "_"):
            return n
    # soft contains match on orch/iowa + dynamic token
    for n in names:
        got = n.lower().replace(" ", "")
        if want.replace("__", "_") in got.replace("__", "_") or got.replace("__", "_") in want.replace("__", "_"):
            return n
    raise ValueError(
        f"Sheet '{sheet_name}' not found in {path.name}. Available: {names}"
    )


def discover_zenodo_dynamic_sheets(
    path: str | Path,
    *,
    instrument: str = "Violin",
    collection: str | None = None,
) -> dict[str, list[str]]:
    """Find ordinario sheets grouped by collection; each collection has pp/mf/ff.

    Returns e.g. {'IOWA': ['Violin_IOWA_pp', ...], 'ORCH': ['Violin__ORCH_pp', ...]}.
    """
    path = Path(path)
    names = pd.ExcelFile(path).sheet_names
    instr = instrument.strip().lower().replace(" ", "")
    found: dict[str, dict[str, str]] = {}
    for n in names:
        key = n.lower().replace(" ", "").replace("-", "_")
        if instr and instr not in key.replace("__", "_"):
            # allow sheets without instrument prefix if unambiguous
            if not any(d in key for d in ("_pp", "_mf", "_ff", "pp", "mf", "ff")):
                continue
        coll = None
        if "iowa" in key:
            coll = "IOWA"
        elif "orch" in key:
            coll = "ORCH"
        if coll is None:
            continue
        dyn = None
        for token, code in (("_pp", "pp"), ("_mf", "mf"), ("_ff", "ff"), ("pp", "pp"), ("mf", "mf"), ("ff", "ff")):
            if key.endswith(token) or f"_{token.lstrip('_')}" in key:
                # prefer explicit trailing dynamic
                if key.endswith("_pp") or key.endswith("pp"):
                    dyn = "pp"
                elif key.endswith("_mf") or key.endswith("mf"):
                    dyn = "mf"
                elif key.endswith("_ff") or key.endswith("ff"):
                    dyn = "ff"
                break
        if dyn is None:
            continue
        found.setdefault(coll, {})[dyn] = n

    if collection:
        coll_u = collection.strip().upper()
        if coll_u not in found:
            raise ValueError(
                f"Collection '{collection}' not found in {path.name}. Available: {sorted(found)}"
            )
        return {coll_u: [found[coll_u][d] for d in ("pp", "mf", "ff") if d in found[coll_u]]}

    return {c: [sheets[d] for d in ("pp", "mf", "ff") if d in sheets] for c, sheets in found.items()}


def load_zenodo_ordinario_collection(
    path: str | Path,
    *,
    collection: str = "ORCH",
    instrument: str = "Violin",
    metric: str = "EWSD_score_acoustic_balanced",
) -> pd.DataFrame:
    """Load all dynamics (pp, mf, ff) for one Zenodo collection."""
    groups = discover_zenodo_dynamic_sheets(path, instrument=instrument, collection=collection)
    sheets = groups.get(collection.strip().upper(), [])
    if not sheets:
        raise ValueError(f"No pp/mf/ff sheets found for collection={collection}")
    frames = [
        load_zenodo_ordinario_sheet(path, sh, instrument=instrument, metric=metric) for sh in sheets
    ]
    out = pd.concat(frames, ignore_index=True)
    return out


def load_zenodo_ordinario_all(
    path: str | Path,
    *,
    instrument: str = "Violin",
    metric: str = "EWSD_score_acoustic_balanced",
    collections: list[str] | None = None,
) -> pd.DataFrame:
    """Load all collections × all three dynamics from a Zenodo ordinario workbook."""
    groups = discover_zenodo_dynamic_sheets(path, instrument=instrument)
    if collections:
        want = {c.strip().upper() for c in collections}
        groups = {k: v for k, v in groups.items() if k in want}
    if not groups:
        raise ValueError(f"No Zenodo dynamic sheets found for instrument={instrument}")
    frames = []
    for coll, sheets in groups.items():
        for sh in sheets:
            frames.append(load_zenodo_ordinario_sheet(path, sh, instrument=instrument, metric=metric))
    return pd.concat(frames, ignore_index=True)


def load_zenodo_ordinario_sheet(
    path: str | Path,
    sheet_name: str,
    *,
    instrument: str = "Violin",
    metric: str = "EWSD_score_acoustic_balanced",
) -> pd.DataFrame:
    """Load a Zenodo-style ordinario sheet (IOWA/ORCH dynamics)."""
    path = Path(path)
    sheet_name = resolve_zenodo_sheet(path, sheet_name)
    raw = pd.read_excel(path, sheet_name=sheet_name)
    # Detect value column
    value_col = None
    for c in raw.columns:
        cl = str(c).lower()
        if "combined density" in cl or cl == "cdm" or "ewsd" in cl:
            value_col = c
            break
    if value_col is None:
        # last numeric-ish column often holds the metric
        for c in reversed(list(raw.columns)):
            if pd.api.types.is_numeric_dtype(raw[c]):
                value_col = c
                break
    if value_col is None:
        raise ValueError(f"No metric column found in sheet {sheet_name}")

    note_col = "Source note" if "Source note" in raw.columns else ("Note" if "Note" in raw.columns else None)
    dyn = raw["Dynamic"].iloc[0] if "Dynamic" in raw.columns else "unspecified"
    # Prefer trailing pp/mf/ff from sheet name (Zenodo triad is authoritative)
    key = sheet_name.lower().replace(" ", "").replace("-", "_")
    for token in ("_pp", "_mf", "_ff"):
        if key.endswith(token) or key.endswith(token.lstrip("_")):
            dyn = token.lstrip("_")
            break
    coll = raw["Collection"].iloc[0] if "Collection" in raw.columns else sheet_name
    # normalize collection labels
    coll_s = str(coll)
    if "iowa" in coll_s.lower() or "iowa" in sheet_name.lower():
        coll_s = "IOWA"
    elif "orch" in coll_s.lower() or "orch" in sheet_name.lower():
        coll_s = "ORCH"

    out = pd.DataFrame(
        {
            "instrument": instrument,
            "collection": coll_s,
            "technique": "ordinario",
            "dynamic": normalize_dynamic(dyn),
            "note": raw[note_col] if note_col else pd.NA,
            "midi": pd.NA,
            "metric": metric,
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
            "ci_low": pd.NA,
            "ci_high": pd.NA,
            "rel_uncertainty": pd.NA,
            "source_file": str(path),
        }
    )
    # MIDI from note names if possible
    out["midi"] = out["note"].map(_note_to_midi)
    out["is_ordinario"] = True
    out["corpus_id"] = out["instrument"] + "|" + out["collection"]
    out["register"] = pd.NA
    return out.dropna(subset=["value"])


_NOTE_BASE = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def _note_to_midi(note) -> float | None:
    if note is None or (isinstance(note, float) and pd.isna(note)):
        return None
    s = str(note).strip().upper().replace("♯", "#").replace("♭", "B")
    # Ab3 / G#3 / A3
    import re

    m = re.fullmatch(r"([A-G])([#B]?)(-?\d+)", s.replace("♭", "B"))
    if not m:
        # try Ab3 style where b is flat
        m = re.fullmatch(r"([A-G])(B?)(-?\d+)", s)
        if not m:
            return None
    name = m.group(1) + (m.group(2) or "")
    octv = int(m.group(3))
    pc = _NOTE_BASE.get(name)
    if pc is None:
        return None
    return float(12 * (octv + 1) + pc)
