"""Canonical note-level schema (instrument-agnostic)."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

REQUIRED = (
    "instrument",
    "collection",
    "technique",
    "dynamic",
    "midi",
    "metric",
    "value",
)

OPTIONAL = (
    "note",
    "register",
    "ci_low",
    "ci_high",
    "rel_uncertainty",
    "source_file",
    "is_ordinario",
    "corpus_id",
    "support_flag",
)

ORDINARIO_ALIASES = {
    "ordinario",
    "arco_normal",
    "arco normale",
    "arco-normal",
    "normal",
    "ord",
    "ordinario_arco",
}

DYNAMIC_ORDER = ("pp", "p", "mp", "mf", "f", "ff", "unspecified")

COLUMN_ALIASES = {
    "instrument": ["instrument", "Instrument", "instr"],
    "collection": ["collection", "Collection", "corpus", "family", "corpus_id"],
    "technique": ["technique", "Technique", "technique/state", "Technique/state", "condition", "family_technique"],
    "dynamic": ["dynamic", "Dynamic", "dynamics"],
    "midi": ["midi", "MIDI", "Midi"],
    "note": ["note", "Note", "Source note", "source_note"],
    "register": ["register", "Register"],
    "metric": ["metric", "Metric", "descriptor"],
    "value": [
        "value",
        "Value",
        "EWSD_score_acoustic_balanced",
        "Combined density metric",
        "Combined Density Metric",
        "CDM",
    ],
    "ci_low": ["ci_low", "EWSD_score_acoustic_balanced_ci_low"],
    "ci_high": ["ci_high", "EWSD_score_acoustic_balanced_ci_high"],
    "rel_uncertainty": ["rel_uncertainty", "EWSD_score_acoustic_balanced_rel_uncertainty"],
    "source_file": ["source_file", "source"],
}


def _first_present(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_dynamic(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unspecified"
    s = str(x).strip().lower().replace(" ", "").replace("_", "-")
    mapping = {
        "pianissimo": "pp",
        "pp": "pp",
        "piano": "p",
        "p": "p",
        "mezzo-piano": "mp",
        "mezzopiano": "mp",
        "mp": "mp",
        "mezzo-forte": "mf",
        "mezzoforte": "mf",
        "mf": "mf",
        "forte": "f",
        "f": "f",
        "fortissimo": "ff",
        "ff": "ff",
        "unknown": "unspecified",
        "none": "unspecified",
        "": "unspecified",
    }
    return mapping.get(s, s if s in DYNAMIC_ORDER else "unspecified")


_DYNAMIC_TOKENS = (
    "pianissimo",
    "fortissimo",
    "mezzo_piano",
    "mezzo_forte",
    "mezzo-piano",
    "mezzo-forte",
    "mp",
    "mf",
    "pp",
    "ff",
    "piano",
    "forte",
)


def _strip_dynamic_suffix(label: str) -> tuple[str, str | None]:
    """Split trailing dynamic from labels like Arco_Normal_forte / sordina_mf."""
    s = label.strip().lower().replace(" ", "_").replace("-", "_")
    # longest-first
    dyn_forms = [
        ("pianissimo", "pp"),
        ("fortissimo", "ff"),
        ("mezzo_piano", "mp"),
        ("mezzo_forte", "mf"),
        ("mezzo-piano", "mp"),
        ("mezzo-forte", "mf"),
        ("_mp", "mp"),
        ("_mf", "mf"),
        ("_pp", "pp"),
        ("_ff", "ff"),
        ("_piano", "p"),
        ("_forte", "f"),
        ("_p", "p"),
        ("_f", "f"),
    ]
    for suf, dyn in dyn_forms:
        if s.endswith(suf):
            core = s[: -len(suf)].rstrip("_")
            return core, dyn
    # leading dynamic: piano_sul_ponticello, mezzo_forte_sul_ponticello
    for pref, dyn in [
        ("mezzo_forte_", "mf"),
        ("mezzo_piano_", "mp"),
        ("pianissimo_", "pp"),
        ("fortissimo_", "ff"),
        ("piano_", "p"),
        ("forte_", "f"),
        ("mf_", "mf"),
        ("mp_", "mp"),
        ("pp_", "pp"),
        ("ff_", "ff"),
        ("p_", "p"),
        ("f_", "f"),
    ]:
        if s.startswith(pref):
            return s[len(pref) :], dyn
    return s, None


def normalize_technique(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unknown"
    raw = str(x).strip().lower().replace(" ", "_").replace("-", "_")
    core, _dyn = _strip_dynamic_suffix(raw)
    # also try original without strip
    candidates = [core, raw]
    aliases = {
        "arco_normal": "ordinario",
        "arco_normale": "ordinario",
        "arco": "ordinario",
        "normal": "ordinario",
        "ord": "ordinario",
        "ordinario": "ordinario",
        "con_sord": "con_sordino",
        "con_sordino": "con_sordino",
        "sordina": "con_sordino",
        "sordino": "con_sordino",
        "sul_ponticello": "sul_ponticello",
        "ponticello": "sul_ponticello",
        "sul_tasto": "sul_tasto",
        "tasto": "sul_tasto",
        "harmonics_natural": "natural_harmonics",
        "natural_harmonics": "natural_harmonics",
        "harmonicos_naturais": "natural_harmonics",
        "harmónicos_naturais": "natural_harmonics",
        "harmonico_naturais": "natural_harmonics",
        "harmonics_artificial": "artificial_harmonics",
        "artificial_harmonics": "artificial_harmonics",
        "harmonicos_artificiais": "artificial_harmonics",
        "harmónicos_artificiais": "artificial_harmonics",
    }
    for c in candidates:
        if c in aliases:
            return aliases[c]
        # substring / startswith helpers for folder-derived labels
        # Mute / special first (names may also contain the word "ordinario")
        if (
            c.startswith("sordina")
            or c.startswith("sordino")
            or "con_sord" in c
            or "_sordina_" in f"_{c}_"
            or c.endswith("_sordina")
            or "sordina_" in c
        ):
            return "con_sordino"
        if "ponticello" in c:
            return "sul_ponticello"
        if c == "tasto" or c.endswith("_tasto") or "sul_tasto" in c:
            return "sul_tasto"
        if "artificiais" in c or "artificial" in c:
            return "artificial_harmonics"
        if "naturais" in c or "natural_harmonic" in c or c.startswith("harmonics_natural"):
            return "natural_harmonics"
        # Ordinario / arco normal — including Orchidea stems like ORCH_ff_Arco_ordinario
        if (
            c.startswith("arco_normal")
            or c.startswith("arco_normale")
            or c == "ordinario"
            or c.endswith("_ordinario")
            or "_ordinario_" in f"_{c}_"
            or "arco_ordinario" in c
            or c.endswith("arco_normal")
            or "arco_normal_" in c
        ):
            return "ordinario"
    return core


def parse_condition_label(label: str) -> tuple[str, str]:
    """Return (technique, dynamic) from a folder/file stem."""
    raw = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    core, dyn = _strip_dynamic_suffix(raw)
    technique = normalize_technique(core if dyn else raw)
    # if normalize already consumed dynamics inside aliases, recover dyn
    if dyn is None:
        _, dyn2 = _strip_dynamic_suffix(raw)
        dyn = dyn2
    return technique, normalize_dynamic(dyn) if dyn else "unspecified"


def is_ordinario(technique: str) -> bool:
    t = normalize_technique(technique)
    return t == "ordinario"


def standardize_frame(df: pd.DataFrame, default_metric: str = "EWSD_score_acoustic_balanced") -> pd.DataFrame:
    """Map heterogeneous tables onto the canonical schema."""
    out = pd.DataFrame(index=df.index.copy())
    for canon, aliases in COLUMN_ALIASES.items():
        src = _first_present(df.columns, aliases)
        if src is not None:
            out[canon] = df[src]

    if "metric" not in out.columns:
        # wide EWSD-style table
        if "EWSD_score_acoustic_balanced" in df.columns:
            out["metric"] = default_metric
            out["value"] = pd.to_numeric(df["EWSD_score_acoustic_balanced"], errors="coerce")
        elif "Combined density metric" in df.columns:
            out["metric"] = default_metric
            out["value"] = pd.to_numeric(df["Combined density metric"], errors="coerce")
        elif "value" not in out.columns:
            raise ValueError("Could not find a metric value column.")

    if "metric" not in out.columns or out["metric"].isna().all():
        out["metric"] = default_metric

    for col, default in [
        ("instrument", "unknown"),
        ("collection", "unknown"),
        ("technique", "unknown"),
        ("dynamic", "unspecified"),
        ("note", None),
        ("register", None),
        ("source_file", None),
    ]:
        if col not in out.columns:
            out[col] = default

    out["technique"] = out["technique"].map(normalize_technique)
    out["dynamic"] = out["dynamic"].map(normalize_dynamic)
    out["midi"] = pd.to_numeric(out.get("midi"), errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    for c in ("ci_low", "ci_high", "rel_uncertainty"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = pd.NA
    out["instrument"] = out["instrument"].astype(str).str.strip()
    out["collection"] = out["collection"].astype(str).str.strip()
    out["is_ordinario"] = out["technique"].map(is_ordinario)
    out["corpus_id"] = out["instrument"].astype(str) + "|" + out["collection"].astype(str)
    return out
