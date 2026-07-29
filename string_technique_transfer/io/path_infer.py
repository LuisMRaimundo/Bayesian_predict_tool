"""Infer instrument / collection / technique / dynamic from research folder paths."""

from __future__ import annotations

from pathlib import Path

from ..schema import normalize_dynamic, normalize_technique, parse_condition_label


def infer_from_path(path: str | Path) -> dict:
    path = Path(path)
    parts_l = [p.lower() for p in path.parts]
    stem = path.stem

    technique, dynamic = parse_condition_label(stem)

    # folder-based technique overrides / reinforcement
    joined = "/".join(parts_l)
    stem_l = stem.lower().replace(" ", "_").replace("-", "_")
    sordina_hint = (
        "con-sord" in joined
        or "con_sord" in joined
        or "sordina" in stem_l
        or "violin+sordina" in joined
        or "+sordina" in joined
    )
    if sordina_hint:
        technique = "con_sordino"
        if dynamic == "unspecified":
            _, dynamic = parse_condition_label(stem)
    elif "ponticello" in joined:
        technique = "sul_ponticello"
        technique2, dyn2 = parse_condition_label(stem)
        if dyn2 != "unspecified":
            dynamic = dyn2
        elif technique2 == "sul_ponticello" and dynamic == "unspecified":
            dynamic = "unspecified"
    elif "/tasto/" in f"/{joined}/" or parts_l[-4:] and "tasto" in parts_l:
        if "tasto" in stem.lower() or any(p == "tasto" for p in parts_l):
            technique = "sul_tasto"
    elif "harmonics" in joined or "harmónicos" in joined or "harmonicos" in joined:
        if "artificiais" in joined or "artificial" in joined:
            technique = "artificial_harmonics"
        else:
            technique = "natural_harmonics"
            _, dyn2 = parse_condition_label(stem)
            if dyn2 != "unspecified":
                dynamic = dyn2
    elif any(
        p.startswith("arco_normal")
        or p.startswith("arco_normale")
        or p == "ordinario"
        or "arco_ordinario" in p
        or p.endswith("_ordinario")
        for p in [stem_l] + parts_l
    ):
        # Orchidea: .../Violin/ordinario/ORCH_arco_Vln_ff/.../ORCH_ff_Arco ordinario.xlsx
        technique = "ordinario"

    # collection = nearest meaningful parent (skip analysis_results / _Sustains)
    collection = "unknown"
    for p in reversed(path.parts[:-1]):
        if p.lower() in {"analysis_results", "_sustains", "sustains", "results"}:
            continue
        collection = p
        break

    # corpus folder one level above technique folder when present (_15_, _1_, con-sord…)
    corpus = collection
    for p in path.parts:
        pl = p.lower()
        if pl in {"_15_", "_1_", "con-sord", "ponticello", "tasto", "harmonics", "iowa", "orch", "orchidea"}:
            corpus = p
            break

    instrument = "unknown"
    for key, name in [
        ("violin", "Violin"),
        ("viola", "Viola"),
        ("cello", "Cello"),
        ("violoncello", "Cello"),
        ("double bass", "Double Bass"),
        ("contrabass", "Double Bass"),
        ("bass", "Double Bass"),
    ]:
        if any(key == p.lower() or key in p.lower() for p in path.parts):
            instrument = name
            break

    return {
        "instrument": instrument,
        "collection": corpus,
        "technique": normalize_technique(technique),
        "dynamic": normalize_dynamic(dynamic),
        "condition_stem": stem,
    }
