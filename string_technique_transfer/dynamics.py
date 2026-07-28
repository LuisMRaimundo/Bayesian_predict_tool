"""Dynamic levels and Zenodo 3-dynamic matching (pp / mf / ff).

Zenodo ordinario workbooks for a given instrument×collection always expose
three dynamics. Bridge corpora may use a finer set (p, mp, mf, f, …).

Robust policy: only pairs within a small loudness distance are treated as
supported; larger jumps (e.g. bridge f → Zenodo pp) are extrapolated/unsupported.
"""

from __future__ import annotations

from typing import Iterable

# Ordered loudness scale used for nearest-neighbour matching
DYNAMIC_LEVEL = {
    "pp": 0,
    "p": 1,
    "mp": 2,
    "mf": 3,
    "f": 4,
    "ff": 5,
    "unspecified": None,
}

# Canonical Zenodo ordinario triad per instrument collection
ZENODO_DYNAMICS = ("pp", "mf", "ff")

# Maximum |level| distance used as a coarse filter
MAX_ADEQUATE_DISTANCE = 1

# Explicit acoustically adequate bridge partners for each Zenodo dynamic.
# Intentionally tight: bridge forte supports Zenodo ff, not pp/mf.
ADEQUATE_BRIDGE_FOR_ZENODO = {
    "pp": frozenset({"pp", "p"}),
    "mf": frozenset({"mf", "mp"}),
    "ff": frozenset({"ff", "f"}),
}

# Preferred bridge partner order for each Zenodo dynamic
ZENODO_PREFERRED_BRIDGE = {
    "pp": ("pp", "p"),
    "mf": ("mf", "mp"),
    "ff": ("ff", "f"),
}


def dynamic_level(dyn: str | None) -> float | None:
    if dyn is None:
        return None
    return DYNAMIC_LEVEL.get(str(dyn).strip().lower(), None)


def dynamic_distance(a: str, b: str) -> float | None:
    la, lb = dynamic_level(a), dynamic_level(b)
    if la is None or lb is None:
        return None
    return abs(la - lb)


def is_adequate_dynamic_pair(target_dyn: str, bridge_dyn: str, max_distance: int = MAX_ADEQUATE_DISTANCE) -> bool:
    """Adequate if in the explicit Zenodo↔bridge map, else fall back to distance."""
    z = str(target_dyn).strip().lower()
    b = str(bridge_dyn).strip().lower()
    allowed = ADEQUATE_BRIDGE_FOR_ZENODO.get(z)
    if allowed is not None:
        return b in allowed
    # generic (non-Zenodo) target dynamics: use loudness distance
    d = dynamic_distance(z, b)
    if d is None:
        return False
    return d <= max_distance


def nearest_dynamic(query: str, available: Iterable[str]) -> tuple[str, str, float | None]:
    """Return (best_match, flag, distance)."""
    avail = [
        str(a).lower()
        for a in available
        if str(a).lower() in DYNAMIC_LEVEL and DYNAMIC_LEVEL[str(a).lower()] is not None
    ]
    q = str(query).strip().lower()
    if not avail:
        return "unspecified", "dynamic_unavailable", None
    if q in avail:
        return q, "dynamic_exact", 0.0
    qlev = dynamic_level(q)
    if qlev is None:
        if "mf" in avail:
            return "mf", "dynamic_nearest:unspecified->mf", dynamic_distance("unspecified", "mf")
        return avail[0], f"dynamic_nearest:unspecified->{avail[0]}", None
    best = min(avail, key=lambda a: (abs(DYNAMIC_LEVEL[a] - qlev), DYNAMIC_LEVEL[a]))
    dist = abs(DYNAMIC_LEVEL[best] - qlev)
    return best, f"dynamic_nearest:{q}->{best}", float(dist)


def map_bridge_dynamic_to_zenodo(bridge_dyn: str, zenodo_dynamics: Iterable[str] = ZENODO_DYNAMICS):
    return nearest_dynamic(bridge_dyn, zenodo_dynamics)


def map_zenodo_dynamic_to_bridge(zenodo_dyn: str, bridge_dynamics: Iterable[str]):
    """For a Zenodo target dynamic, pick the most adequate bridge dynamic present."""
    z = str(zenodo_dyn).strip().lower()
    avail = [str(a).lower() for a in bridge_dynamics if str(a).lower() in DYNAMIC_LEVEL]
    for cand in ZENODO_PREFERRED_BRIDGE.get(z, ()):
        if cand in avail and is_adequate_dynamic_pair(z, cand):
            flag = "dynamic_exact" if cand == z else f"dynamic_nearest:{z}->{cand}"
            return cand, flag, float(dynamic_distance(z, cand) or 0.0)
    # No adequate partner: still return nearest for diagnostics, with large distance
    best, flag, dist = nearest_dynamic(z, avail)
    # Force non-adequate distance if not in explicit map
    if not is_adequate_dynamic_pair(z, best):
        dist = 99.0 if dist is None else float(max(dist, 99.0))
        flag = f"dynamic_inadequate:{z}->{best}"
    return best, flag, dist


def supported_zenodo_dynamics_for_bridge(
    bridge_dynamics: Iterable[str],
    zenodo_dynamics: Iterable[str] = ZENODO_DYNAMICS,
    max_distance: int = MAX_ADEQUATE_DISTANCE,
) -> list[str]:
    """Which Zenodo triad members are adequately supported by the bridge dynamics."""
    avail = [str(a).lower() for a in bridge_dynamics]
    out = []
    for z in zenodo_dynamics:
        allowed = ADEQUATE_BRIDGE_FOR_ZENODO.get(z, frozenset())
        if any(a in allowed for a in avail):
            out.append(z)
    return out


def detect_zenodo_style_sheets(sheet_names: Iterable[str]) -> dict[str, dict[str, str]]:
    """Detect {COLLECTION: {dynamic: sheet_name}} from workbook sheet names."""
    import re

    out: dict[str, dict[str, str]] = {}
    pat = re.compile(
        r"^(?P<instr>[A-Za-z]+)[_]+(?P<coll>[A-Za-z]+)[_]+(?P<dyn>pp|mf|ff|p|mp|f)$",
        re.I,
    )
    for name in sheet_names:
        compact = name.strip().replace(" ", "")
        m = pat.match(compact.replace("__", "_"))
        if not m:
            continue
        coll = m.group("coll").upper()
        dyn = m.group("dyn").lower()
        if dyn == "p":
            dyn = "pp"
        if dyn == "f":
            dyn = "ff"
        out.setdefault(coll, {})[dyn] = name
    return out
