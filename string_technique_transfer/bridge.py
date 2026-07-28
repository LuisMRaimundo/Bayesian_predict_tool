"""Build technique-to-ordinario log-ratio bridge observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .acoustics import shrink_log_ratio
from .dynamics import MAX_ADEQUATE_DISTANCE, is_adequate_dynamic_pair, nearest_dynamic
from .schema import is_ordinario


def _log_se_from_ci(value, lo, hi) -> float:
    if pd.isna(lo) or pd.isna(hi) or value is None or value <= 0 or lo <= 0 or hi <= 0:
        return np.nan
    return float((np.log(hi) - np.log(lo)) / (2 * 1.96))


def build_log_ratios(
    panel: pd.DataFrame,
    metric: str = "EWSD_score_acoustic_balanced",
    require_same_collection: bool = True,
    max_dynamic_distance: int = MAX_ADEQUATE_DISTANCE,
    winsor_q: float = 0.05,
) -> pd.DataFrame:
    """Paired bridge rows: δ = log(Y_technique / Y_ordinario).

    Prefer same instrument + collection + dynamic + midi. Uses nearest adequate
    dynamic only within max_dynamic_distance. Extreme ratios are winsorized and
    lightly shrunk toward acoustic priors.
    """
    df = panel.loc[panel["metric"] == metric].copy()
    if df.empty:
        df = panel.copy()
        df["metric"] = metric

    ord_df = df.loc[df["technique"].map(is_ordinario)].copy()
    tech_df = df.loc[~df["technique"].map(is_ordinario)].copy()
    if ord_df.empty or tech_df.empty:
        found = sorted(df["technique"].dropna().astype(str).unique().tolist())
        n_ord, n_tech = len(ord_df), len(tech_df)
        raise ValueError(
            "Need both ordinario and special-technique rows to build bridge ratios.\n"
            f"Found techniques: {found}\n"
            f"ordinario rows={n_ord}, special-technique rows={n_tech}.\n"
            "Tip: include at least one Arco_Normal / ordinario workbook plus a special technique "
            "(e.g. sordina_*, *ponticello*, tasto, harmonics)."
        )

    def _pick_ordinario_dynamic(tech_dyn: str, avail: list[str]) -> tuple[str, str, float | None] | None:
        """Choose an ordinario dynamic for a technique row under the adequacy policy."""
        avail = [str(a).lower() for a in avail if str(a).lower() not in {"", "nan", "none"}]
        if not avail:
            return None
        td = str(tech_dyn).strip().lower()
        # Folder stems like tasto.xlsx often lack a dynamic label.
        # Pair with available ordinario at the same MIDI; prefer quieter levels first
        # (common for sul tasto / sordina research folders), then mf.
        if td in {"unspecified", "unknown", "nan"}:
            for cand in ("pp", "p", "mp", "mf", "f", "ff"):
                if cand in avail:
                    return cand, f"technique_dynamic_unspecified->{cand}", 0.0
            return avail[0], f"technique_dynamic_unspecified->{avail[0]}", 0.0
        if td in avail:
            return td, "dynamic_exact", 0.0
        adequate = [d for d in avail if is_adequate_dynamic_pair(td, d, max_dynamic_distance)]
        if not adequate:
            return None
        dyn_used, dyn_flag, dyn_dist = nearest_dynamic(td, adequate)
        return dyn_used, dyn_flag, dyn_dist

    rows = []
    n_skip_no_midi = 0
    n_skip_dyn = 0
    for _, r in tech_df.iterrows():
        base = ord_df[(ord_df["instrument"] == r["instrument"]) & (ord_df["midi"] == r["midi"])]
        if require_same_collection:
            base_strict = base[base["collection"] == r["collection"]]
        else:
            base_strict = base

        support = "paired_same_collection"
        tech_dyn = str(r.get("dynamic", "unspecified"))
        paired = pd.DataFrame()
        dyn_used, dyn_flag, dyn_dist = tech_dyn, "dynamic_exact", 0.0

        if not base_strict.empty:
            avail = base_strict["dynamic"].dropna().astype(str).unique().tolist()
            pick = _pick_ordinario_dynamic(tech_dyn, avail)
            if pick is not None:
                dyn_used, dyn_flag, dyn_dist = pick
                paired = base_strict[base_strict["dynamic"].astype(str).str.lower() == str(dyn_used).lower()]
                if "unspecified" in dyn_flag:
                    support = "unspecified_technique_dynamic"
                elif dyn_flag != "dynamic_exact":
                    support = "nearest_dynamic_ordinario"
            else:
                n_skip_dyn += 1

        if paired.empty:
            # relax collection, still require adequate / unspecified policy
            loose = ord_df[(ord_df["instrument"] == r["instrument"]) & (ord_df["midi"] == r["midi"])]
            if loose.empty:
                n_skip_no_midi += 1
                continue
            avail = loose["dynamic"].dropna().astype(str).unique().tolist()
            pick = _pick_ordinario_dynamic(tech_dyn, avail)
            if pick is None:
                n_skip_dyn += 1
                continue
            dyn_used, dyn_flag, dyn_dist = pick
            paired = loose[loose["dynamic"].astype(str).str.lower() == str(dyn_used).lower()]
            support = (
                "pooled_unspecified_technique_dynamic"
                if "unspecified" in dyn_flag
                else "pooled_ordinario_same_instrument"
            )

        if paired.empty:
            continue

        y_ord = float(paired["value"].median())
        y_t = float(r["value"])
        if y_ord <= 0 or y_t <= 0 or not np.isfinite(y_ord) or not np.isfinite(y_t):
            continue

        se_t = _log_se_from_ci(y_t, r.get("ci_low"), r.get("ci_high"))
        se_o = np.nan
        if {"ci_low", "ci_high"}.issubset(paired.columns):
            se_o = float(
                np.nanmedian(
                    [
                        _log_se_from_ci(v, lo, hi)
                        for v, lo, hi in zip(paired["value"], paired["ci_low"], paired["ci_high"])
                    ]
                )
            )
        se_obs = np.sqrt(
            np.nansum([se_t**2 if np.isfinite(se_t) else 0.0, se_o**2 if np.isfinite(se_o) else 0.0])
        )
        if se_obs == 0:
            se_obs = np.nan

        log_ratio = float(np.log(y_t / y_ord))
        # Label bridge row with the matched ordinario dynamic so Zenodo support mapping works
        # (tasto.xlsx had dynamic=unspecified).
        rows.append(
            {
                "instrument": r["instrument"],
                "bridge_collection": r["collection"],
                "corpus_id": r["corpus_id"],
                "technique": r["technique"],
                "dynamic": dyn_used,
                "technique_dynamic_raw": tech_dyn,
                "ordinario_dynamic_used": dyn_used,
                "dynamic_match": dyn_flag,
                "dynamic_distance": dyn_dist if dyn_dist is not None else np.nan,
                "midi": float(r["midi"]),
                "note": r.get("note"),
                "register": r.get("register"),
                "metric": metric,
                "y_technique": y_t,
                "y_ordinario": y_ord,
                "log_ratio_raw": log_ratio,
                "log_ratio": log_ratio,
                "factor": float(y_t / y_ord),
                "se_log_obs": se_obs,
                "support_flag": support,
                "is_transport_prior": support
                not in {"paired_same_collection", "unspecified_technique_dynamic"},
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        tech_dyns = sorted(tech_df["dynamic"].dropna().astype(str).unique().tolist())
        ord_dyns = sorted(ord_df["dynamic"].dropna().astype(str).unique().tolist())
        raise ValueError(
            "No usable bridge log-ratios could be formed under the adequate-dynamic policy.\n"
            "Need overlapping MIDI with ordinario at a nearby dynamic "
            f"(max distance {max_dynamic_distance} on pp<p<mp<mf<f<ff).\n"
            f"Special-technique dynamics found: {tech_dyns}\n"
            f"Ordinario dynamics found: {ord_dyns}\n"
            f"Skipped (no MIDI match)={n_skip_no_midi}, skipped (inadequate dynamic)={n_skip_dyn}.\n"
            "Tip: if the special-technique file has no dynamic in the name (e.g. tasto.xlsx), "
            "the tool now maps unspecified→available ordinario (pp/p/…); "
            "also ensure ordinario workbooks share notes with the technique file."
        )

    # Winsorize per technique, then shrink toward acoustic prior
    cleaned = []
    for tech, g in out.groupby("technique"):
        g = g.copy()
        lo = g["log_ratio_raw"].quantile(winsor_q)
        hi = g["log_ratio_raw"].quantile(1 - winsor_q)
        g["log_ratio"] = g["log_ratio_raw"].clip(lo, hi)
        g["log_ratio"] = g["log_ratio"].map(lambda x: shrink_log_ratio(float(x), str(tech), n_eff=1.0))
        g["factor"] = np.exp(g["log_ratio"])
        g["winsor_lo"] = lo
        g["winsor_hi"] = hi
        cleaned.append(g)
    out = pd.concat(cleaned, ignore_index=True)
    return out.reset_index(drop=True)


def summarize_factors(bridge: pd.DataFrame) -> pd.DataFrame:
    g = (
        bridge.groupby(["technique", "dynamic"], dropna=False)["factor"]
        .agg(
            n="count",
            median_factor="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            median_log_ratio=lambda s: np.log(s).median(),
        )
        .reset_index()
    )
    return g.sort_values(["technique", "dynamic"])
