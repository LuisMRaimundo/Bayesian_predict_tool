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

    rows = []
    for _, r in tech_df.iterrows():
        base = ord_df[(ord_df["instrument"] == r["instrument"]) & (ord_df["midi"] == r["midi"])]
        if require_same_collection:
            base_strict = base[base["collection"] == r["collection"]]
        else:
            base_strict = base

        support = "paired_same_collection"
        paired = base_strict[base_strict["dynamic"] == r["dynamic"]]
        dyn_used = str(r["dynamic"])
        dyn_flag = "dynamic_exact"
        dyn_dist = 0.0

        if paired.empty and not base_strict.empty:
            avail = base_strict["dynamic"].dropna().astype(str).unique().tolist()
            # keep only adequate dynamics
            adequate = [d for d in avail if is_adequate_dynamic_pair(str(r["dynamic"]), d, max_dynamic_distance)]
            pool = adequate if adequate else []
            if pool:
                dyn_used, dyn_flag, dyn_dist = nearest_dynamic(str(r["dynamic"]), pool)
                paired = base_strict[base_strict["dynamic"] == dyn_used]
                support = "nearest_dynamic_ordinario"
            else:
                paired = pd.DataFrame()

        if paired.empty:
            # relax collection, still require adequate dynamic distance
            loose = ord_df[(ord_df["instrument"] == r["instrument"]) & (ord_df["midi"] == r["midi"])]
            if loose.empty:
                continue
            avail = loose["dynamic"].dropna().astype(str).unique().tolist()
            adequate = [d for d in avail if is_adequate_dynamic_pair(str(r["dynamic"]), d, max_dynamic_distance)]
            if not adequate:
                continue  # skip acoustically inadequate pairings
            dyn_used, dyn_flag, dyn_dist = nearest_dynamic(str(r["dynamic"]), adequate)
            paired = loose[loose["dynamic"] == dyn_used]
            support = "pooled_ordinario_same_instrument"

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
        rows.append(
            {
                "instrument": r["instrument"],
                "bridge_collection": r["collection"],
                "corpus_id": r["corpus_id"],
                "technique": r["technique"],
                "dynamic": r["dynamic"],
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
                "is_transport_prior": support != "paired_same_collection",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No usable bridge log-ratios could be formed under the adequate-dynamic policy.\n"
            "Need overlapping MIDI with ordinario at a nearby dynamic "
            f"(max distance {max_dynamic_distance} on pp<p<mp<mf<f<ff)."
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
