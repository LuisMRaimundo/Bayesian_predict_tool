"""Build technique-to-ordinario log-ratio bridge observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dynamics import MAX_ADEQUATE_DISTANCE, is_adequate_dynamic_pair, nearest_dynamic
from .schema import is_ordinario

_UNSPECIFIED = {"unspecified", "unknown", "nan", ""}


def _log_se_from_ci(value, lo, hi) -> float:
    if pd.isna(lo) or pd.isna(hi) or value is None or value <= 0 or lo <= 0 or hi <= 0:
        return np.nan
    return float((np.log(hi) - np.log(lo)) / (2 * 1.96))


def _is_unspecified_dynamic(dyn: str) -> bool:
    return str(dyn).strip().lower() in _UNSPECIFIED


def winsorize_log_ratios(bridge: pd.DataFrame, winsor_q: float = 0.05) -> pd.DataFrame:
    """Winsorize ``log_ratio`` from ``log_ratio_raw`` per technique (train-fold safe)."""
    if bridge is None or len(bridge) == 0:
        return bridge
    out = bridge.copy()
    if "log_ratio_raw" not in out.columns:
        out["log_ratio_raw"] = out["log_ratio"]
    if winsor_q is None or winsor_q <= 0:
        out["log_ratio"] = out["log_ratio_raw"].astype(float)
        out["factor"] = np.exp(out["log_ratio"])
        out["winsor_lo"] = np.nan
        out["winsor_hi"] = np.nan
        return out.reset_index(drop=True)

    cleaned = []
    for _tech, g in out.groupby("technique", dropna=False):
        g = g.copy()
        lo = float(g["log_ratio_raw"].quantile(winsor_q))
        hi = float(g["log_ratio_raw"].quantile(1 - winsor_q))
        g["log_ratio"] = g["log_ratio_raw"].clip(lo, hi)
        g["factor"] = np.exp(g["log_ratio"])
        g["winsor_lo"] = lo
        g["winsor_hi"] = hi
        cleaned.append(g)
    return pd.concat(cleaned, ignore_index=True)


def build_log_ratios(
    panel: pd.DataFrame,
    metric: str = "EWSD_score_acoustic_balanced",
    require_same_collection: bool = True,
    max_dynamic_distance: int = MAX_ADEQUATE_DISTANCE,
    winsor_q: float = 0.05,
) -> pd.DataFrame:
    """Paired bridge rows: δ = log(Y_technique / Y_ordinario).

    Policy (audit-aligned):
    - ``require_same_collection=True``: never fall back to another collection.
    - Cross-collection pairs (only when allowed) are labelled ``transport_prior``.
    - Unspecified technique dynamics stay ``dynamic=unspecified`` (not invented).
    - Acoustic priors are **not** applied to responses here (model-level only).
    - Stores ``special_corpus_id`` and ``ordinario_corpus_id`` separately.
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
        avail = [str(a).lower() for a in avail if not _is_unspecified_dynamic(a)]
        if not avail:
            return None
        td = str(tech_dyn).strip().lower()
        if _is_unspecified_dynamic(td):
            # Baseline for ratio only — does **not** assign a technique dynamic.
            for cand in ("mf", "f", "mp", "p", "pp", "ff"):
                if cand in avail:
                    return cand, "ordinario_baseline_for_unspecified_technique", 0.0
            return avail[0], "ordinario_baseline_for_unspecified_technique", 0.0
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
    n_skip_collection = 0
    for _, r in tech_df.iterrows():
        same_coll = ord_df[
            (ord_df["instrument"] == r["instrument"])
            & (ord_df["midi"] == r["midi"])
            & (ord_df["collection"] == r["collection"])
        ]
        any_coll = ord_df[(ord_df["instrument"] == r["instrument"]) & (ord_df["midi"] == r["midi"])]

        tech_dyn_raw = str(r.get("dynamic", "unspecified"))
        tech_unspecified = _is_unspecified_dynamic(tech_dyn_raw)
        paired = pd.DataFrame()
        dyn_used, dyn_flag, dyn_dist = tech_dyn_raw, "dynamic_exact", 0.0
        same_collection_pair = False

        if not same_coll.empty:
            if tech_unspecified:
                paired = same_coll
                dyn_used = "unspecified"
                dyn_flag = "technique_dynamic_unspecified"
                dyn_dist = np.nan
                same_collection_pair = True
            else:
                avail = same_coll["dynamic"].dropna().astype(str).unique().tolist()
                pick = _pick_ordinario_dynamic(tech_dyn_raw, avail)
                if pick is not None:
                    dyn_used, dyn_flag, dyn_dist = pick
                    paired = same_coll[
                        same_coll["dynamic"].astype(str).str.lower() == str(dyn_used).lower()
                    ]
                    same_collection_pair = True
                else:
                    n_skip_dyn += 1
        elif require_same_collection:
            n_skip_collection += 1
            continue
        else:
            # Explicit cross-collection transport (only when allowed)
            if any_coll.empty:
                n_skip_no_midi += 1
                continue
            if tech_unspecified:
                paired = any_coll
                dyn_used = "unspecified"
                dyn_flag = "technique_dynamic_unspecified"
                dyn_dist = np.nan
            else:
                avail = any_coll["dynamic"].dropna().astype(str).unique().tolist()
                pick = _pick_ordinario_dynamic(tech_dyn_raw, avail)
                if pick is None:
                    n_skip_dyn += 1
                    continue
                dyn_used, dyn_flag, dyn_dist = pick
                paired = any_coll[
                    any_coll["dynamic"].astype(str).str.lower() == str(dyn_used).lower()
                ]
            same_collection_pair = False

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
        special_corpus = str(r.get("corpus_id") or f"{r['instrument']}|{r['collection']}")
        ord_corpus = (
            str(paired["corpus_id"].mode().iloc[0])
            if "corpus_id" in paired.columns and paired["corpus_id"].notna().any()
            else str(f"{paired['instrument'].iloc[0]}|{paired['collection'].iloc[0]}")
        )
        ord_dyn_used = (
            "median_across_dynamics"
            if tech_unspecified
            else str(dyn_used)
        )

        if tech_unspecified:
            support = (
                "same_collection_unspecified_dynamic"
                if same_collection_pair
                else "transport_prior_unspecified_dynamic"
            )
            dynamic_support = "unknown"
            out_dynamic = "unspecified"
            is_transport = not same_collection_pair
        elif same_collection_pair:
            if dyn_flag == "dynamic_exact":
                support = "paired_same_collection"
            else:
                support = "nearest_dynamic_ordinario"
            dynamic_support = "measured" if dyn_flag == "dynamic_exact" else "nearest_adequate"
            out_dynamic = str(tech_dyn_raw).strip().lower()
            is_transport = False
        else:
            support = "transport_prior"
            dynamic_support = "measured" if dyn_flag == "dynamic_exact" else "nearest_adequate"
            out_dynamic = str(tech_dyn_raw).strip().lower()
            is_transport = True

        rows.append(
            {
                "instrument": r["instrument"],
                "bridge_collection": r["collection"],
                "ordinario_collection": paired["collection"].mode().iloc[0]
                if paired["collection"].notna().any()
                else r["collection"],
                "special_corpus_id": special_corpus,
                "ordinario_corpus_id": ord_corpus,
                # Backward-compatible single id (special / technique side)
                "corpus_id": special_corpus,
                "technique": r["technique"],
                "dynamic": out_dynamic,
                "technique_dynamic_raw": tech_dyn_raw,
                "ordinario_dynamic_used": ord_dyn_used,
                "dynamic_match": dyn_flag,
                "dynamic_distance": dyn_dist if dyn_dist is not None else np.nan,
                "dynamic_support": dynamic_support,
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
                "is_transport_prior": is_transport,
                "same_collection_pair": same_collection_pair,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        tech_dyns = sorted(tech_df["dynamic"].dropna().astype(str).unique().tolist())
        ord_dyns = sorted(ord_df["dynamic"].dropna().astype(str).unique().tolist())
        raise ValueError(
            "No usable bridge log-ratios could be formed under the pairing policy.\n"
            f"require_same_collection={require_same_collection}; "
            f"max dynamic distance {max_dynamic_distance} on pp<p<mp<mf<f<ff.\n"
            f"Special-technique dynamics found: {tech_dyns}\n"
            f"Ordinario dynamics found: {ord_dyns}\n"
            f"Skipped (no MIDI match)={n_skip_no_midi}, "
            f"skipped (inadequate dynamic)={n_skip_dyn}, "
            f"skipped (no same-collection ordinario)={n_skip_collection}.\n"
            "Tip: for strict same-collection pairing, include ordinario rows from the same "
            "collection as each special technique; or allow cross-collection transport "
            "(labelled transport_prior) via require_same_collection=False."
        )

    # Winsorize only — acoustic shrink is applied once at the model-coefficient level.
    return winsorize_log_ratios(out, winsor_q=winsor_q)


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
